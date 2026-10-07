from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import json
import time
import traceback
import uvicorn

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.graph import (
    run_travel_agent, stream_travel_agent, new_thread_id, get_trip_plan, open_travel_graph, delete_trip_state,
)
from app.db import open_db, close_db
from app.auth import authenticate_user, create_access_token, create_user, get_current_user
from app.trips import create_trip, delete_trip, list_trips, make_title, rename_trip, touch_trip, user_owns_trip

# The frontend lives in <repo>/frontend; FastAPI serves it.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await open_db()
    await open_travel_graph()
    yield
    await close_db()


app = FastAPI(
    title="Roamer AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0",
    lifespan=lifespan
)


app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(FRONTEND_DIR / "templates")
)


# Return errors from HTTPException (e.g. 401 from get_current_user) in the same
# {"success": false, "error": ...} shape as the other responses.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
        headers=exc.headers,
    )



class TripDates(BaseModel):
    """Dates picked in the form. A new trip needs a start and an end date."""
    departure_date: date | None = None
    return_date: date | None = None
    # "exact", "1"/"2"/"3" for ± days around the start date, or "month" for any day
    # in the start date's month (the trip keeps the length of start to end).
    flexibility: Literal["exact", "1", "2", "3", "month"] = "exact"


class TravelRequest(BaseModel):
    message: str = ""
    thread_id: str | None = None
    # Answer to a paused run's flight question: an option number, or 0 for none.
    # Only /api/travel/stream uses it.
    flight_choice: int | None = None
    dates: TripDates | None = None


def check_dates(dates: TripDates | None) -> str | None:
    """An error message for dates that can't be searched, or None if they're fine."""
    if not dates or not dates.departure_date or not dates.return_date:
        return "Pick a start date and an end date."
    if dates.departure_date <= date.today():
        return "The start date must be after today."
    if dates.return_date <= dates.departure_date:
        return "The end date must be after the start date."
    return None


def dates_for_graph(dates: TripDates | None) -> dict:
    if not dates:
        return {}
    return {
        "departure_date": dates.departure_date.isoformat() if dates.departure_date else None,
        "return_date": dates.return_date.isoformat() if dates.return_date else None,
        "flexibility": dates.flexibility,
    }


class RenameRequest(BaseModel):
    title: str


class AuthRequest(BaseModel):
    email: str
    password: str


MIN_PASSWORD_LENGTH = 8

TRIP_NOT_FOUND = {"success": False, "error": "No plan found for this thread."}



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================
# Auth
# =========================

@app.post("/api/auth/register", status_code=201)
async def register(request_data: AuthRequest):
    email = request_data.email.strip()

    if "@" not in email or len(request_data.password) < MIN_PASSWORD_LENGTH:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Enter a valid email and a password of at least {MIN_PASSWORD_LENGTH} characters."
            }
        )

    user = await create_user(email, request_data.password)

    if user is None:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "This email is already registered."}
        )

    return {"success": True, "token": create_access_token(user["id"]), "email": user["email"]}


@app.post("/api/auth/login")
async def login(request_data: AuthRequest):
    user = await authenticate_user(request_data.email, request_data.password)

    if user is None:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "Wrong email or password."}
        )

    return {"success": True, "token": create_access_token(user["id"]), "email": user["email"]}


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {"success": True, "id": user["id"], "email": user["email"]}


# =========================
# Trips
# =========================

@app.get("/api/trips")
async def get_trips(user: dict = Depends(get_current_user)):
    return {"success": True, "trips": await list_trips(user["id"])}


MAX_TITLE_LENGTH = 80


@app.patch("/api/trips/{thread_id}")
async def rename_trip_route(thread_id: str, request_data: RenameRequest, user: dict = Depends(get_current_user)):
    title = " ".join(request_data.title.split())
    if not title or len(title) > MAX_TITLE_LENGTH:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Enter a name of 1 to {MAX_TITLE_LENGTH} characters."},
        )

    if not await rename_trip(user["id"], thread_id, title):
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    return {"success": True, "thread_id": thread_id, "title": title}


@app.delete("/api/trips/{thread_id}")
async def delete_trip_route(thread_id: str, user: dict = Depends(get_current_user)):
    if not await user_owns_trip(user["id"], thread_id):
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    run = ACTIVE_RUNS.get(user["id"])
    if run and run["thread_id"] == thread_id:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "This trip is being planned. Stop it before deleting it."},
        )

    await delete_trip(user["id"], thread_id)
    await delete_trip_state(thread_id)
    return {"success": True}


# =========================
# One run at a time
# =========================

# user id -> {"thread_id", "since", "started"}. One plan per account at a time:
# the local model can only work on one anyway. Kept in memory, which is enough
# for a single server process.
ACTIVE_RUNS: dict[int, dict] = {}

# A run registered but never started (the client left before the stream began)
# stops blocking after this long.
UNSTARTED_RUN_SECONDS = 60

RUN_IN_PROGRESS = {
    "success": False,
    "error": "A plan is already being generated. Wait for it to finish or press Stop.",
}


def claim_run(user_id: int, thread_id: str) -> dict | None:
    """Registers a run for this user, or returns None if one is already going."""
    current = ACTIVE_RUNS.get(user_id)
    if current and (current["started"] or time.monotonic() - current["since"] < UNSTARTED_RUN_SECONDS):
        return None

    run = {"thread_id": thread_id, "since": time.monotonic(), "started": False}
    ACTIVE_RUNS[user_id] = run
    return run


def release_run(user_id: int, run: dict):
    if ACTIVE_RUNS.get(user_id) is run:
        del ACTIVE_RUNS[user_id]


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest, user: dict = Depends(get_current_user)):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        thread_id = request_data.thread_id

        # Only continue threads this user owns. New thread ids are made by the
        # server, so a made-up id is treated as not found rather than created.
        if thread_id and not await user_owns_trip(user["id"], thread_id):
            return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

        run = claim_run(user["id"], thread_id or "")
        if run is None:
            return JSONResponse(status_code=409, content=RUN_IN_PROGRESS)

        run["started"] = True
        try:
            result = await run_travel_agent(
                user_input=user_message,
                thread_id=thread_id
            )
        finally:
            release_run(user["id"], run)

        if thread_id:
            await touch_trip(user["id"], thread_id)
        else:
            await create_trip(user["id"], result["thread_id"], user_message)

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "route": result["route"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )



# =========================
# Streaming planner
# =========================

# A comment line sent while nothing else happens, so the browser and any proxy
# know the connection is alive during the model's long silent reading phase.
HEARTBEAT_SECONDS = 10


def sse(event: dict) -> str:
    """One Server-Sent Event: `event: <type>` plus the JSON payload."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def friendly_error(error: Exception) -> str:
    text = str(error)
    if "Connection refused" in text or "ConnectError" in type(error).__name__:
        return "The AI model is not running. Start Ollama (brew services start ollama) and try again."
    return f"Planning failed: {text}"


@app.post("/api/travel/stream")
async def travel_planner_stream(request_data: TravelRequest, user: dict = Depends(get_current_user)):
    user_message = request_data.message.strip()
    flight_choice = request_data.flight_choice

    if not user_message and flight_choice is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "Message cannot be empty."})

    # A new trip needs dates. Feedback on a plan and a flight choice don't (feedback
    # like "go a week later" changes them in words).
    starts_new_trip = not request_data.thread_id and flight_choice is None
    if starts_new_trip or request_data.dates:
        date_error = check_dates(request_data.dates)
        if date_error:
            return JSONResponse(status_code=400, content={"success": False, "error": date_error})

    # Same ownership rule as /api/travel; checked before the stream starts so
    # errors are plain JSON responses.
    thread_id = request_data.thread_id
    if thread_id and not await user_owns_trip(user["id"], thread_id):
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    if flight_choice is not None:
        # Resuming: the trip must be paused on a flight choice, and the answer
        # must be one of its options (or 0 for none).
        pending = (await get_trip_plan(thread_id))["flight_options"] if thread_id else None
        if not pending:
            return JSONResponse(status_code=409, content={"success": False, "error": "This trip isn't waiting for a flight choice."})
        if flight_choice != 0 and flight_choice not in {option["number"] for option in pending}:
            return JSONResponse(status_code=400, content={"success": False, "error": "That flight option doesn't exist."})

    is_new_trip = not thread_id
    thread_id = thread_id or new_thread_id()

    run = claim_run(user["id"], thread_id)
    if run is None:
        return JSONResponse(status_code=409, content=RUN_IN_PROGRESS)

    async def events():
        run["started"] = True

        # The graph runs in its own task and hands events over through a queue,
        # so a heartbeat can be sent while the graph is quiet.
        queue: asyncio.Queue = asyncio.Queue()

        async def run_graph():
            try:
                async for event in stream_travel_agent(
                    thread_id, user_message or None, flight_choice, dates_for_graph(request_data.dates)
                ):
                    # Save the trip when the plan is finished, or when the run pauses
                    # for a flight choice, so the paused trip shows in My Trips.
                    if event["type"] in ("done", "choose"):
                        if is_new_trip:
                            await create_trip(user["id"], thread_id, user_message)
                        else:
                            await touch_trip(user["id"], thread_id)
                    await queue.put(event)
            except Exception as e:
                traceback.print_exc()
                await queue.put({"type": "error", "message": friendly_error(e)})
            finally:
                await queue.put(None)

        graph_task = asyncio.create_task(run_graph())

        try:
            # Tell the UI which trip this is right away, so a new trip can be
            # listed (and switched back to) while it is still being planned.
            yield sse({
                "type": "start",
                "thread_id": thread_id,
                "is_new": is_new_trip,
                "title": make_title(user_message) if user_message else "",
            })

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if event is None:
                    break
                yield sse(event)
        finally:
            # The browser closed the connection (Stop button, closed tab): stop the
            # graph, which also cancels the request to the model.
            if not graph_task.done():
                graph_task.cancel()
                print(f"Stream closed early, cancelled run for {thread_id}")
            release_run(user["id"], run)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # don't let a proxy hold events back
        },
    )


@app.get("/api/travel/{thread_id}")
async def get_travel_plan(thread_id: str, user: dict = Depends(get_current_user)):
    # Someone else's trip gets the same 404 as a missing one, so ids can't be probed.
    if not await user_owns_trip(user["id"], thread_id):
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    trip = await get_trip_plan(thread_id)

    if not trip["answer"] and not trip["flight_options"]:
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    # flight_options is set when the trip is paused, waiting for a flight choice.
    return {"success": True, "thread_id": thread_id, **trip}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Roamer AI API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )