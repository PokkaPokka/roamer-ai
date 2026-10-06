from contextlib import asynccontextmanager
from pathlib import Path
import traceback
import uvicorn

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.graph import run_travel_agent, get_trip_plan, open_travel_graph
from app.db import open_db, close_db
from app.auth import authenticate_user, create_access_token, create_user, get_current_user
from app.trips import create_trip, list_trips, touch_trip, user_owns_trip

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



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


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

        result = await run_travel_agent(
            user_input=user_message,
            thread_id=thread_id
        )

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



@app.get("/api/travel/{thread_id}")
async def get_travel_plan(thread_id: str, user: dict = Depends(get_current_user)):
    # Someone else's trip gets the same 404 as a missing one, so ids can't be probed.
    if not await user_owns_trip(user["id"], thread_id):
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    answer = await get_trip_plan(thread_id)

    if not answer:
        return JSONResponse(status_code=404, content=TRIP_NOT_FOUND)

    return {"success": True, "thread_id": thread_id, "answer": answer}


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