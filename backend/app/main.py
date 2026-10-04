from contextlib import asynccontextmanager
from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.graph import run_travel_agent, get_trip_plan, open_travel_graph
from app.db import open_db, close_db

# The frontend lives in <repo>/frontend; FastAPI serves it.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await open_db()
    await open_travel_graph()
    yield
    await close_db()


app = FastAPI(
    title="TripMate AI",
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



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
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

        result = await run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "route": result["route"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "itinerary": result["itinerary"],
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
async def get_travel_plan(thread_id: str):
    answer = await get_trip_plan(thread_id)

    if not answer:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "No plan found for this thread."}
        )

    return {"success": True, "thread_id": thread_id, "answer": answer}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
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