"""
Roamer as an MCP server: lets an MCP client such as Claude Desktop use Roamer's
flight search, hotel search, Wikivoyage knowledge base and your saved trips.

Runs over stdio. Start it with the project's Python, from anywhere:
    /path/to/.venv/bin/python /path/to/backend/app/mcp_server.py

The trip tools need ROAMER_TOKEN (make one with `python -m scripts.mcp_token`).
Nothing here may print to stdout: stdout is the MCP channel.
"""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import os
import re
import sys

# Started as a file path (as MCP clients do), so make `app.*` importable.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR.parent / ".env")

import anyio
from mcp.server.fastmcp import FastMCP

from app.auth import decode_access_token
from app.db import close_db, open_db
from app.graph import get_trip_plan as load_trip_plan, open_travel_graph
from app.knowledge_base import search_knowledge_base
from app.tools.flight_tool import resolve_location_to_iata
from app.tools.google_flights_tool import search_google_flights
from app.tools.tavily_tool import tavily_search
from app.trips import list_trips, user_owns_trip


@asynccontextmanager
async def lifespan(server: FastMCP):
    await open_db()
    await open_travel_graph()
    try:
        yield
    finally:
        await close_db()


mcp = FastMCP(
    "Roamer",
    instructions=(
        "Travel tools from the Roamer AI travel planner: real Google Flights fares, hotel "
        "suggestions from the web, excerpts from Wikivoyage city guides with source links, "
        "and the user's saved Roamer trip plans. Cite the Wikivoyage links when you use guide text."
    ),
    lifespan=lifespan,
)


# =========================
# Search tools (no login needed)
# =========================

def to_airport(place: str) -> str | None:
    """'MEL' stays as is; a city name is looked up, e.g. 'Melbourne' -> 'MEL'."""
    place = place.strip()
    if re.fullmatch(r"[A-Za-z]{3}", place) and place.isupper():
        return place
    return resolve_location_to_iata(place)


def future_date(value: str, name: str) -> str:
    try:
        day = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be a date like 2026-11-20.")
    if day <= date.today():
        raise ValueError(f"{name} must be after today.")
    return day.isoformat()


@mcp.tool()
async def search_flights(
    origin: str,
    destination: str,
    start_date: str,
    end_date: str | None = None,
    adults: int = 1,
) -> str:
    """
    Search Google Flights for real fares. origin and destination are city names or
    airport codes (e.g. "Melbourne" or "MEL"). Dates are YYYY-MM-DD; leave end_date
    out for a one-way trip. Prices are for the whole trip and all adults.
    """
    origin_code, destination_code = to_airport(origin), to_airport(destination)
    if not origin_code or not destination_code:
        missing = origin if not origin_code else destination
        return f"Couldn't find an airport for {missing!r}. Try the airport code, e.g. NRT."

    start = future_date(start_date, "start_date")
    end = future_date(end_date, "end_date") if end_date else None
    if end and end <= start:
        return "end_date must be after start_date."

    return await anyio.to_thread.run_sync(
        lambda: search_google_flights(origin_code, destination_code, start, end, max(adults, 1))
    )


@mcp.tool()
async def search_hotels(destination: str, preferences: str = "") -> str:
    """
    Find hotel suggestions on the web for a destination, e.g. destination "Tokyo",
    preferences "near Shinjuku station, mid-range". Results are web pages, not live prices.
    """
    query = f"Best hotels in {destination} {preferences}".strip()
    return await anyio.to_thread.run_sync(lambda: tavily_search(query))


@mcp.tool()
async def search_travel_guide(question: str, city: str | None = None, limit: int = 5) -> str:
    """
    Search Roamer's knowledge base of Wikivoyage guides for 500 popular cities, using
    hybrid search (meaning + keywords). Give a city to search only its guide, e.g.
    question "cheap places to eat", city "Tokyo". Returns numbered excerpts with links.
    """
    results = await search_knowledge_base(question, city=city, limit=min(max(limit, 1), 10))
    if not results:
        return "No guide excerpts found." + (f" Is {city!r} one of the guide's cities?" if city else "")

    blocks = [
        f"[{number}] {result['content']}\nSource: {result['source_url']}"
        for number, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks) + "\n\nGuide text from Wikivoyage, CC BY-SA 4.0."


# =========================
# Trip tools (need ROAMER_TOKEN)
# =========================

def current_user_id() -> int:
    user_id = decode_access_token(os.getenv("ROAMER_TOKEN", ""))
    if user_id is None:
        raise ValueError(
            "Roamer trips need a login token. Run `python -m scripts.mcp_token --email <you>` "
            "in the backend folder and set it as ROAMER_TOKEN in the MCP config."
        )
    return user_id


@mcp.tool()
async def list_my_trips() -> str:
    """List the user's saved Roamer trips, newest first, with their trip ids."""
    trips = await list_trips(current_user_id())
    if not trips:
        return "No saved trips yet."
    return "\n".join(f"- {trip['title']} (trip id: {trip['thread_id']}, updated {trip['updated_at'][:10]})" for trip in trips)


@mcp.tool()
async def get_trip_plan(trip_id: str) -> str:
    """Get the full plan (Markdown) of one of the user's saved trips. Use list_my_trips for the ids."""
    if not await user_owns_trip(current_user_id(), trip_id):
        return "No trip with that id in your account."

    trip = await load_trip_plan(trip_id)
    if trip["answer"]:
        return trip["answer"]
    if trip["flight_options"]:
        return "This trip is waiting for a flight choice in the Roamer web app; it has no plan yet."
    return "This trip has no plan yet."


if __name__ == "__main__":
    mcp.run()
