"""
Puts the finished itinerary on a map using the official Mapbox MCP server.

Roamer is the MCP client here: it starts `@mapbox/mcp-server` as a separate process
(stdio) through langchain-mcp-adapters and calls two of its tools:
- search_and_geocode_tool: place name -> coordinates
- matrix_tool: travel times between a day's stops

Flow: an LLM call lists each day's places (with their names in the local language,
which Mapbox needs for Japan), Mapbox finds them, and code builds the map data and
a "Getting around" section for the plan.
"""

from dataclasses import dataclass, field
import json
import math
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from pydantic import BaseModel, Field

MAPBOX_SERVER_PACKAGE = "@mapbox/mcp-server@0.14.0"

MAX_PLACES_PER_DAY = 5
MAX_DAYS = 10
# Places further than this from the destination are wrong matches (e.g. a café on Paros for "Senso-ji").
MAX_DISTANCE_KM = 60
# Legs up to this straight-line distance are timed on foot, longer ones by car.
WALKING_KM = 2.0
BUSY_DAY_MINUTES = 180
# A stop this far from the rest of its day is usually a wrong match with the same
# name (Bali: "Tegallalang" matched a place 30 km north of the rice terraces).
OUTLIER_KM = 20
# Longer legs are shown but not added up: they cross water or are wrong matches.
MAX_LEG_MINUTES = 120


# =========================
# Places in the itinerary (LLM)
# =========================

class MapPlace(BaseModel):
    name: str = Field(description="The place's usual English name, e.g. 'Senso-ji Temple'.")
    local_name: str | None = Field(None, description="The name in the local language and script, e.g. '浅草寺'. Null if the same as name.")


class MapDay(BaseModel):
    day: int = Field(description="Day number in the itinerary, starting at 1.")
    places: list[MapPlace] = Field(description="Specific places visited that day, in order.")


class ItineraryPlaces(BaseModel):
    local_language: str = Field(description="ISO 639-1 code of the destination's main language, e.g. 'ja' for Japan, 'fr' for France.")
    days: list[MapDay]


EXTRACT_PROMPT = (
    "List the specific places to visit on each day of this travel itinerary, in order: sights, "
    "museums, temples, parks, markets, neighbourhoods and named restaurants. Skip the airport, "
    "the hotel and generic places like 'a local restaurant'. At most "
    f"{MAX_PLACES_PER_DAY} places per day. Give each place's name in the local language too.\n"
    'Example: Day 1 "visit Senso-ji, then walk to Nakamise Street" in Tokyo -> '
    '{"local_language": "ja", "days": [{"day": 1, "places": ['
    '{"name": "Senso-ji Temple", "local_name": "浅草寺"}, {"name": "Nakamise Street", "local_name": "仲見世通り"}]}]}'
)


async def extract_places(llm, itinerary: str, destination: str) -> ItineraryPlaces:
    extractor = llm.with_structured_output(ItineraryPlaces)
    return await extractor.ainvoke([
        SystemMessage(content=EXTRACT_PROMPT),
        HumanMessage(content=f"Destination: {destination}\n\nItinerary:\n{itinerary}"),
    ])


# =========================
# Mapbox results (text) -> data
# =========================

RESULT = re.compile(
    r"\d+\.\s+([^\n]+)\n\s+Address:[^\n]*\n\s+Coordinates:\s*(-?[\d.]+),\s*(-?[\d.]+)\n\s+Type:\s*(\w+)"
)


def result_text(result) -> str:
    """Tool results come back as a string or a list of content blocks."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in result)
    return str(result)


def parse_places(result) -> list[dict]:
    return [
        {"name": name.strip(), "lat": float(lat), "lon": float(lon), "type": kind}
        for name, lat, lon, kind in RESULT.findall(result_text(result))
    ]


def parse_json(result) -> dict:
    text = result_text(result)
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def distance_km(a: dict, b: dict) -> float:
    """Great-circle distance between two {lat, lon} points."""
    lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
    dlat, dlon = lat2 - lat1, math.radians(b["lon"] - a["lon"])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def words(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 2}


def good_match(found: dict, query: str, centre: dict) -> bool:
    """A point of interest near the destination, or a neighbourhood whose name matches the query."""
    if distance_km(found, centre) > MAX_DISTANCE_KM:
        return False
    if found["type"] == "poi":
        return True
    # "Tokyo Tower" -> the city of Tokyo is a wrong match; "Yanaka" -> the Yanaka neighbourhood is right.
    return found["type"] in ("neighborhood", "locality") and words(query) <= words(found["name"]) | {"district", "area"}


def drop_outliers(stops: list[dict]) -> tuple[list[dict], list[dict]]:
    """Splits a day's stops into kept ones and ones far from the others (needs 3+ stops)."""
    if len(stops) < 3:
        return stops, []
    kept, dropped = [], []
    for index, stop in enumerate(stops):
        others = stops[:index] + stops[index + 1:]
        middle = {
            "lat": sorted(other["lat"] for other in others)[len(others) // 2],
            "lon": sorted(other["lon"] for other in others)[len(others) // 2],
        }
        (dropped if distance_km(stop, middle) > OUTLIER_KM else kept).append(stop)
    return kept, dropped


def format_minutes(minutes: float) -> str:
    minutes = round(minutes)
    return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"


# =========================
# The agent
# =========================

@dataclass
class TripMap:
    data: dict | None = None          # for the web map: centre, days, places, legs
    section: str = ""                 # "Getting around" Markdown for the plan
    notes: list[str] = field(default_factory=list)   # status lines for the progress panel


def mapbox_server() -> dict:
    return {
        "command": "npx",
        "args": ["-y", MAPBOX_SERVER_PACKAGE],
        "transport": "stdio",
        "env": {"MAPBOX_ACCESS_TOKEN": os.environ["MAPBOX_ACCESS_TOKEN"], "PATH": os.environ.get("PATH", "")},
    }


async def build_trip_map(llm, itinerary: str, destination: str, report=lambda message: None) -> TripMap:
    if not os.getenv("MAPBOX_ACCESS_TOKEN"):
        return TripMap(notes=["No MAPBOX_ACCESS_TOKEN set, so no map"])

    report("Finding each day's places in the itinerary")
    found_places = await extract_places(llm, itinerary, destination)
    language = (found_places.local_language or "en").lower()[:2]
    days = [day for day in found_places.days if day.places][:MAX_DAYS]

    client = MultiServerMCPClient({"mapbox": mapbox_server()})
    # One Mapbox server process for the whole run, not one per tool call.
    async with client.session("mapbox") as session:
        tools = {tool.name: tool for tool in await load_mcp_tools(session)}
        geocode, matrix = tools["search_and_geocode_tool"], tools["matrix_tool"]

        # Only cities, regions and countries: a plain "Tokyo" search can return a
        # restaurant called Tokyo near the server instead.
        centre_results = parse_places(await geocode.ainvoke(
            {"q": destination, "language": "en", "types": ["place", "region", "country"]}
        ))
        if not centre_results:
            return TripMap(notes=[f"Mapbox couldn't find {destination}"])
        centre = centre_results[0]
        near = {"longitude": centre["lon"], "latitude": centre["lat"]}

        async def locate(place: MapPlace) -> dict | None:
            # The local name in the local language works best (Japan has few English
            # place names in Mapbox), then the English name.
            attempts = [(place.local_name, language), (place.name, language), (place.name, "en")]
            for query, lang in attempts:
                if not query:
                    continue
                for result in parse_places(await geocode.ainvoke({"q": query, "language": lang, "proximity": near})):
                    if good_match(result, query, centre):
                        return {"name": place.name, "lat": result["lat"], "lon": result["lon"]}
            return None

        report(f"Looking up {sum(len(day.places) for day in days)} places on Mapbox")
        map_days, missing = [], []
        for day in days:
            stops = []
            for place in day.places[:MAX_PLACES_PER_DAY]:
                spot = await locate(place)
                if spot:
                    stops.append(spot)
                else:
                    missing.append(place.name)

            stops, far = drop_outliers(stops)
            missing += [stop["name"] for stop in far]

            legs = []
            if len(stops) >= 2:
                coordinates = [{"longitude": stop["lon"], "latitude": stop["lat"]} for stop in stops]
                durations = {}
                for profile in ("walking", "driving"):
                    reply = parse_json(await matrix.ainvoke({"coordinates": coordinates, "profile": f"mapbox/{profile}"}))
                    durations[profile] = reply.get("durations") or []
                for index in range(len(stops) - 1):
                    mode = "walking" if distance_km(stops[index], stops[index + 1]) <= WALKING_KM else "driving"
                    rows = durations[mode]
                    seconds = rows[index][index + 1] if len(rows) > index and rows[index] else None
                    if seconds is not None:
                        legs.append({"from": index, "to": index + 1, "minutes": round(seconds / 60), "mode": mode,
                                     "far": seconds / 60 > MAX_LEG_MINUTES})

            map_days.append({"day": day.day, "stops": stops, "legs": legs,
                             "travel_minutes": sum(leg["minutes"] for leg in legs if not leg["far"])})

    located = sum(len(day["stops"]) for day in map_days)
    notes = [f"Found {located} of {located + len(missing)} places on the map"]
    if not located:
        return TripMap(notes=notes)

    lines = [
        "## Getting around",
        "",
        "_Travel times between each day's stops, from Mapbox: walking under 2 km, otherwise by car. "
        "Public transport times may differ._",
        "",
    ]
    for day in map_days:
        if not day["stops"]:
            continue
        route = day["stops"][0]["name"]
        for leg in day["legs"]:
            way = "walk" if leg["mode"] == "walking" else "by car"
            timing = "far apart, check this" if leg["far"] else f"{format_minutes(leg['minutes'])} {way}"
            route += f" → {day['stops'][leg['to']]['name']} ({timing})"
        line = f"- **Day {day['day']}:** {route}"
        if day["travel_minutes"]:
            line += f". About {format_minutes(day['travel_minutes'])} of travel."
        if day["travel_minutes"] > BUSY_DAY_MINUTES:
            line += " That's a lot of time in transit; consider fewer stops or grouping them by area."
            notes.append(f"Day {day['day']} has about {format_minutes(day['travel_minutes'])} of travel")
        lines.append(line)
    if missing:
        lines += ["", f"Not on the map (not found, or far from the rest of that day): {', '.join(missing)}."]
    lines += ["", "Map data © Mapbox, © OpenStreetMap."]

    data = {"centre": {"lat": centre["lat"], "lon": centre["lon"]}, "days": map_days}
    return TripMap(data=data, section="\n".join(lines), notes=notes)
