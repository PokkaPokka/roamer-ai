"""
Eval 3: how often does each place-lookup strategy put a landmark in the right spot?

40 landmarks in 4 cities. The true coordinates and the local-language names come
from Wikipedia, so no LLM is involved: the only difference between the strategies
is how Roamer calls the Mapbox MCP server's search_and_geocode_tool.

- plain:        the English name, first result (a bare MCP tool call)
- destination:  the English name, near the destination, and only results that pass
                Roamer's checks (point of interest within 60 km of the city)
- roamer:       what the map agent does: the local-language name first, then the
                English name, with the same checks

A landmark counts as placed when the result is within 1 km of Wikipedia's
coordinates (big landmarks like palaces and parks span hundreds of metres).

Run from the backend folder:
    python -m evals.map_lookup make   # fetch coordinates and local names from Wikipedia
    python -m evals.map_lookup run    # score the three strategies (about 200 Mapbox calls)
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.map_agent import distance_km, good_match, mapbox_server, parse_places  # noqa: E402

LANDMARKS = Path(__file__).parent / "data" / "map_landmarks.json"
RESULTS = Path(__file__).parent / "data" / "map_results.json"
PLACED_KM = 1.0

# English Wikipedia titles. Cities picked so most are not English-speaking.
CITIES = {
    "Tokyo": ("ja", ["Sensō-ji", "Meiji Shrine", "Tokyo Tower", "Tokyo Skytree", "Shinjuku Gyoen",
                     "Ueno Park", "Tokyo Imperial Palace", "Shibuya Crossing", "Roppongi Hills", "Tsukiji fish market"]),
    "Paris": ("fr", ["Louvre", "Eiffel Tower", "Musée d'Orsay", "Jardin du Luxembourg", "Place de la Concorde",
                     "Notre-Dame de Paris", "Arc de Triomphe", "Centre Pompidou", "Palais Garnier", "Panthéon, Paris"]),
    "Bangkok": ("th", ["Wat Arun", "Wat Pho", "Grand Palace", "Chatuchak Weekend Market", "Lumphini Park",
                       "Jim Thompson House", "Khaosan Road", "MBK Center", "Wat Saket", "Siam Paragon"]),
    "Rome": ("it", ["Colosseum", "Trevi Fountain", "Pantheon, Rome", "Spanish Steps", "Piazza Navona",
                    "Roman Forum", "Castel Sant'Angelo", "Villa Borghese gardens", "Campo de' Fiori", "Circus Maximus"]),
}

WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RoamerAI-eval/1.0 (personal project; map lookup eval)"}


def make():
    landmarks = []
    for city, (language, titles) in CITIES.items():
        response = requests.get(WIKIPEDIA, headers=HEADERS, timeout=30, params={
            "action": "query", "format": "json", "redirects": 1, "titles": "|".join(titles),
            "prop": "coordinates|langlinks", "lllang": language, "lllimit": "max",
        }).json()
        for page in response["query"]["pages"].values():
            if "coordinates" not in page:
                print(f"  skipped {page['title']}: no coordinates")
                continue
            local = next((link["*"] for link in page.get("langlinks", [])), None)
            landmarks.append({
                "city": city, "language": language, "name": page["title"], "local_name": local,
                "lat": page["coordinates"][0]["lat"], "lon": page["coordinates"][0]["lon"],
            })
            print(f"{city}: {page['title']} / {local}")
    LANDMARKS.write_text(json.dumps(landmarks, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(landmarks)} landmarks to {LANDMARKS}")


async def run():
    landmarks = json.loads(LANDMARKS.read_text())
    client = MultiServerMCPClient({"mapbox": mapbox_server()})

    async with client.session("mapbox") as session:
        geocode = {tool.name: tool for tool in await load_mcp_tools(session)}["search_and_geocode_tool"]

        async def search(query: str, **options) -> list[dict]:
            return parse_places(await geocode.ainvoke({"q": query, **options}))

        centres = {}
        for city in CITIES:
            centres[city] = (await search(city, language="en", types=["place", "region", "country"]))[0]

        async def plain(landmark):
            results = await search(landmark["name"])
            return results[0] if results else None

        async def checked(landmark, attempts):
            centre = centres[landmark["city"]]
            near = {"longitude": centre["lon"], "latitude": centre["lat"]}
            for query, language in attempts:
                if not query:
                    continue
                for result in await search(query, language=language, proximity=near):
                    if good_match(result, query, centre):
                        return result
            return None

        strategies = {
            "plain": plain,
            "destination": lambda l: checked(l, [(l["name"], "en")]),
            "roamer": lambda l: checked(l, [(l["local_name"], l["language"]), (l["name"], l["language"]), (l["name"], "en")]),
        }

        rows = []
        for landmark in landmarks:
            row = {"city": landmark["city"], "name": landmark["name"]}
            for name, strategy in strategies.items():
                found = await strategy(landmark)
                row[name] = round(distance_km(found, landmark), 2) if found else None
            rows.append(row)
            print(f"{landmark['city']:8} {landmark['name'][:26]:26} " +
                  "  ".join(f"{name} {row[name] if row[name] is not None else '-':>7}" for name in strategies))

    print(f"\n{len(rows)} landmarks, placed = within {PLACED_KM:g} km of Wikipedia's coordinates:")
    summary = {}
    for name in strategies:
        errors = [row[name] for row in rows if row[name] is not None]
        placed = sum(error <= PLACED_KM for error in errors)
        by_city = {city: sum(1 for row in rows if row["city"] == city and row[name] is not None and row[name] <= PLACED_KM)
                   for city in CITIES}
        summary[name] = {"placed": placed, "total": len(rows), "not_found": len(rows) - len(errors),
                         "median_km": round(statistics.median(errors), 2) if errors else None, "by_city": by_city}
        print(f"  {name:12} {placed}/{len(rows)} placed  | not found {len(rows) - len(errors)} | by city {by_city}")

    RESULTS.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["make", "run"])
    args = parser.parse_args()
    make() if args.stage == "make" else asyncio.run(run())
