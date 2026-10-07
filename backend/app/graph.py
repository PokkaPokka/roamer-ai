import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated, Literal
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import asyncio
import operator
import re
import uuid

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_config, get_stream_writer
from langgraph.types import Command, interrupt
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.tools.tavily_tool import tavily_search
from app.tools.flight_tool import resolve_location_to_iata, parse_route, DEFAULT_ORIGIN_IATA
from app.tools.google_flights_tool import search_google_flights_with_options
from app.db import get_pool
from app.knowledge_base import search_knowledge_base, city_in_knowledge_base
from app.critic import review_plan, section_lines
from app.map_agent import build_trip_map


# =========================
# LLM
# =========================

# "ollama" runs a local model for free development; "groq" uses the hosted API.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def get_llm():
    if LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=False,
            # Ollama's default context window is small and silently cuts long prompts.
            # Guide excerpts + flights + hotels need about 8K tokens.
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            # A plan is about 1,500 tokens. Without a cap, a model that starts copying
            # its prompt keeps going until the context is full (once: 7,900 tokens, 26 min).
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "3000")),
        )

    if LLM_PROVIDER == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'groq'.")


llm = get_llm()


# =========================
# Progress messages
# =========================

def report(message: str):
    """
    Sends a status line to the UI while the graph is streaming, e.g.
    "Searching Google Flights MEL → NRT". Does nothing in a plain ainvoke() run.
    """
    node = get_config()["metadata"].get("langgraph_node", "")
    get_stream_writer()({"node": node, "message": message})


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    latest_input: str      # what the user just sent: a new trip or feedback
    user_query: str        # the current trip request the plan is built from
    route: str             # "plan", "new_search" or "revise"
    trip_request: dict     # TripRequest fields extracted once by plan_trip
    date_overrides: dict   # dates picked in the form for this run; they win over the text
    destination: str       # where the trip goes, as extracted from the request
    flight_results: str
    flight_options: list[dict]   # the same flights as data, for the UI's flight cards
    chosen_flight: str     # the flight the user picked, as text for the plan prompt
    chosen_option: dict    # the same flight as data (price, dates) for the critic
    hotel_results: str
    guide_context: str     # numbered Wikivoyage excerpts for the plan prompt
    guide_sources: list[dict]
    final_plan: str
    critic_problems: list[str]   # problems the critic wants an LLM rewrite for
    fix_attempts: int            # rewrites so far in this run (at most MAX_FIX_ATTEMPTS)
    map_data: dict               # the itinerary's places and travel times, for the web map


# =========================
# Router Agent
# =========================

class FeedbackDecision(BaseModel):
    action: Literal["revise", "new_search"] = Field(
        description=(
            "'new_search' if the feedback changes the destination, origin, travel dates, "
            "number of travellers, asks for different flights or hotels, or says flights "
            "or hotels aren't needed (e.g. going by train instead). "
            "'revise' if the plan can be edited without new search results."
        )
    )
    updated_request: str = Field(
        description="The full trip request rewritten to include the feedback."
    )


def router_agent(state: TravelState):
    latest_input = state["latest_input"]

    # First message in this thread: build a plan from scratch.
    if not state.get("final_plan"):
        return {"route": "plan", "user_query": latest_input}

    report("Deciding whether your feedback needs new searches")
    decider = llm.with_structured_output(FeedbackDecision)

    try:
        decision = decider.invoke([
            SystemMessage(content=(
                "The user already has a travel plan and is giving feedback on it. "
                "Decide whether new flight or hotel searches are needed, and rewrite "
                "the trip request to include the feedback.\n"
                'Example: request "3 days in Tokyo from Melbourne", feedback "make it 5 days" -> '
                '{"action": "new_search", "updated_request": "5 days in Tokyo from Melbourne"}\n'
                'Example: request "3 days in Tokyo from Melbourne leaving 2026-11-20", feedback "go a week later, my dates are flexible" -> '
                '{"action": "new_search", "updated_request": "3 days in Tokyo from Melbourne leaving 2026-11-27, dates flexible by a few days"}\n'
                'Example: request "3 days in Tokyo from Kyoto", feedback "flight is not needed, go with train" -> '
                '{"action": "new_search", "updated_request": "3 days in Tokyo from Kyoto, travelling by train, no flights needed"}\n'
                'Example: request "3 days in Tokyo from Melbourne", feedback "add more food spots on day 2" -> '
                '{"action": "revise", "updated_request": "3 days in Tokyo from Melbourne, with more food spots on day 2"}'
            )),
            HumanMessage(content=(
                f"Current trip request: {state['user_query']}\n"
                f"Feedback: {latest_input}"
            ))
        ])
    except Exception as e:
        print("Feedback routing failed, defaulting to revise:", e)
        decision = FeedbackDecision(action="revise", updated_request=state["user_query"])

    return {
        "route": decision.action,
        "user_query": decision.updated_request or state["user_query"],
    }


def route_after_router(state: TravelState):
    return "revise_agent" if state["route"] == "revise" else "plan_trip"


# =========================
# Flight Agent
# =========================

class TripRequest(BaseModel):
    origin: str | None = Field(None, description="City the traveller departs from (the place after 'from'). Null if not mentioned.")
    destination: str | None = Field(None, description="City or country the traveller is going to (the place after 'to', or the place the trip is about).")
    departure_date: str | None = Field(None, description="Departure date as YYYY-MM-DD. Null if not mentioned.")
    return_date: str | None = Field(None, description="Return date as YYYY-MM-DD. Null if not mentioned.")
    trip_days: int | None = Field(None, description="Trip length in days, e.g. 3 for 'a 3 day trip'.")
    adults: int = Field(1, description="Number of adult travellers.")
    flex_days: int = Field(0, description=(
        "How many days earlier or later the traveller could leave, 0 to 3. "
        "'around 15 Nov' or 'give or take a few days' -> 2 or 3. 0 if exact or not mentioned."
    ))
    month: str | None = Field(None, description=(
        "YYYY-MM if the traveller can leave any time in a month ('sometime in November'). Otherwise null."
    ))
    needs_flights: bool = Field(True, description=(
        "False only if the traveller is not flying: driving, taking a train or bus, "
        "or already has flights booked. Otherwise true."
    ))
    needs_hotels: bool = Field(True, description=(
        "False only if the traveller already has somewhere to stay (staying with friends "
        "or family, hotel already booked) or it is a day trip. Otherwise true."
    ))


def parse_future_date(value: str | None, today: date) -> date | None:
    try:
        parsed = date.fromisoformat(value) if value else None
    except ValueError:
        return None
    return parsed if parsed and parsed > today else None


def extract_trip_request(query: str) -> TripRequest:
    today = date.today()
    extractor = llm.with_structured_output(TripRequest)

    try:
        return extractor.invoke([
            SystemMessage(content=(
                f"Extract the trip details from the user's request. Today is {today.isoformat()}. "
                "Convert relative dates like 'next Friday' to YYYY-MM-DD. "
                "Use null for anything the user did not mention.\n"
                'Example: "5 days in Paris from London, leaving 2026-05-01" -> '
                '{"origin": "London", "destination": "Paris", "departure_date": "2026-05-01", '
                '"return_date": null, "trip_days": 5, "adults": 1, "flex_days": 0, "month": null, '
                '"needs_flights": true, "needs_hotels": true}\n'
                'Example: "a week in Bangkok from Sydney around 2026-07-15, dates are flexible" -> '
                '{"origin": "Sydney", "destination": "Bangkok", "departure_date": "2026-07-15", '
                '"return_date": null, "trip_days": 7, "adults": 1, "flex_days": 3, "month": null, '
                '"needs_flights": true, "needs_hotels": true}\n'
                'Example: "3 day road trip from Melbourne to Sydney, I\'m driving" -> '
                '{"origin": "Melbourne", "destination": "Sydney", "departure_date": null, '
                '"return_date": null, "trip_days": 3, "adults": 1, "needs_flights": false, "needs_hotels": true}\n'
                'Example: "A week in Rome from Paris, staying with my sister" -> '
                '{"origin": "Paris", "destination": "Rome", "departure_date": null, '
                '"return_date": null, "trip_days": 7, "adults": 1, "needs_flights": true, "needs_hotels": false}'
            )),
            HumanMessage(content=query)
        ])
    except Exception as e:
        print("Trip extraction failed, falling back to regex:", e)
        return TripRequest()


# =========================
# Plan Trip: the supervisor (runs before the parallel searches)
# =========================

# The small model sometimes skips a search the user never mentioned, so a skip
# only counts when the request says something related.
NO_FLIGHT_HINTS = re.compile(
    r"\b(driv\w*|road ?trip|cars?|trains?|rail\w*|shinkansen|bus|buses|coach|ferry|ferries"
    r"|no flights?|not flying|flights? (is |are )?not needed"
    r"|already (have|booked) (my |our )?(flights?|tickets?)|flights? (are |is )?(already )?booked)\b",
    re.I,
)
NO_HOTEL_HINTS = re.compile(
    r"\b(staying (with|at)|stay with|friends'|(a|one) day trip"
    r"|already (have|booked) (a |my |our )?(hotel|room|accommodation|place)"
    r"|(hotel|accommodation) (is |are )?(already )?booked)\b",
    re.I,
)

FLIGHTS_NOT_NEEDED = "Flight search not needed: the traveller isn't flying (e.g. driving) or already has flights."
HOTELS_NOT_NEEDED = "Hotel search not needed: the traveller already has somewhere to stay."


MAX_FLEX_DAYS = 3


def describe_dates(trip: TripRequest) -> str:
    """E.g. "leaving 2026-11-20 ±2 days, returning 2026-11-23" or "leaving any time in 2026-11, 5 days"."""
    if trip.month:
        parts = [f"leaving any time in {trip.month}"]
    elif trip.departure_date:
        parts = [f"leaving {trip.departure_date}" + (f" ±{trip.flex_days} days" if trip.flex_days else "")]
    else:
        parts = []
    if trip.return_date:
        parts.append(f"returning {trip.return_date}")
    elif trip.trip_days:
        parts.append(f"{trip.trip_days} days")
    return ", ".join(parts)


def plan_trip(state: TravelState):
    # Extract the trip once, so the searches can all start at the same time, and
    # decide which searches this trip needs (see route_searches).
    report("Reading your trip request")
    query = state["user_query"]
    trip = extract_trip_request(query)
    trip.flex_days = min(max(trip.flex_days or 0, 0), MAX_FLEX_DAYS)
    # The model sometimes writes the month as "January"; use the departure date's month then.
    if trip.month and not re.fullmatch(r"\d{4}-\d{2}", trip.month):
        trip.month = trip.departure_date[:7] if trip.departure_date else None

    # Dates picked in the form win over what the model read from the text. They
    # are also written into the request text, so feedback later ("make it 5 days")
    # is rewritten from a request that still has them.
    overrides = state.get("date_overrides") or {}
    if overrides:
        trip.departure_date = overrides.get("departure_date") or trip.departure_date
        trip.return_date = overrides.get("return_date") or trip.return_date
        flexibility = overrides.get("flexibility", "exact")
        trip.flex_days = int(flexibility) if flexibility.isdigit() else 0
        trip.month = None

        if trip.departure_date and trip.return_date:
            # "N days" means N days there: 20 -> 23 Nov is 3 days.
            trip.trip_days = (date.fromisoformat(trip.return_date) - date.fromisoformat(trip.departure_date)).days
            if flexibility == "month":
                # Any day that month: the end date no longer applies, only the trip length.
                trip.month = trip.departure_date[:7]
                trip.return_date = None
        query = f"{query} (travel dates: {describe_dates(trip)})"

    if not trip.needs_flights and not NO_FLIGHT_HINTS.search(state["user_query"]):
        trip.needs_flights = True
    if not trip.needs_hotels and not NO_HOTEL_HINTS.search(state["user_query"]):
        trip.needs_hotels = True

    details = [part for part in [
        f"to {trip.destination}" if trip.destination else "",
        f"from {trip.origin}" if trip.origin else "",
        describe_dates(trip),
    ] if part]
    report("Trip: " + ", ".join(details) if details else "Could not find trip details; using defaults")

    updates = {
        "user_query": query,
        "trip_request": trip.model_dump(),
        "destination": trip.destination or "",
        # Clear the previous run's flight choice; a new search asks again.
        "flight_options": [],
        "chosen_flight": "",
        "chosen_option": {},
    }

    # A skipped search still writes its results, so a new search in the same
    # thread doesn't reuse old flights and the plan can say why there are none.
    if not trip.needs_flights:
        updates["flight_results"] = FLIGHTS_NOT_NEEDED
    if not trip.needs_hotels:
        updates["hotel_results"] = HOTELS_NOT_NEEDED

    return updates


def route_searches(state: TravelState) -> list[str]:
    """Conditional fan-out: only the searches this trip needs run in parallel."""
    trip = state.get("trip_request", {})
    searches = []
    if trip.get("needs_flights", True):
        searches.append("flight_agent")
    if trip.get("needs_hotels", True):
        searches.append("hotel_agent")
    searches.append("guide_agent")
    return searches


# Each flexible date is one SerpApi search, so cap them to protect the monthly quota.
FLEX_MAX_SEARCHES = int(os.getenv("FLEX_MAX_SEARCHES", "7"))
MAX_FLIGHT_CARDS = 5


def candidate_departures(trip: TripRequest, departure: date, today: date) -> tuple[list[date], str]:
    """
    The departure dates to search, and a note for the plan when they are a sample.
    ±N days searches every day in the window; a whole month is sampled evenly.
    """
    if trip.month:
        try:
            first = date.fromisoformat(f"{trip.month}-01")
        except ValueError:
            return [departure], ""
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        start = max(first, today + timedelta(days=1))
        if start > last:
            return [departure], ""
        span = (last - start).days
        if span + 1 <= FLEX_MAX_SEARCHES:
            return [start + timedelta(days=i) for i in range(span + 1)], ""
        days = sorted({start + timedelta(days=round(i * span / (FLEX_MAX_SEARCHES - 1)))
                       for i in range(FLEX_MAX_SEARCHES)})
        return days, f"searched {len(days)} departure dates spread across {trip.month}, not every day"

    if trip.flex_days:
        window = [departure + timedelta(days=offset) for offset in range(-trip.flex_days, trip.flex_days + 1)]
        window = [day for day in window if day > today]
        # Keep the dates closest to the one asked for if the cap is smaller than the window.
        window = sorted(window, key=lambda day: abs((day - departure).days))[:FLEX_MAX_SEARCHES]
        return sorted(window), ""

    return [departure], ""


def short_date(day: date) -> str:
    return f"{day.strftime('%a')} {day.day} {day.strftime('%b')}"


def flight_agent(state: TravelState):
    query = state["user_query"]
    trip = TripRequest(**state.get("trip_request", {}))

    # Fall back to the regex parser for any location the LLM missed.
    regex_origin, regex_destination = parse_route(query)
    origin_iata = resolve_location_to_iata(trip.origin) if trip.origin else None
    destination_iata = resolve_location_to_iata(trip.destination) if trip.destination else None
    origin_iata = origin_iata or regex_origin or DEFAULT_ORIGIN_IATA
    destination_iata = destination_iata or regex_destination

    today = date.today()
    assumptions = []

    departure = parse_future_date(trip.departure_date, today)
    if not departure and trip.month:
        departure = parse_future_date(f"{trip.month}-01", today) or today + timedelta(days=1)
    if not departure:
        departure = today + timedelta(days=30)
        assumptions.append(f"no valid travel date given, so searched {departure.isoformat()} (30 days from today)")

    return_day = parse_future_date(trip.return_date, departure)
    if not return_day and trip.trip_days:
        # "3 days in Tokyo" leaving the 20th returns on the 23rd, not the 22nd: a
        # long flight shouldn't eat one of the days.
        return_day = departure + timedelta(days=max(trip.trip_days, 1))
    # Flexible dates move the whole trip, so the length stays the same.
    trip_length = (return_day - departure).days if return_day else None

    departures, sample_note = candidate_departures(trip, departure, today)
    if sample_note:
        assumptions.append(sample_note)
    date_pairs = [(day, day + timedelta(days=trip_length) if trip_length else None) for day in departures]

    flight_options = []

    if not destination_iata:
        flight_data = "Flight search skipped: could not work out the destination from the request."
        report("Skipped: could not work out the destination airport")
    else:
        if len(date_pairs) == 1:
            report(f"Searching Google Flights {origin_iata} → {destination_iata}, {departure.isoformat()}")
        else:
            report(f"Searching Google Flights {origin_iata} → {destination_iata} on {len(date_pairs)} dates, "
                   f"{short_date(departures[0])} to {short_date(departures[-1])}")

        def search(pair):
            outbound, inbound = pair
            return search_google_flights_with_options(
                origin_iata=origin_iata,
                destination_iata=destination_iata,
                outbound_date=outbound.isoformat(),
                return_date=inbound.isoformat() if inbound else None,
                adults=max(trip.adults, 1),
            )

        # The searches are independent web requests, so run them at the same time.
        with ThreadPoolExecutor(max_workers=len(date_pairs)) as pool:
            results = list(pool.map(search, date_pairs))

        for (outbound, inbound), (_, options) in zip(date_pairs, results):
            for option in options:
                option["outbound_date"] = outbound.isoformat()
                option["return_date"] = inbound.isoformat() if inbound else None

        if len(date_pairs) == 1:
            flight_data, flight_options = results[0]
        else:
            flight_data, flight_options = combine_flexible_results(
                origin_iata, destination_iata, date_pairs, results, trip.adults
            )

        prices = [option["price"] for option in flight_options if option["price"] is not None]
        if not prices:
            report("No flight options found")
        elif len(date_pairs) == 1:
            report(f"Found {len(flight_options)} options from {flight_options[0]['currency']} {min(prices):,}")
        else:
            cheapest = flight_options[0]
            report(f"Searched {len(date_pairs)} dates; cheapest {cheapest['currency']} {cheapest['price']:,} "
                   f"leaving {short_date(date.fromisoformat(cheapest['outbound_date']))}")

    if assumptions:
        flight_data = "Assumptions: " + "; ".join(assumptions) + "\n\n" + flight_data

    return {
        "flight_results": flight_data,
        "flight_options": flight_options,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
    }


def combine_flexible_results(origin, destination, date_pairs, results, adults) -> tuple[str, list[dict]]:
    """
    Merges one search per date pair into a summary (cheapest fare per date pair)
    plus the cheapest options overall, renumbered 1..N for the flight cards.
    """
    all_options = [option for _, options in results for option in options]
    if not all_options:
        # Every search failed or found nothing: pass on the first message.
        return results[0][0], []

    currency = all_options[0]["currency"]
    lines = [
        f"Flexible-date search, {origin} -> {destination}, {adults} adult{'s' if adults > 1 else ''}, "
        f"prices in {currency}: searched {len(date_pairs)} date pairs.",
        "Cheapest fare for each date pair:",
    ]
    for (outbound, inbound), (_, options) in zip(date_pairs, results):
        prices = [option["price"] for option in options if option["price"] is not None]
        when = f"leave {outbound.isoformat()}" + (f", return {inbound.isoformat()}" if inbound else "")
        lines.append(f"- {when}: {f'{currency} {min(prices):,}' if prices else 'no flights found'}")

    best = sorted(all_options, key=lambda option: (option["price"] is None, option["price"] or 0))
    best = best[:MAX_FLIGHT_CARDS]
    for number, option in enumerate(best, start=1):
        when = f"leave {option['outbound_date']}" + (f", return {option['return_date']}" if option["return_date"] else "")
        option["number"] = number
        option["text"] = re.sub(r"^Option \d+:", f"Option {number} ({when}):", option["text"])

    return "\n".join(lines) + "\n\n" + "\n\n".join(option["text"] for option in best), best


# =========================
# Choose Flight (human in the loop)
# =========================

NO_FLIGHT_CHOSEN = (
    "The user chose none of the flights found. Don't recommend a specific flight or price; "
    "suggest comparing fares closer to the travel date."
)


def chosen_flight_text(option: dict, adults: int) -> str:
    lines = [
        f"The user picked this flight (Option {option['number']}):",
        option["text"],
        f"Price: {option['currency']} {option['price']} "
        f"{'for the whole round trip' if option['round_trip'] else 'one way'}, "
        f"{adults} adult{'s' if adults > 1 else ''}. Count it once in the budget.",
    ]
    if option.get("outbound_date"):
        lines.append(
            f"Travel dates for the whole plan: leave {option['outbound_date']}"
            + (f", return {option['return_date']}." if option.get("return_date") else ".")
        )
    arrival = option.get("arrival_time") or ""
    if " " in arrival:
        arrival_date, arrival_clock = arrival.split(" ", 1)
        lines.append(
            f"The flight lands at {option.get('arrival_airport', 'the destination')} on {arrival_date} at "
            f"{arrival_clock}. Day 1 of the itinerary is {arrival_date} and starts after landing and "
            f"getting to the hotel; schedule nothing before {arrival_clock} that day."
        )
    if option["round_trip"]:
        lines.append(
            "Only the outbound flights are listed; the return flight is picked when booking, "
            "so don't invent return flight numbers or times. Keep the last day light and tell "
            "the traveller to leave 3 hours to get to the airport."
        )
    return "\n".join(lines)


def cheapest_option(options: list[dict]) -> dict:
    priced = [option for option in options if option["price"] is not None]
    return min(priced, key=lambda option: option["price"]) if priced else options[0]


def choose_flight(state: TravelState):
    """
    Fan-in point after the searches. When the run asks for it (the web UI does),
    the graph pauses here with interrupt() until the user picks a flight. The
    paused state is saved in Postgres, so the user can come back later.
    """
    options = state.get("flight_options") or []
    if not options:
        return {}

    if get_config()["configurable"].get("ask_flight_choice"):
        report("Waiting for you to pick a flight")
        # The value goes to the UI; the run resumes with Command(resume=<option number>),
        # or 0 to skip. On resume this node runs again from the top.
        choice = interrupt({
            "type": "choose_flight",
            "options": [{key: value for key, value in option.items() if key != "text"} for option in options],
        })
    else:
        # curl and scripts can't answer a question, so take the cheapest.
        choice = cheapest_option(options)["number"]

    chosen = next((option for option in options if option["number"] == choice), None)
    if chosen is None:
        report("No flight chosen")
        return {"chosen_flight": NO_FLIGHT_CHOSEN, "chosen_option": {}}

    adults = max(state.get("trip_request", {}).get("adults") or 1, 1)
    report(f"Using option {chosen['number']}: {', '.join(chosen['airlines'])}, "
           f"{chosen['currency']} {chosen['price']:,}")
    return {
        "chosen_flight": chosen_flight_text(chosen, adults),
        "chosen_option": {key: value for key, value in chosen.items() if key != "text"},
    }


def flight_section(state: TravelState) -> str:
    """The Flights part of the plan prompt: the chosen flight instead of every option."""
    results = state.get("flight_results", "")
    chosen = state.get("chosen_flight")
    if not chosen:
        return results
    # Keep the summary lines (assumptions, route, price insight) but not the option list.
    summary = re.split(r"\n\nOption 1\b", results)[0]
    return f"{summary}\n\n{chosen}"




# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    report("Searching the web for hotels")
    hotel_results = tavily_search(query)
    report("Found hotel suggestions")

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
    }




# =========================
# Guide Agent (RAG)
# =========================

# One search per part of the itinerary, so the excerpts cover more than one topic.
GUIDE_SEARCHES = [
    ("top sights and attractions in {place}", "See"),
    ("things to do and activities in {place}", "Do"),
    ("where to eat local food in {place}", "Eat"),
    ("getting around {place} by public transport", "Get around"),
]
RESULTS_PER_SEARCH = 3

SOURCES_HEADING = "## Sources"


# "react": the model picks its own searches with a tool; "fixed": the 4 searches above.
GUIDE_MODE = os.getenv("GUIDE_MODE", "react").lower()
MAX_GUIDE_SEARCHES = 5
MAX_EXCERPTS = 12

GUIDE_AGENT_PROMPT = (
    "You decide what to look up in a travel guide before a trip plan is written. "
    "Call search_guide 3 to 5 times, one short topic per call, covering the main sights, "
    "food, getting around, and anything the traveller asked for (interests, budget, kids, "
    "nightlife, day trips...). Don't repeat a topic. When you have searched enough, "
    "reply with the single word DONE."
)


def keep_result(result: dict, destination: str, city_filter: str | None, seen: set) -> bool:
    """Skips duplicates, and for places without their own guide, chunks that don't name them."""
    key = (result["city"], result["content"][:200])
    mentions_place = destination.lower() in result["content"].lower()
    if key in seen or (city_filter is None and not mentions_place):
        return False
    seen.add(key)
    return True


async def fixed_guide_search(destination: str, city_filter: str | None) -> list[dict]:
    searches = [
        search_knowledge_base(question.format(place=destination), city=city_filter, limit=RESULTS_PER_SEARCH * 2)
        for question, _ in GUIDE_SEARCHES
    ]
    chunks, seen = [], set()
    for results in await asyncio.gather(*searches):
        kept = [result for result in results if keep_result(result, destination, city_filter, seen)]
        chunks += kept[:RESULTS_PER_SEARCH]
    return chunks


async def react_guide_search(request: str, destination: str, city_filter: str | None, write) -> tuple[list[dict], list[str]]:
    """
    A small ReAct agent: the model calls search_guide with topics it picks from the
    request, sees what came back, and decides whether to search again.
    Returns the excerpts found and the queries it used.
    """
    chunks, seen, queries = [], set(), []

    @tool
    async def search_guide(topic: str) -> str:
        """Search the Wikivoyage travel guide for this trip's destination. Use a short topic, e.g. "street food markets" or "day hikes"."""
        if len(queries) >= MAX_GUIDE_SEARCHES:
            return "Search limit reached. Reply DONE."
        queries.append(topic)
        write({"node": "guide_agent", "message": f'Searching the guide for "{topic}"'})

        results = await search_knowledge_base(f"{topic} in {destination}", city=city_filter, limit=RESULTS_PER_SEARCH * 2)
        new = [result for result in results if keep_result(result, destination, city_filter, seen)][:RESULTS_PER_SEARCH]
        chunks.extend(new)
        if not new:
            return "Nothing new for that topic."
        return "\n".join(
            f"- {result['city']}, {result['section']}: {result['content'].split(chr(10), 1)[-1][:300]}"
            for result in new
        )

    agent = create_agent(llm, [search_guide], system_prompt=GUIDE_AGENT_PROMPT)
    try:
        # Each search is two steps (model, tool), plus the final answer.
        await agent.ainvoke(
            {"messages": [HumanMessage(content=f"Trip request: {request}\nDestination: {destination}")]},
            config={"recursion_limit": 2 * MAX_GUIDE_SEARCHES + 3},
        )
    except GraphRecursionError:
        pass  # it kept searching; keep what it found

    # Leave room for the core sections top_up() adds.
    return chunks[:MAX_EXCERPTS - len(CORE_SECTIONS)], queries


CORE_SECTIONS = ["See", "Eat", "Get around"]


def top_up(chunks: list[dict], extra: list[dict]) -> list[dict]:
    """Adds extra chunks up to MAX_EXCERPTS, first ones from core sections not covered yet."""
    keys = {(chunk["city"], chunk["content"][:200]) for chunk in chunks}
    extra = [chunk for chunk in extra if (chunk["city"], chunk["content"][:200]) not in keys]
    covered = {chunk["section"] for chunk in chunks}
    missing_first = sorted(extra, key=lambda chunk: (chunk["section"] in covered or chunk["section"] not in CORE_SECTIONS))

    result = list(chunks)
    for chunk in missing_first:
        if len(result) >= MAX_EXCERPTS:
            break
        result.append(chunk)
        covered.add(chunk["section"])
    return result


async def guide_agent(state: TravelState):
    destination = (state.get("destination") or "").strip()

    if not destination:
        return {"guide_context": "", "guide_sources": []}

    # Search only this city when it's in the knowledge base. Otherwise (e.g. "Bali",
    # a region) search every city and keep chunks that mention the place by name,
    # which finds Ubud and Kuta for Bali.
    city_filter = destination if await city_in_knowledge_base(destination) else None
    if not city_filter and "," in destination:
        # "Shibuya, Tokyo": no guide for the neighbourhood, so use the city's guide.
        for part in reversed([part.strip() for part in destination.split(",") if part.strip()]):
            if await city_in_knowledge_base(part):
                city_filter = destination = part
                break
    if city_filter:
        report(f"Searching the Wikivoyage guide for {destination}")
    else:
        report(f"{destination} has no guide of its own; searching guides that mention it")

    chunks = []
    if GUIDE_MODE == "react":
        try:
            # The tool runs inside the agent's own graph, so hand it this node's
            # stream writer for its status lines.
            chunks, queries = await react_guide_search(state["user_query"], destination, city_filter, get_stream_writer())
            print(f"Guide agent searched: {queries}")
        except Exception as e:
            print("Guide agent failed, using the fixed searches:", e)
        if not chunks:
            report("Using the standard guide searches")

    # The agent follows the traveller's interests but often stops after 3 searches
    # and can miss the basics (Bali: no food or transport). Fill the free slots from
    # the fixed searches, sections the plan always needs first.
    fixed = await fixed_guide_search(destination, city_filter)
    chunks = top_up(chunks, fixed)

    report(f"Found {len(chunks)} guide excerpts")

    excerpts, sources = [], []
    for number, chunk in enumerate(chunks, start=1):
        excerpts.append(f"[{number}] {chunk['content']}")
        sources.append({
            "number": number,
            "city": chunk["city"],
            "section": chunk["section"],
            "url": chunk["source_url"],
        })

    return {
        "guide_context": "\n\n".join(excerpts),
        "guide_sources": sources,
        "messages": [
            AIMessage(content=f"Found {len(chunks)} travel guide excerpts for {destination}.")
        ],
    }


def guide_prompt_section(state: TravelState) -> str:
    if state.get("guide_context"):
        return f"""
Travel Guide Excerpts (from Wikivoyage, numbered):
{state['guide_context']}

How to use the excerpts:
- Prefer places, tips and prices from the excerpts.
- Put the excerpt number in square brackets right after each fact you take from it, e.g. "Senso-ji temple [2]".
- Only cite a number if that excerpt really says it. Don't cite anything for general knowledge.
- Do not invent opening hours, prices or addresses that are not in the excerpts.
"""
    return """
Travel Guide Excerpts: none. The travel guide has no information for this destination.
Say briefly in the plan that the itinerary is general advice not based on a travel guide,
and do not invent specific prices, opening hours or addresses.
"""


def cited_numbers(text: str) -> set[int]:
    return {int(number) for number in re.findall(r"\[(\d{1,2})\]", text)}


def format_sources(plan: str, sources: list[dict]) -> str:
    """The Sources list is built by code, not the LLM, so the links are always real."""
    if not sources:
        return ""

    cited = cited_numbers(plan)
    lines = [
        f"- [{source['number']}] [Wikivoyage: {source['city']}, {source['section']}]({source['url']})"
        for source in sources if source["number"] in cited
    ]

    if not lines:
        return ""

    return (
        f"\n\n{SOURCES_HEADING}\n\n" + "\n".join(lines) +
        "\n\nTravel guide text from [Wikivoyage](https://en.wikivoyage.org), "
        "available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)."
    )


def strip_sources(plan: str) -> str:
    return plan.split(f"\n\n{SOURCES_HEADING}")[0]


# =========================
# Final Response Agent
# =========================

# One call writes the whole plan. It used to be two (draft itinerary, then a
# rewrite into sections), which doubled the run time on a local model.
async def final_agent(state: TravelState):
    final_prompt = f"""
Create the complete travel plan for the user.

User Request:
{state['user_query']}

Flights:
{flight_section(state)}

Hotels:
{state['hotel_results']}
{guide_prompt_section(state)}
Format the answer using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Make the plan practical, budget-aware, and easy to follow.
- Base the flight part of the budget on the real price in the Flights section, counted once, and state its currency.
- Mention any search assumptions listed in the Flights section, such as assumed travel dates.
- If the flight search failed, say so and do not invent flight prices.
- If flights or hotels were not needed, say so briefly in that section instead of suggesting any.
- If the Flights section says when the flight lands, start Day 1 after that time (an evening arrival gets only an evening plan) and keep the last day light for the flight home.
- If the Flights section is a flexible-date search, build the plan on the chosen flight's travel dates and say in the Trip Summary which dates were picked and why (for example the cheapest fare in the window).
- Do not add a sources or references section; it is added automatically.
"""

    # The model reads the whole prompt before writing; on a laptop that takes
    # 1-2 minutes, so say so instead of looking stuck.
    report("Reading the search results")
    # Async, so stopping the stream cancels the request to the model.
    response = await llm.ainvoke([
        SystemMessage(content="You are an expert travel planner and booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    plan = strip_sources(response.content)

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "messages": [response],
    }


# =========================
# Revise Agent
# =========================

async def revise_agent(state: TravelState):
    revise_prompt = f"""
Revise the user's existing travel plan based on their feedback.

Feedback:
{state['latest_input']}

Current Plan:
{strip_sources(state['final_plan'])}

Flights found earlier (real search results):
{flight_section(state)}

Hotels found earlier (real search results):
{state.get('hotel_results', '')}
{guide_prompt_section(state)}
Important:
- Apply the feedback and keep everything else the same.
- Keep the same six sections as the current plan.
- Keep existing citation numbers like [2] next to their facts; cite new facts from the excerpts the same way.
- Do not add a sources or references section; it is added automatically.
- Only use flights and prices from the search results above. Do not invent new ones.
- Return the full revised plan, not just the changes.
"""

    report("Reading your plan and the search results")
    response = await llm.ainvoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=revise_prompt)
    ])

    plan = strip_sources(response.content)

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "messages": [response],
    }


# =========================
# Critic and fix (reflection loop)
# =========================

# A rewrite takes minutes on a laptop, so allow one.
MAX_FIX_ATTEMPTS = 1


def travel_dates(state: TravelState) -> list[str]:
    """The dates the plan must use: the chosen flight's, or exact dates from the request."""
    option = state.get("chosen_option") or {}
    if option.get("outbound_date"):
        return [day for day in (option["outbound_date"], option.get("return_date")) if day]
    trip = state.get("trip_request") or {}
    if trip.get("flex_days") or trip.get("month"):
        return []  # no flight picked from a flexible search: any date in the window is fine
    return [day for day in (trip.get("departure_date"), trip.get("return_date")) if day]


def critic(state: TravelState):
    """
    Checks the plan with code (app/critic.py). Mechanical problems are fixed here;
    content problems go to fix_plan once. What still fails after that is shown in
    a note at the end of the plan.
    """
    report("Checking the budget, dates and citations")
    option = state.get("chosen_option") or {}
    review = review_plan(
        strip_sources(state.get("final_plan", "")),
        guide_context=state.get("guide_context", ""),
        flight_price=option.get("price"),
        travel_dates=travel_dates(state),
        arrival_time=option.get("arrival_time"),
    )

    plan = review.plan
    attempts = state.get("fix_attempts", 0)
    problems = review.problems

    if review.fixes:
        report("Fixed: " + "; ".join(review.fixes))
    if problems and attempts < MAX_FIX_ATTEMPTS:
        report(f"Found {len(problems)} problem{'s' if len(problems) > 1 else ''} to rewrite: " + " ".join(problems))
    elif problems:
        plan += ("\n\n> **Automatic check:** " + " ".join(problems) +
                 " A rewrite didn't fix this, so please double-check these parts.")
        report("Some problems remain; added a note to the plan")
        problems = []
    elif not review.fixes:
        report("No problems found")

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "critic_problems": problems,
    }


def route_after_critic(state: TravelState):
    return "fix_plan" if state.get("critic_problems") else "map_agent"


# =========================
# Map Agent (MCP client: the Mapbox MCP server)
# =========================

GETTING_AROUND = re.compile(r"\n## Getting around\b.*?(?=\n## |\Z)", re.S)


async def map_agent(state: TravelState):
    """
    Runs last, after the critic, so a rewrite can't drop its section. Places each
    day's stops on a map through the Mapbox MCP server and adds travel times.
    The plan is finished without a map if Mapbox isn't available.
    """
    plan = strip_sources(state.get("final_plan", ""))
    start, end = section_lines(plan, ["itinerary", "day-by-day", "day by day"])
    itinerary = "\n".join(plan.splitlines()[start:end])
    destination = state.get("destination") or ""

    if not itinerary.strip() or not destination:
        return {"map_data": {}}

    try:
        trip_map = await build_trip_map(llm, itinerary, destination, report=report)
    except Exception as e:
        print("Map agent failed:", repr(e))
        report("The map isn't available right now")
        return {"map_data": {}}

    for note in trip_map.notes:
        report(note)
    if not trip_map.section:
        return {"map_data": trip_map.data or {}}

    # A revision already has a Getting around section; replace it.
    plan = GETTING_AROUND.sub("", plan).rstrip() + "\n\n" + trip_map.section
    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "map_data": trip_map.data,
    }


async def fix_plan(state: TravelState):
    problems = "\n".join(f"- {problem}" for problem in state["critic_problems"])
    fix_prompt = f"""
Rewrite this travel plan to fix the problems listed. Keep everything else the same.

Problems found by an automatic check:
{problems}

Current plan:
{strip_sources(state['final_plan'])}

Flights (real search results):
{flight_section(state)}

Important:
- Keep the same six sections: Trip Summary, Flight Information, Hotel Suggestions,
  Day-by-Day Itinerary, Estimated Budget, Final Recommendations.
- Keep existing citation numbers like [2] next to the same facts. Don't add new ones.
- Do not add a sources or references section; it is added automatically.
- Return the full plan, not just the changes.
"""

    report("Rewriting the plan")
    response = await llm.ainvoke([
        SystemMessage(content="You are a careful editor of travel plans."),
        HumanMessage(content=fix_prompt)
    ])

    plan = strip_sources(response.content)

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "fix_attempts": state.get("fix_attempts", 0) + 1,
        "messages": [response],
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("router_agent", router_agent)
graph.add_node("plan_trip", plan_trip)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("guide_agent", guide_agent)
graph.add_node("choose_flight", choose_flight)
graph.add_node("final_agent", final_agent)
graph.add_node("revise_agent", revise_agent)
graph.add_node("critic", critic)
graph.add_node("fix_plan", fix_plan)
graph.add_node("map_agent", map_agent)

graph.add_edge(START, "router_agent")
graph.add_conditional_edges("router_agent", route_after_router, ["plan_trip", "revise_agent"])
# Fan out: the searches this trip needs run in the same step, in parallel.
graph.add_conditional_edges("plan_trip", route_searches, ["flight_agent", "hotel_agent", "guide_agent"])
# Fan in: the searches all run in one step, so choose_flight runs once after them,
# however many were skipped. (A list edge would wait forever for a skipped one.)
graph.add_edge("flight_agent", "choose_flight")
graph.add_edge("hotel_agent", "choose_flight")
graph.add_edge("guide_agent", "choose_flight")
graph.add_edge("choose_flight", "final_agent")
# Reflection: every plan is checked; a failed check gets one rewrite, then is checked again.
graph.add_edge("final_agent", "critic")
graph.add_edge("revise_agent", "critic")
graph.add_conditional_edges("critic", route_after_critic, ["fix_plan", "map_agent"])
graph.add_edge("fix_plan", "critic")
graph.add_edge("map_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================

# Compiled by open_travel_graph() at app startup, after open_db().
travel_graph = None


async def open_travel_graph():
    global travel_graph

    checkpointer = AsyncPostgresSaver(get_pool())
    await checkpointer.setup()

    travel_graph = graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

class LLMCallCounter(BaseCallbackHandler):
    """Counts chat model calls in one graph run, including failed and retried ones."""

    def __init__(self):
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.count += 1


def new_thread_id() -> str:
    return f"user_{uuid.uuid4().hex}"


def run_config(thread_id: str, llm_counter: LLMCallCounter, ask_flight_choice: bool = False) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            # Pause for the user to pick a flight; only the streaming UI can answer.
            "ask_flight_choice": ask_flight_choice,
        },
        "callbacks": [llm_counter],
    }


def run_input(user_input: str, dates: dict | None = None) -> dict:
    # Only send the new input. Earlier results in this thread are loaded from the
    # checkpoint, so feedback can revise the existing plan. date_overrides is
    # always sent, so form dates from an earlier run don't stick.
    return {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "latest_input": user_input,
        "date_overrides": dates or {},
        "fix_attempts": 0,
        "critic_problems": [],
    }


async def run_travel_agent(user_input: str, thread_id: str | None = None):
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    thread_id = thread_id or new_thread_id()
    llm_counter = LLMCallCounter()

    result = await travel_graph.ainvoke(
        run_input(user_input),
        config=run_config(thread_id, llm_counter)
    )

    return {
        "thread_id": thread_id,
        "answer": result.get("final_plan", ""),
        "route": result.get("route", ""),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "guide_sources": result.get("guide_sources", []),
        "llm_calls": llm_counter.count,
    }


# What the UI shows for each node in the progress list.
STEP_LABELS = {
    "router_agent": "Understanding your request",
    "plan_trip": "Working out the trip details",
    "flight_agent": "Searching flights",
    "hotel_agent": "Searching hotels",
    "guide_agent": "Searching the travel guide",
    "choose_flight": "Choosing your flight",
    "final_agent": "Writing your plan",
    "revise_agent": "Revising your plan",
    "critic": "Checking the plan",
    "fix_plan": "Fixing problems",
    "map_agent": "Mapping the itinerary",
}

ROUTE_STEPS = {
    "plan": ["plan_trip", "flight_agent", "hotel_agent", "guide_agent", "choose_flight", "final_agent", "critic", "map_agent"],
    "new_search": ["plan_trip", "flight_agent", "hotel_agent", "guide_agent", "choose_flight", "final_agent", "critic", "map_agent"],
    "revise": ["revise_agent", "critic", "map_agent"],
}

# Only these nodes' tokens are shown live; the router and trip parser return JSON.
WRITING_NODES = {"final_agent", "revise_agent", "fix_plan"}


def step_list(nodes: list[str]) -> list[dict]:
    return [{"node": node, "label": STEP_LABELS[node]} for node in nodes]


def pending_flight_choice(snapshot) -> dict | None:
    """The flight options a paused run is waiting on, or None if it isn't paused."""
    for item in snapshot.interrupts:
        if isinstance(item.value, dict) and item.value.get("type") == "choose_flight":
            return item.value
    return None


def skipped_steps(trip_request: dict) -> list[tuple[str, str]]:
    """The searches the supervisor skipped, with the reason shown in the UI."""
    skipped = []
    if not trip_request.get("needs_flights", True):
        skipped += [("flight_agent", "Not needed for this trip"), ("choose_flight", "No flights to choose")]
    if not trip_request.get("needs_hotels", True):
        skipped.append(("hotel_agent", "Not needed: you have somewhere to stay"))
    return skipped


async def stream_travel_agent(
    thread_id: str,
    user_input: str | None = None,
    flight_choice: int | None = None,
    dates: dict | None = None,
):
    """
    Runs the graph like run_travel_agent, but yields progress events as it goes:
      steps  - the list of steps this run will take (sent again once the route is known)
      step   - a node is running, done, failed, skipped or paused
      status - a status line from a node, e.g. "Found 5 options from AUD 1,354"
      token  - a piece of the plan as the model writes it
      choose - the run paused for the user to pick a flight
      rewrite - the critic sent the plan back; the text written so far will be replaced
      done   - the finished plan
    Pass user_input to start a run, or flight_choice (an option number, 0 to skip
    flights) to resume a paused one. An exception from the graph is raised to the caller.
    """
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    llm_counter = LLMCallCounter()
    config = run_config(thread_id, llm_counter, ask_flight_choice=True)

    if flight_choice is not None:
        graph_input = Command(resume=flight_choice)
        shown_steps = ["choose_flight", "final_agent", "critic", "map_agent"]
    else:
        graph_input = run_input(user_input, dates)
        shown_steps = ["router_agent"]
    yield {"type": "steps", "steps": step_list(shown_steps)}

    # choose_flight still runs (and passes straight through) when flights were
    # skipped; keep it shown as skipped.
    skipped_nodes = set()

    async for mode, chunk in travel_graph.astream(
        graph_input,
        config=config,
        stream_mode=["tasks", "custom", "messages"],
    ):
        if mode == "tasks":
            node = chunk["name"]
            if node not in STEP_LABELS or node in skipped_nodes:
                continue
            if "result" not in chunk:
                if node == "fix_plan" and node not in shown_steps:
                    # Only shown when the critic asks for a rewrite, before the map step.
                    shown_steps.insert(shown_steps.index("map_agent") if "map_agent" in shown_steps else len(shown_steps), node)
                    yield {"type": "steps", "steps": step_list(shown_steps)}
                if node == "fix_plan":
                    # The rewrite replaces the text written so far.
                    yield {"type": "rewrite"}
                yield {"type": "step", "node": node, "status": "running"}
            elif chunk.get("interrupts"):
                yield {"type": "step", "node": node, "status": "paused"}
            elif chunk["error"] is not None:
                yield {"type": "step", "node": node, "status": "failed"}
            else:
                yield {"type": "step", "node": node, "status": "done"}
                if node == "router_agent":
                    route = chunk["result"].get("route", "plan")
                    shown_steps = ["router_agent", *ROUTE_STEPS[route]]
                    yield {"type": "steps", "steps": step_list(shown_steps)}
                if node == "plan_trip":
                    for skipped, reason in skipped_steps(chunk["result"].get("trip_request", {})):
                        skipped_nodes.add(skipped)
                        yield {"type": "step", "node": skipped, "status": "skipped"}
                        yield {"type": "status", "node": skipped, "message": reason}

        elif mode == "custom":
            yield {"type": "status", "node": chunk["node"], "message": chunk["message"]}

        elif mode == "messages":
            message, metadata = chunk
            if metadata.get("langgraph_node") in WRITING_NODES and message.content:
                yield {"type": "token", "text": message.content}

    snapshot = await travel_graph.aget_state(config)

    choice = pending_flight_choice(snapshot)
    if choice:
        yield {"type": "choose", "thread_id": thread_id, "options": choice["options"]}
        return

    yield {
        "type": "done",
        "thread_id": thread_id,
        "answer": snapshot.values.get("final_plan", ""),
        "route": snapshot.values.get("route", ""),
        "map": snapshot.values.get("map_data") or None,
        "llm_calls": llm_counter.count,
    }


async def get_trip_plan(thread_id: str) -> dict:
    """The saved plan, and the flight options if the run is paused on a choice."""
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    snapshot = await travel_graph.aget_state({"configurable": {"thread_id": thread_id}})
    choice = pending_flight_choice(snapshot)
    return {
        "answer": snapshot.values.get("final_plan", ""),
        "flight_options": choice["options"] if choice else None,
        "map": snapshot.values.get("map_data") or None,
    }


async def delete_trip_state(thread_id: str):
    """Deletes every saved checkpoint of a thread (the plan, search results, pauses)."""
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    await travel_graph.checkpointer.adelete_thread(thread_id)
