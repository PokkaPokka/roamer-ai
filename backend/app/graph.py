import os 
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated, Literal
from datetime import date, timedelta
import asyncio
import operator
import re
import uuid

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_config, get_stream_writer
from langgraph.types import Command, interrupt
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
    destination: str       # where the trip goes, as extracted from the request
    flight_results: str
    flight_options: list[dict]   # the same flights as data, for the UI's flight cards
    chosen_flight: str     # the flight the user picked, as text for the plan prompt
    hotel_results: str
    guide_context: str     # numbered Wikivoyage excerpts for the plan prompt
    guide_sources: list[dict]
    final_plan: str


# =========================
# Router Agent
# =========================

class FeedbackDecision(BaseModel):
    action: Literal["revise", "new_search"] = Field(
        description=(
            "'new_search' if the feedback changes the destination, origin, travel dates, "
            "number of travellers, or asks for different flights or hotels. "
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
                '"return_date": null, "trip_days": 5, "adults": 1, "needs_flights": true, "needs_hotels": true}\n'
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
    r"\b(driv\w*|road ?trip|by (car|train|bus|ferry)|take the (train|bus)"
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


def plan_trip(state: TravelState):
    # Extract the trip once, so the searches can all start at the same time, and
    # decide which searches this trip needs (see route_searches).
    report("Reading your trip request")
    trip = extract_trip_request(state["user_query"])
    if not trip.needs_flights and not NO_FLIGHT_HINTS.search(state["user_query"]):
        trip.needs_flights = True
    if not trip.needs_hotels and not NO_HOTEL_HINTS.search(state["user_query"]):
        trip.needs_hotels = True

    details = [part for part in [
        f"to {trip.destination}" if trip.destination else "",
        f"from {trip.origin}" if trip.origin else "",
        f"leaving {trip.departure_date}" if trip.departure_date else "",
        f"{trip.trip_days} days" if trip.trip_days else "",
    ] if part]
    report("Trip: " + ", ".join(details) if details else "Could not find trip details; using defaults")

    updates = {
        "trip_request": trip.model_dump(),
        "destination": trip.destination or "",
        # Clear the previous run's flight choice; a new search asks again.
        "flight_options": [],
        "chosen_flight": "",
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
    if not departure:
        departure = today + timedelta(days=30)
        assumptions.append(f"no valid travel date given, so searched {departure.isoformat()} (30 days from today)")

    return_day = parse_future_date(trip.return_date, departure)
    if not return_day and trip.trip_days:
        return_day = departure + timedelta(days=max(trip.trip_days - 1, 1))

    flight_options = []

    if not destination_iata:
        flight_data = "Flight search skipped: could not work out the destination from the request."
        report("Skipped: could not work out the destination airport")
    else:
        report(f"Searching Google Flights {origin_iata} → {destination_iata}, {departure.isoformat()}")
        flight_data, flight_options = search_google_flights_with_options(
            origin_iata=origin_iata,
            destination_iata=destination_iata,
            outbound_date=departure.isoformat(),
            return_date=return_day.isoformat() if return_day else None,
            adults=max(trip.adults, 1),
        )
        prices = [option["price"] for option in flight_options if option["price"] is not None]
        if prices:
            report(f"Found {len(flight_options)} options from {flight_options[0]['currency']} {min(prices):,}")
        else:
            report("No flight options found")

    if assumptions:
        flight_data = "Assumptions: " + "; ".join(assumptions) + "\n\n" + flight_data

    return {
        "flight_results": flight_data,
        "flight_options": flight_options,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
    }


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
    if option["round_trip"]:
        lines.append(
            "Only the outbound flights are listed; the return flight is picked when booking, "
            "so don't invent return flight numbers or times."
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
        return {"chosen_flight": NO_FLIGHT_CHOSEN}

    adults = max(state.get("trip_request", {}).get("adults") or 1, 1)
    report(f"Using option {chosen['number']}: {', '.join(chosen['airlines'])}, "
           f"{chosen['currency']} {chosen['price']:,}")
    return {"chosen_flight": chosen_flight_text(chosen, adults)}


def flight_section(state: TravelState) -> str:
    """The Flights part of the plan prompt: the chosen flight instead of every option."""
    results = state.get("flight_results", "")
    chosen = state.get("chosen_flight")
    if not chosen:
        return results
    # Keep the summary lines (assumptions, route, price insight) but not the option list.
    summary = results.split("\n\nOption 1:")[0]
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


async def guide_agent(state: TravelState):
    destination = (state.get("destination") or "").strip()

    if not destination:
        return {"guide_context": "", "guide_sources": []}

    # Search only this city when it's in the knowledge base. Otherwise (e.g. "Bali",
    # a region) search every city and keep chunks that mention the place by name,
    # which finds Ubud and Kuta for Bali.
    city_filter = destination if await city_in_knowledge_base(destination) else None
    if city_filter:
        report(f"Searching the Wikivoyage guide for {destination}")
    else:
        report(f"{destination} has no guide of its own; searching guides that mention it")

    searches = [
        search_knowledge_base(question.format(place=destination), city=city_filter, limit=RESULTS_PER_SEARCH * 2)
        for question, _ in GUIDE_SEARCHES
    ]
    result_lists = await asyncio.gather(*searches)

    chunks, seen = [], set()
    for results in result_lists:
        kept = 0
        for result in results:
            key = (result["city"], result["content"][:200])
            mentions_place = destination.lower() in result["content"].lower()
            if key in seen or (city_filter is None and not mentions_place):
                continue
            seen.add(key)
            chunks.append(result)
            kept += 1
            if kept == RESULTS_PER_SEARCH:
                break

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
graph.add_edge("final_agent", END)
graph.add_edge("revise_agent", END)


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


def run_input(user_input: str) -> dict:
    # Only send the new input. Earlier results in this thread are loaded from the
    # checkpoint, so feedback can revise the existing plan.
    return {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "latest_input": user_input,
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
}

ROUTE_STEPS = {
    "plan": ["plan_trip", "flight_agent", "hotel_agent", "guide_agent", "choose_flight", "final_agent"],
    "new_search": ["plan_trip", "flight_agent", "hotel_agent", "guide_agent", "choose_flight", "final_agent"],
    "revise": ["revise_agent"],
}

# Only these nodes' tokens are shown live; the router and trip parser return JSON.
WRITING_NODES = {"final_agent", "revise_agent"}


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


async def stream_travel_agent(thread_id: str, user_input: str | None = None, flight_choice: int | None = None):
    """
    Runs the graph like run_travel_agent, but yields progress events as it goes:
      steps  - the list of steps this run will take (sent again once the route is known)
      step   - a node is running, done, failed, skipped or paused
      status - a status line from a node, e.g. "Found 5 options from AUD 1,354"
      token  - a piece of the plan as the model writes it
      choose - the run paused for the user to pick a flight
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
        yield {"type": "steps", "steps": step_list(["choose_flight", "final_agent"])}
    else:
        graph_input = run_input(user_input)
        yield {"type": "steps", "steps": step_list(["router_agent"])}

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
                yield {"type": "step", "node": node, "status": "running"}
            elif chunk.get("interrupts"):
                yield {"type": "step", "node": node, "status": "paused"}
            elif chunk["error"] is not None:
                yield {"type": "step", "node": node, "status": "failed"}
            else:
                yield {"type": "step", "node": node, "status": "done"}
                if node == "router_agent":
                    route = chunk["result"].get("route", "plan")
                    yield {"type": "steps", "steps": step_list(["router_agent", *ROUTE_STEPS[route]])}
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
    }


async def delete_trip_state(thread_id: str):
    """Deletes every saved checkpoint of a thread (the plan, search results, pauses)."""
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    await travel_graph.checkpointer.adelete_thread(thread_id)
