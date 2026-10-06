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
from app.tools.google_flights_tool import search_google_flights
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
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    latest_input: str      # what the user just sent: a new trip or feedback
    user_query: str        # the current trip request the plan is built from
    route: str             # "plan", "new_search" or "revise"
    destination: str       # where the trip goes, as extracted from the request
    flight_results: str
    hotel_results: str
    guide_context: str     # numbered Wikivoyage excerpts for the itinerary prompt
    guide_sources: list[dict]
    itinerary: str
    final_plan: str
    llm_calls: int


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
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def route_after_router(state: TravelState):
    return "revise_agent" if state["route"] == "revise" else "flight_agent"


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
                '"return_date": null, "trip_days": 5, "adults": 1}'
            )),
            HumanMessage(content=query)
        ])
    except Exception as e:
        print("Trip extraction failed, falling back to regex:", e)
        return TripRequest()


def flight_agent(state: TravelState):
    query = state["user_query"]
    trip = extract_trip_request(query)

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

    if not destination_iata:
        flight_data = "Flight search skipped: could not work out the destination from the request."
    else:
        flight_data = search_google_flights(
            origin_iata=origin_iata,
            destination_iata=destination_iata,
            outbound_date=departure.isoformat(),
            return_date=return_day.isoformat() if return_day else None,
            adults=max(trip.adults, 1),
        )

    if assumptions:
        flight_data = "Assumptions: " + "; ".join(assumptions) + "\n\n" + flight_data

    return {
        "destination": trip.destination or "",
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
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
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}
{guide_prompt_section(state)}
Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Base the flight part of the budget on the real prices in the Flights section, and state its currency.
- Mention any search assumptions listed in the Flights section, such as assumed travel dates.
- If the flight search failed, say so and do not invent flight prices.
- Keep every citation number like [2] from the itinerary next to the same fact. Do not add new numbers.
- Do not add a sources or references section; it is added automatically.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    plan = strip_sources(response.content)

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Revise Agent
# =========================

def revise_agent(state: TravelState):
    revise_prompt = f"""
Revise the user's existing travel plan based on their feedback.

Feedback:
{state['latest_input']}

Current Plan:
{strip_sources(state['final_plan'])}

Flights found earlier (real search results):
{state.get('flight_results', '')}

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

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=revise_prompt)
    ])

    plan = strip_sources(response.content)

    return {
        "final_plan": plan + format_sources(plan, state.get("guide_sources", [])),
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("router_agent", router_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("guide_agent", guide_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)
graph.add_node("revise_agent", revise_agent)

graph.add_edge(START, "router_agent")
graph.add_conditional_edges("router_agent", route_after_router, ["flight_agent", "revise_agent"])
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "guide_agent")
graph.add_edge("guide_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
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

async def run_travel_agent(user_input: str, thread_id: str | None = None):
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Only send the new input. Earlier results in this thread are loaded from the
    # checkpoint, so feedback can revise the existing plan.
    result = await travel_graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "latest_input": user_input,
            "llm_calls": 0
        },
        config=config
    )

    return {
        "thread_id": thread_id,
        "answer": result.get("final_plan", ""),
        "route": result.get("route", ""),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "guide_sources": result.get("guide_sources", []),
        "llm_calls": result.get("llm_calls", 0),
    }


async def get_trip_plan(thread_id: str):
    if travel_graph is None:
        raise RuntimeError("Travel graph is not open. Call open_travel_graph() first.")

    snapshot = await travel_graph.aget_state({"configurable": {"thread_id": thread_id}})
    return snapshot.values.get("final_plan", "")
