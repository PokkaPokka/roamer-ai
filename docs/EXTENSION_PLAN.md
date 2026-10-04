# TripMate AI — Extension Plan

Goal: turn TripMate into a resume project for **Agentic AI System** roles.

## Known weakness to fix

The current "multi-agent" graph is a fixed chain of four steps:
`flight_agent → hotel_agent → itinerary_agent → final_agent`.

- `hotel_agent` makes no LLM call (`flight_agent` now does — see below).
- ~~No node makes a decision~~ → `router_agent` now routes feedback (see below). The planning path itself is still a fixed chain.
- `llm_calls` counts the hotel tool step too, so it reports 4 when only 3 are real LLM calls.

---

## Stack and API changes

All services now run on free tiers.

| Area         | Before                                                                   | Now                                                          | Why                                                      |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------- |
| Database     | Render Postgres                                                          | Render Postgres (kept), pgvector 0.8.1 enabled               | Supabase free project limit reached                      |
| LLM          | Groq `llama-3.3-70b-versatile`                                           | Local Ollama `qwen3:8b` by default with Groq still swappable | Zero cost during development                             |
| Flights      | AviationStack (live status only, no fares, no future dates on free plan) | SerpApi Google Flights                                       | AviationStack can't price a future trip                  |
| Trip parsing | Regex only                                                               | LLM structured output (`TripRequest`) with regex fallback    | Extracts dates and traveller count; better agentic story |
| Hotels       | Tavily                                                                   | Tavily                                                       | —                                                        |

---

## Phase 1 — Harden the Render Postgres backend

- [x] Enable pgvector: run `CREATE EXTENSION IF NOT EXISTS vector;` on the Render database
- [x] Replace the single shared `psycopg` connection with `psycopg_pool` + `AsyncPostgresSaver`, and make `run_travel_agent` async (verified: `/health` answers in 10 ms while a plan is generating)
- [ ] Add a `users` table and simple JWT auth in FastAPI so each user has their own trips and threads
- [ ] Scope every user-data query by `user_id` in the app (no Supabase RLS here)
- [ ] Check the free-tier limits in the Render dashboard (1 GB storage; free databases expire after a set period). Keep the RAG data small enough to fit

## Phase 2 — RAG implementation

- [ ] Download Wikivoyage dumps (free, CC-BY-SA); pick about 50–100 destinations to stay under the 1 GB limit
- [ ] Chunk and embed them into a pgvector table
- [ ] Add a `tsvector` column for full-text search
- [ ] Hybrid retrieval: vector + full-text results merged with reciprocal rank fusion (RRF)
- [ ] Itinerary output cites its sources

Why it matters: hybrid search with citations stands out from the usual "I called a vector DB" project.

## Phase 3 — Make the agents decide things

The planner used to be one-off: each message started from scratch and ignored the saved thread.

```
START → router_agent ─┬─ "plan" / "new_search" → flight → hotel → itinerary → final → END
                      └─ "revise" ─────────────→ revise_agent ──────────────────────→ END
```

- `router_agent`: first message in a thread → `plan`. Later messages are feedback; the LLM (structured output `FeedbackDecision`) picks:
  - `new_search`: changes dates, places, travellers, or asks for different flights/hotels → rewrites the trip request and reruns the searches
  - `revise`: edits the existing plan using the saved search results (no API calls)
- State now keeps `final_plan`, `latest_input` and `route`; `run_travel_agent` only sends the new input, so earlier results load from the checkpoint
- New endpoint `GET /api/travel/{thread_id}` restores a plan; the UI restores it on page load
- UI: after the first plan, the input becomes "Revise Plan"; a **New Trip** button starts a new thread
- Tested: plan → "add more food spots on day 2" (`revise`, no searches) → "make it 5 days" (`new_search`, return date moved 11-22 → 11-24)

### Still to do

- [ ] **Supervisor/router** node with conditional edges (e.g. skip flights when the user says "I'm driving")
- [ ] Run flight and hotel searches **in parallel** (they currently run one after the other)
- [ ] Turn the agents into **tool-calling ReAct agents** that choose their own tools
- [ ] **Human-in-the-loop:** LangGraph `interrupt()` so the user approves the flight before the itinerary is built
- [ ] **Reflection loop:** a critic node checks the plan against budget and dates, and the graph replans if it fails
- [ ] Fix `llm_calls` so it counts only real LLM calls

## Phase 4 — MCP in both directions

- [ ] **MCP server:** expose `search_flights`, `search_hotels`, `query_travel_kb` and `get_my_trips`
- [ ] Demo the server working inside Claude Desktop or Cursor (record a GIF for the README)
- [ ] **MCP client:** the graph uses external MCP servers (weather, maps) through `langchain-mcp-adapters`

Why it matters: building both sides shows you understand the protocol, not just one SDK.

## Phase 5 — Evals and observability

- [ ] LangSmith tracing on every graph run
- [ ] Golden set of about 30 trip queries
- [ ] Ragas metrics: faithfulness and context precision
- [ ] Put the before/after numbers in the README (e.g. "faithfulness 0.71 → 0.89 after hybrid search")

Why it matters: a measured improvement is the kind of result hiring managers look for.
