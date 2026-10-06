# Roamer AI — Extension Plan


Goal: extend Roamer AI to include user auth, plan revision, RAG, MCP, and more functionalities.

---

## Stack and API changes

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
- [x] Add a `users` table and simple JWT auth in FastAPI so each user has their own trips and threads
- [x] Scope every user-data query by `user_id` in the app (no Supabase RLS here)

How it's built:

- `backend/app/db.py`: one connection pool shared by the checkpointer and the app; creates `users` and `trips` at startup. `trips` links each graph thread to its owner
- `backend/app/auth.py`: Argon2 password hashes (`pwdlib`), 7-day HS256 JWTs (`pyjwt`), and a `get_current_user` dependency
- `backend/app/trips.py`: trip queries, all filtered by `user_id`
- Endpoints: `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, `GET /api/trips`; `/api/travel` routes require login
- Someone else's thread, or a made-up thread id, returns the same 404 as a missing one; only the server creates thread ids
- Frontend: login/register form, token sent on every request, My Trips list, Log out
- Tested with two accounts: user B gets 404 for user A's plan and feedback, and sees an empty trip list

## Phase 2 — RAG implementation

- [x] Download Wikivoyage guides (free, CC BY-SA) for the 500 most-read city articles, through the API instead of the full dumps
- [x] Chunk and embed them into a pgvector table
- [x] Add a `tsvector` column for full-text search
- [x] Hybrid retrieval: vector + full-text results merged with reciprocal rank fusion (RRF)
- [x] Itinerary output cites its sources

How it's built:

- `backend/scripts/rank_cities.py`: sums 12 months of Wikimedia "top 1000" page views, keeps `Category:City articles`, takes the top 500, looks up each city's current country on Wikidata → `backend/data/kb_cities.json`
- `backend/scripts/build_kb.py`, three resumable stages:
  - `fetch`: plain-text guides with section headings, cached in `backend/data/cache/` (not committed)
  - `chunk`: split by top-level section, ~350 words, 40-word overlap, each chunk starts with `City, Country — Section`
  - `embed`: `bge-m3` (1024 dims) via Ollama, one transaction per city, `COPY` for speed, skips cities already loaded from the same revision
- Tables `kb_cities` and `kb_chunks` (HNSW cosine index on `embedding`, GIN index on the generated `tsv` column)
- `backend/app/knowledge_base.py`: one SQL query runs both searches (40 candidates each), ranks keyword matches containing all the words first, and fuses with `1 / (60 + rank)`; the optional city filter uses pgvector 0.8 iterative index scans
- `guide_agent` node (flight → hotel → **guide** → itinerary → final): 4 searches (sights, things to do, food, transport), 3 results each, deduplicated, numbered. Destinations not in the knowledge base (e.g. Bali) fall back to all cities and keep chunks that mention the place (finds Ubud, Kuta, Nusa Lembongan)
- The itinerary cites excerpts as `[n]`; the Sources list and CC BY-SA attribution are added by code, not the LLM; `num_ctx` raised to 16K so the excerpts aren't cut off

Measured:

- 500 cities, 20,963 chunks, 377 MB of the 1 GB Render limit (HNSW index 164 MB); embedding took 107 min on a laptop
- Tokyo and Bali test plans cited 7 sources each; full runs took about 10–12 min on `qwen3:8b` (the larger prompt made them slower)
- HNSW vs exact search at 21K vectors: same recall and same speed, because the free-tier database reads the index from disk on every search

Known issues:

- Some citations are wrong: the Tokyo test plan cited a guide excerpt for a hotel that came from the Tavily search. Measure citation accuracy in Phase 5
- Fallback searches (destination not in the knowledge base) take about 16 s because common words match thousands of chunks
- Singapore, Bali and Macau are not city articles on Wikivoyage, so they aren't in the list (Bali works through the fallback)

Why it matters: hybrid search with citations stands out from the usual "I called a vector DB" project.

## Phase 3 — Make the agents decide things

The planner used to be one-off: each message started from scratch and ignored the saved thread.

```
START → router_agent ─┬─ "plan" / "new_search" → plan_trip ─┬─ flight ─┬─→ final_agent → END
                      │                                     ├─ hotel ──┤
                      │                                     └─ guide ──┘
                      └─ "revise" ─────────────→ revise_agent ──────────────→ END
```

- `router_agent`: first message in a thread → `plan`. Later messages are feedback; the LLM (structured output `FeedbackDecision`) picks:
  - `new_search`: changes dates, places, travellers, or asks for different flights/hotels → rewrites the trip request and reruns the searches
  - `revise`: edits the existing plan using the saved search results (no API calls)
- State now keeps `final_plan`, `latest_input` and `route`; `run_travel_agent` only sends the new input, so earlier results load from the checkpoint
- New endpoint `GET /api/travel/{thread_id}` restores a plan; the UI restores it on page load
- UI: after the first plan, the input becomes "Revise Plan"; a **New Trip** button starts a new thread
- Tested: plan → "add more food spots on day 2" (`revise`, no searches) → "make it 5 days" (`new_search`, return date moved 11-22 → 11-24)

Stage A (done):

- [x] Run flight, hotel and guide searches **in parallel**: a new `plan_trip` node extracts the trip once, then the three searches fan out and `final_agent` waits for all of them (search stage 9 s → 5 s)
- [x] Fix `llm_calls`: a LangChain callback (`LLMCallCounter.on_chat_model_start`) counts real chat-model calls, including failed ones; removed the hand-written counter from the state and every node
- [x] Merged `itinerary_agent` into `final_agent`: one call writes the six-section plan instead of a draft plus a rewrite

Measured (Tokyo, local `qwen3:8b` on an M2):

- Generating text is the bottleneck: about 6 tokens/s, so 80% of the run is the model writing
- Tokyo 622 s → 373 s, Bali 741 s → 432 s; all six sections present; LLM calls per plan 3 → 2
- Known issue (both before and after): the 8B model mishandles the flight budget, e.g. counts a round-trip fare twice or leaves it out, and invents return-leg flight numbers. The Stage E critic should catch this

Stage B (done): live progress instead of a spinner

- [x] Graph: `stream_travel_agent()` runs `astream()` with three modes:
  - `tasks`: a node started or finished, which drives the ○ / ● / ✓ / ✗ step list
  - `custom`: status lines sent by the agents with `get_stream_writer()`, e.g. "Searching Google Flights MEL → NRT", "Found 5 options from AUD 1,174"
  - `messages`: the plan as the model writes it, from `final_agent` and `revise_agent` only; the router's and trip parser's JSON stays hidden
- [x] `final_agent` and `revise_agent` use `await llm.ainvoke()`, so cancelling the run also cancels the request to Ollama (its log shows `cancel task`)
- [x] Backend: `POST /api/travel/stream` sends Server-Sent Events (`steps`, `step`, `status`, `token`, `done`, `error`). It uses the same login and ownership checks, saves the trip only on `done`, and sends a `: ping` heartbeat every 10 s from a queue, so the connection stays open during the model's silent reading phase. Closing the connection cancels the graph run. `POST /api/travel` stays for curl and scripts
- [x] Frontend: `fetch()` + `response.body.getReader()` (`EventSource` can't send the login token); step list with the latest status line for each step; the plan renders live, at most every 200 ms; Markdown is cleaned with DOMPurify, which closes the old `marked` → `innerHTML` XSS hole
- [x] Stuck warnings: "connection may have dropped" after 30 s with no data (heartbeats included), and "slower than usual" after 4 min without a progress event. The plan's original "90 s with no events" rule was dropped because the model spends 60–120 s reading the prompt before its first token
- [x] Stop button: aborts the fetch, the server cancels the run, nothing is saved, and the previous plan is shown again

Measured: first token after about 60 s of reading; a 2-day Rome plan took 5:18 in the browser. Timers and token/word counts were tried and removed to keep the panel simple.

Known issue found while testing: a revision ("add more food spots on day 2") ran away. The model copied the raw hotel and guide text into the plan until it hit the 16K context limit (7,900 tokens, 26 min). Capped responses at 3,000 tokens (`OLLAMA_NUM_PREDICT`, about twice a normal plan); the Stage E critic is the real fix.

### Still to do

- [ ] **Supervisor/router** node with conditional edges (e.g. skip flights when the user says "I'm driving")
- [ ] Turn the agents into **tool-calling ReAct agents** that choose their own tools
- [ ] **Human-in-the-loop:** LangGraph `interrupt()` so the user approves the flight before the itinerary is built
- [ ] **Reflection loop:** a critic node checks the plan against budget and dates, and the graph replans if it fails
- [ ] **Trip dates with flexibility:** the user picks a start and end date, and can mark each as flexible
  - UI: start and end date pickers, plus a flexibility option for each date:
    - Exact: only that date
    - ± 1 / 2 / 3 days around the date
    - Whole month: "anytime in November"
  - Optional trip length (e.g. 5–7 nights) when the dates are loose
  - Backend: add `start_date`, `end_date`, `flex_days` and `trip_length` to `TripRequest`. The LLM parser fills them from free text ("around mid-Nov for a week"), and the form values override it
  - Flights: search each date in the flexible window and show the cheapest combinations (SerpApi Google Flights price insights). Cap the number of searches so the API quota isn't burned
  - Hotels and itinerary: use the date range chosen by the user. Say in the plan which dates were picked and why (e.g. "cheapest fares")
  - Router: feedback like "make the dates flexible" or "go a week later" counts as `new_search`
  - Validation: end date must be after the start date, and neither can be in the past

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
