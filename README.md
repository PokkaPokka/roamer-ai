# ✈️ Roamer AI — A Multi-Agent Travel Planner with LangGraph

An open-source AI travel planner that turns a natural-language trip request into a practical travel plan with flight suggestions, hotel ideas, and a day-by-day itinerary. The project uses a multi-agent workflow built with LangGraph, LangChain, and FastAPI.

## Why this project?

Planning a trip usually means jumping between multiple websites, tools, and spreadsheets. This project brings that flow into one experience by combining:

- a trip-parsing step,
- flight-search, hotel-research and travel-guide agents that run in parallel, and
- a planner agent that writes the cited plan,

all coordinated through a LangGraph workflow.

## Features

- ✈️ Real flight fares from Google Flights (via SerpApi), with LLM extraction of trip details
- 🏨 Hotel suggestions using Tavily search
- 🧠 Multi-agent orchestration with LangGraph
- 📝 Structured travel itinerary generation
- 📚 RAG over Wikivoyage guides for 500 popular cities: hybrid search (pgvector + Postgres full-text, merged with RRF) and itineraries that cite their sources
- 🌐 FastAPI backend with a simple web interface
- 💬 Give feedback on a plan and the agents revise it
- 🔐 User accounts (JWT); each user only sees their own trips
- 💾 Conversation state persistence using PostgreSQL
- ⚡ Local LLM with Ollama (free), or hosted with Groq

## Tech Stack

- Python 3.10+
- FastAPI
- Jinja2 + HTML/CSS/JavaScript frontend
- LangGraph
- LangChain
- Groq LLMs
- PostgreSQL
- Tavily API
- SerpApi (Google Flights)

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py       # FastAPI app entry point and routes
│   │   ├── graph.py      # LangGraph travel workflow
│   │   ├── db.py         # PostgreSQL pool and app tables
│   │   └── tools/        # Flight and web search integrations
│   ├── scripts/
│   │   └── run_agent.py  # Run one request from the terminal
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── templates/        # HTML templates
│   └── static/           # JavaScript and CSS
├── docs/                 # Extension plan and diagrams
└── Dockerfile
```

## Prerequisites

Before running the project locally, make sure you have:

- Python 3.10 or newer installed
- PostgreSQL running and accessible
- API keys for:
  - Groq
  - Tavily
  - SerpApi (free plan: 250 searches/month)

## Environment Variables

Create a .env file in the project root with the following variables:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/travel_db
GROQ_API_KEY=your_groq_api_key
SERPAPI_API_KEY=your_serpapi_api_key
FLIGHT_CURRENCY=AUD
TAVILY_API_KEY=your_tavily_api_key
DEFAULT_ORIGIN_IATA=DAC

# Signs login tokens. Use a long random string, e.g. the output of:
# python -c "import secrets; print(secrets.token_urlsafe(48))"
JWT_SECRET=your_long_random_secret

# LLM provider: "ollama" (local, free) or "groq" (hosted, needs GROQ_API_KEY)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
OLLAMA_NUM_CTX=16384   # context window; guide excerpts need about 8K tokens
OLLAMA_NUM_PREDICT=3000 # max tokens per response; stops a runaway answer
FLEX_MAX_SEARCHES=7       # max Google Flights searches for one flexible-date trip
GUIDE_MODE=react          # "react": the guide agent picks its own searches; "fixed": 4 set searches
MAPBOX_ACCESS_TOKEN=pk... # Mapbox public token for the itinerary map (optional; the plan works without it)
```

For local development, install [Ollama](https://ollama.com), then run `ollama pull qwen3:8b` and `ollama pull bge-m3` (the embedding model for the travel guide search).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Running the App

Start the FastAPI server from the `backend` folder:

```bash
cd backend
python -m app.main
```

Then open your browser at:

```text
http://127.0.0.1:8000/
```

To run one request in the terminal instead: `python -m scripts.run_agent` (also from `backend`).

## Building the Travel Guide Knowledge Base

The guide agent searches Wikivoyage guides stored in PostgreSQL (pgvector). Build them once, from the `backend` folder:

```bash
python -m scripts.rank_cities          # pick the 500 most-read Wikivoyage city guides -> data/kb_cities.json (already committed)
python -m scripts.build_kb fetch       # download the guides to data/cache/ (about 30 min, rate limited)
python -m scripts.build_kb chunk       # split them into ~350-word chunks
python -m scripts.build_kb embed       # embed with bge-m3 and load into Postgres (about 2 hours on a laptop)
python -m scripts.search_kb "cheap places to eat" --city Ubud   # try the hybrid search
```

Every stage can be stopped and re-run; finished work is skipped. The full knowledge base is about 21,000 chunks and 370 MB.

## API Endpoints

- GET /health - Health check
- POST /api/auth/register - Create an account, returns a token
- POST /api/auth/login - Log in, returns a token
- GET /api/auth/me - The logged-in user
- GET /api/trips - Your saved trips
- PATCH /api/trips/{thread_id} - Rename a trip (`{"title": "..."}`)
- DELETE /api/trips/{thread_id} - Delete a trip and its saved plan
- POST /api/travel - Submit a travel request, or feedback on an existing trip (`thread_id`)
- POST /api/travel/stream - Same as above, but streams progress as Server-Sent Events (`steps`, `step`, `status`, `token`, `choose`, `done`, `error`); the web UI uses this. It pauses with a `choose` event so the user can pick a flight; resume with `{"thread_id": ..., "flight_choice": <option number, or 0 for none>}`. One plan per account runs at a time; a second request gets 409 (also `POST /api/travel`)
- GET /api/travel/{thread_id} - Get the latest plan for one of your trips

All `/api/trips` and `/api/travel` routes need an `Authorization: Bearer <token>` header.

Example requests:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"your-password"}' | python -c "import sys,json; print(json.load(sys.stdin)['token'])")

curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"Plan a 3-day trip to Tokyo from Melbourne leaving 2026-11-20"}'
```

## MCP: Roamer in Claude Desktop, and Mapbox in Roamer

Roamer speaks the Model Context Protocol in both directions.

**As a server**, `backend/app/mcp_server.py` gives any MCP client these tools: `search_flights` (Google Flights fares), `search_hotels`, `search_travel_guide` (the Wikivoyage knowledge base, with source links), `list_my_trips` and `get_trip_plan`. To use it in Claude Desktop:

1. Make a token for your Roamer account (trip tools only see that account's trips), from the `backend` folder:
   `python -m scripts.mcp_token --email you@example.com` (valid 90 days; treat it like a password)
2. Add Roamer to `~/Library/Application Support/Claude/claude_desktop_config.json` with absolute paths:

```json
{
  "mcpServers": {
    "roamer": {
      "command": "/path/to/repo/.venv/bin/python",
      "args": ["/path/to/repo/backend/app/mcp_server.py"],
      "env": { "ROAMER_TOKEN": "<token from step 1>" }
    }
  }
}
```

3. Restart Claude Desktop and ask, e.g. "Find flights from Melbourne to Tokyo for 20–23 November and what the guide says about Asakusa". Ollama must be running for the guide search (it embeds the question).

**As a client**, the map agent starts the official [Mapbox MCP server](https://docs.mapbox.com/api/guides/mcp-server) (`npx @mapbox/mcp-server`, needs Node 22+ and `MAPBOX_ACCESS_TOKEN`) through `langchain-mcp-adapters`, geocodes each day's places and gets travel times between them. The plan gets a "Getting around" section and the web page shows a map of each day's stops.

## How the Workflow Works

1. The user submits a travel request (or feedback on an existing plan, which the router sends to a revise step or a new search).
2. The plan step (the supervisor) extracts the trip details once and decides which searches are needed, e.g. no flight search for "I'm driving".
3. The needed agents run in parallel: flights (Google Flights), hotels (Tavily), and the guide agent, a small ReAct agent that chooses its own searches of the Wikivoyage knowledge base (hybrid search) from the traveller's interests.
4. The graph pauses (LangGraph `interrupt()`) so the user can pick a flight; the paused state is saved in Postgres.
5. The final agent writes the plan around the chosen flight in one LLM call and cites the excerpts it used, like `[2]`.
6. A critic checks the plan with code: it fixes the budget total, the flight cost and mismatched citations itself, and sends missing sections or wrong dates back for one rewrite.
7. The map agent puts each day's stops on a map through the Mapbox MCP server and adds travel times.
8. The code appends a Sources list with links for the cited excerpts.

Run the critic's tests from the `backend` folder: `python -m unittest tests.test_critic`.

## Contributing

Contributions are welcome. If you want to improve the app, add new travel features, or fix issues:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request

## Acknowledgments

This project is built with the help of modern LLM tooling and travel APIs, and it is intended as a practical example of combining LangGraph agents with real-world applications.

Travel guide content comes from [Wikivoyage](https://en.wikivoyage.org) and is used under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) licence. Plans link to the Wikivoyage articles they cite.
