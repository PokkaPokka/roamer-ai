# ✈️ TripMate AI — A Multi-Agent Travel Planner with LangGraph

An open-source AI travel planner that turns a natural-language trip request into a practical travel plan with flight suggestions, hotel ideas, and a day-by-day itinerary. The project uses a multi-agent workflow built with LangGraph, LangChain, and FastAPI.

## Why this project?

Planning a trip usually means jumping between multiple websites, tools, and spreadsheets. This project brings that flow into one experience by combining:

- a flight-search agent,
- a hotel-research agent,
- an itinerary-planning agent, and
- a final response agent,

all coordinated through a LangGraph workflow.

## Features

- ✈️ Real flight fares from Google Flights (via SerpApi), with LLM extraction of trip details
- 🏨 Hotel suggestions using Tavily search
- 🧠 Multi-agent orchestration with LangGraph
- 📝 Structured travel itinerary generation
- 🌐 FastAPI backend with a simple web interface
- 💾 Conversation state persistence using PostgreSQL
- ⚡ LLM-powered responses with Groq

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

# LLM provider: "ollama" (local, free) or "groq" (hosted, needs GROQ_API_KEY)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b
```

For local development, install [Ollama](https://ollama.com), then run `ollama pull qwen3:8b`.

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

## API Endpoints

- GET /health - Health check
- POST /api/travel - Submit a travel request

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Tokyo with a budget of $1200"}'
```

## How the Workflow Works

1. The user submits a travel request.
2. The flight agent gathers flight-related information.
3. The hotel agent searches for accommodation suggestions.
4. The itinerary agent creates a practical travel plan.
5. The final agent formats the result into a polished response.

## Contributing

Contributions are welcome. If you want to improve the app, add new travel features, or fix issues:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a pull request

## Acknowledgments

This project is built with the help of modern LLM tooling and travel APIs, and it is intended as a practical example of combining LangGraph agents with real-world applications.
