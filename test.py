import asyncio

from tools.tavily_tool import tavily_search
from tools.google_flights_tool import search_google_flights
from backend import run_travel_agent, open_travel_graph
from db import open_db, close_db

# res = tavily_search("Best hotels in India")
# print(res)


# res = search_google_flights("MEL", "NRT", "2026-11-10", "2026-11-13")
# print(res)


async def main():
    user_input = input("Enter travel request: ")

    await open_db()
    try:
        await open_travel_graph()
        response = await run_travel_agent(
            user_input=user_input,
            thread_id="test_user"
        )
    finally:
        await close_db()

    print("\nFINAL RESPONSE:\n")
    print(response["answer"])


asyncio.run(main())
