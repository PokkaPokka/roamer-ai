"""
Try the knowledge-base search from the terminal.

Run from the backend folder:
    python -m scripts.search_kb "cheap places to eat" --city Ubud
    python -m scripts.search_kb "Bebek Bengil"
"""

import argparse
import asyncio
import time

from app.db import open_db, close_db
from app.knowledge_base import search_knowledge_base


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--city")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    await open_db()
    try:
        started = time.time()
        results = await search_knowledge_base(args.question, city=args.city, limit=args.limit)
        elapsed = time.time() - started
    finally:
        await close_db()

    for number, result in enumerate(results, start=1):
        found_by = " + ".join(name for name, found in
                              [("meaning", result["from_meaning"]), ("keyword", result["from_keyword"])] if found)
        body = result["content"].split("\n", 1)[-1].replace("\n", " ")
        print(f"{number}. {result['city']} — {result['section']}  (score {result['score']:.4f}, {found_by})")
        print(f"   {body[:220]}...")
        print(f"   {result['source_url']}")

    print(f"\n{len(results)} results in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
