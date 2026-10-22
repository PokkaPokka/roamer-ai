"""
Eval 1: which search finds the right guide excerpt most often?

Golden sets: excerpts sampled from different cities; for each, an LLM wrote a query
the excerpt answers, reviewed by hand. The LLM only drafts test queries; the scores
come from the search alone (only the bge-m3 embedder is involved).

- questions_a: 30 natural questions, searched across all 21K excerpts. The first run
               showed hybrid losing to meaning-only, which led to the all-words change.
- questions_b: 30 fresh natural questions written after that, to check the change on
               queries it wasn't designed around.
- topics:      30 short topic searches with a city filter, the way the app's guide
               agent searches ("street food markets", city "Tokyo").

A search "hits" when it returns that exact excerpt. Modes:
- hybrid:           meaning + keyword (any word), fused with RRF (the original design)
- hybrid_all_words: the keyword half only counts excerpts with every query word
- meaning / keyword: each search alone

Run from the backend folder:
    python -m evals.retrieval make --set questions_b     # sample excerpts, draft queries
    python -m evals.retrieval run --set questions_a
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from app.db import close_db, get_pool, open_db
from app.knowledge_base import search_knowledge_base

DATA = Path(__file__).parent / "data"
SECTIONS = ("See", "Do", "Eat", "Drink", "Buy", "Sleep", "Get around")
QUERIES_PER_SET = 30
TOP_K = 10

QUESTION_PROMPT = """Here is an excerpt from the Wikivoyage travel guide for {city}.

{excerpt}

Write one question a traveller might type into a travel planner that this excerpt
answers specifically (a named place, a price, a practical tip). Mention {city}.
Paraphrase: don't copy unusual phrases from the excerpt. Reply with the question only."""

TOPIC_PROMPT = """Here is an excerpt from the Wikivoyage travel guide for {city}.

{excerpt}

Write a short search phrase (3 to 6 words) that a travel planning assistant would use to
find the information in this excerpt, like "cheap street food stalls" or "ferry to the
islands". Don't include the city name. Reply with the phrase only."""

SETS = {
    # set name: (random seed, prompt, search with the excerpt's city as a filter?)
    "questions_a": (7, QUESTION_PROMPT, False),
    "questions_b": (11, QUESTION_PROMPT, False),
    "topics": (23, TOPIC_PROMPT, True),
}

MODES = {
    "hybrid": {"mode": "hybrid", "require_all_words": False},
    "hybrid_all_words": {"mode": "hybrid", "require_all_words": True},
    "meaning": {"mode": "meaning"},
    "keyword": {"mode": "keyword", "require_all_words": False},
}


def golden_path(name: str) -> Path:
    return DATA / f"retrieval_{name}.json"


async def make(name: str):
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6)
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), reasoning=False, temperature=0)

    seed, prompt, _ = SETS[name]
    # Excerpts used by the other sets are left out, so the sets don't overlap.
    used = {item["chunk_id"] for other in SETS if other != name and golden_path(other).exists()
            for item in json.loads(golden_path(other).read_text())}

    async with get_pool().connection() as conn:
        cur = await conn.execute(
            """
            SELECT k.id, c.title AS city, k.section, k.content
            FROM kb_chunks k JOIN kb_cities c ON c.id = k.city_id
            WHERE k.section = ANY(%s) AND length(k.content) > 800
            """,
            (list(SECTIONS),),
        )
        chunks = [chunk for chunk in await cur.fetchall() if chunk["id"] not in used]

    # One excerpt per city, so each set covers 30 different guides.
    random.Random(seed).shuffle(chunks)
    picked, cities = [], set()
    for chunk in chunks:
        if chunk["city"] not in cities:
            cities.add(chunk["city"])
            picked.append(chunk)
        if len(picked) == QUERIES_PER_SET:
            break

    golden = []
    for chunk in picked:
        excerpt = chunk["content"].split("\n", 1)[-1]
        reply = await llm.ainvoke(prompt.format(city=chunk["city"], excerpt=excerpt))
        golden.append({
            "chunk_id": chunk["id"],
            "city": chunk["city"],
            "section": chunk["section"],
            "question": reply.content.strip().strip('"'),
            "excerpt_start": excerpt[:160],
        })
        print(f"{chunk['city']} / {chunk['section']}: {golden[-1]['question']}")

    golden_path(name).write_text(json.dumps(golden, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(golden)} queries to {golden_path(name)}. Review them before running.")


def scores(ranks: list[int | None]) -> dict:
    """hit@1, hit@5 and MRR@10 from the gold excerpt's rank in each result list (None = not found)."""
    n = len(ranks)
    return {
        "hit@1": sum(rank == 1 for rank in ranks) / n,
        "hit@5": sum(rank is not None and rank <= 5 for rank in ranks) / n,
        "mrr@10": sum(1 / rank for rank in ranks if rank) / n,
    }


async def run(name: str):
    golden = json.loads(golden_path(name).read_text())
    _, _, use_city = SETS[name]
    ranks = {mode: [] for mode in MODES}
    rows = []

    for item in golden:
        row = {"question": item["question"]}
        for mode, options in MODES.items():
            results = await search_knowledge_base(
                item["question"], city=item["city"] if use_city else None, limit=TOP_K, **options
            )
            ids = [result["chunk_id"] for result in results]
            rank = ids.index(item["chunk_id"]) + 1 if item["chunk_id"] in ids else None
            ranks[mode].append(rank)
            row[mode] = rank
        rows.append(row)
        print("  ".join(f"{mode} {row[mode]!s:>4}" for mode in MODES) + f"  | {item['question'][:60]}", flush=True)

    summary = {mode: scores(mode_ranks) for mode, mode_ranks in ranks.items()}
    where = "with the city filter" if use_city else "across all 21K excerpts"
    print(f"\n{name}: {len(golden)} queries, searching {where}:")
    for mode, result in summary.items():
        print(f"  {mode:17} hit@1 {result['hit@1']:.0%}  hit@5 {result['hit@5']:.0%}  MRR@10 {result['mrr@10']:.2f}")

    (DATA / f"retrieval_results_{name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["make", "run"])
    parser.add_argument("--set", choices=list(SETS), default="questions_a")
    args = parser.parse_args()

    os.environ.setdefault("LANGSMITH_TRACING", "false")
    await open_db()
    try:
        await (make(args.set) if args.stage == "make" else run(args.set))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
