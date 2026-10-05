"""
Build the Wikivoyage knowledge base for RAG, one stage at a time.

Run from the backend folder:
    python -m scripts.build_kb fetch    # download each city's guide (cached, safe to re-run)
    python -m scripts.build_kb chunk    # split the guides into chunks and print size estimates
    python -m scripts.build_kb embed    # embed the chunks with bge-m3 and load them into Postgres
    python -m scripts.build_kb embed --limit 5   # try the first 5 cities only

The city list comes from data/kb_cities.json (made by scripts.rank_cities).
Downloads and chunks go to data/cache/, which is not committed.
Wikivoyage text is CC BY-SA 4.0; every chunk keeps its source URL for attribution.
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import requests
from langchain_ollama import OllamaEmbeddings

from app.db import get_database_url

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CITIES_PATH = DATA_DIR / "kb_cities.json"
PAGES_DIR = DATA_DIR / "cache" / "pages"
CHUNKS_PATH = DATA_DIR / "cache" / "chunks.jsonl"

USER_AGENT = "RoamerAI/0.1 (student travel-planner project; builds a small Wikivoyage knowledge base)"
WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"

# Chunk size is measured in words; one English word is about 1.33 tokens,
# so 350 words is roughly 470 tokens. Overlap repeats the end of one chunk
# at the start of the next, so a fact split across the boundary isn't lost.
MAX_WORDS = 350
OVERLAP_WORDS = 40
MIN_WORDS = 30
TOKENS_PER_WORD = 1.33

EMBEDDING_MODEL = "bge-m3"
EMBED_BATCH_SIZE = 32

HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def load_cities() -> list[dict]:
    return json.loads(CITIES_PATH.read_text())["cities"]


def page_path(title: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", title)
    return PAGES_DIR / f"{safe}.json"


# =========================
# Stage 1: fetch
# =========================

def get_json(params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        response = session.get(WIKIVOYAGE_API, params=params, timeout=30)

        if response.status_code == 429 and attempt < retries - 1:
            wait = int(response.headers.get("Retry-After", 0)) or 5 * 2 ** attempt
            print(f"    rate limited, waiting {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        time.sleep(1)
        return response.json()


def fetch_page(title: str) -> dict:
    # Plain text with "== Heading ==" markers. Full-page extracts come one page per request.
    data = get_json({
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "titles": title,
        "redirects": 1,
        "prop": "extracts|revisions",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "rvprop": "ids",
    })
    page = data["query"]["pages"][0]

    return {
        "title": page["title"],
        "revision_id": page["revisions"][0]["revid"],
        "text": page.get("extract", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_all():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    cities = load_cities()

    for city in cities:
        path = page_path(city["title"])
        if path.exists():
            continue

        try:
            page = fetch_page(city["title"])
        except Exception as error:
            print(f"  #{city['rank']} {city['title']}: FAILED ({error})")
            continue

        path.write_text(json.dumps(page, ensure_ascii=False))
        print(f"  #{city['rank']} {city['title']}: {len(page['text'].split())} words")

    done = sum(page_path(city["title"]).exists() for city in cities)
    print(f"{done}/{len(cities)} guides downloaded to {PAGES_DIR}")


# =========================
# Stage 2: chunk
# =========================

def split_sections(text: str) -> list[tuple[str, list[str]]]:
    """
    Split a guide into its top-level sections (== See ==, == Eat ==, ...).
    Subsections (=== Budget ===) stay inside their section as a "Budget:" line,
    so a chunk never mixes two top-level sections.
    Returns [(section name, [paragraphs])].
    """
    sections = [("Overview", [])]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = HEADING.match(line)
        if match and len(match.group(1)) == 2:
            sections.append((match.group(2), []))
        elif match:
            sections[-1][1].append(f"{match.group(2)}:")
        else:
            sections[-1][1].append(line)

    return [(name, paragraphs) for name, paragraphs in sections if paragraphs]


def split_long_paragraph(paragraph: str) -> list[str]:
    words = paragraph.split()
    if len(words) <= MAX_WORDS:
        return [paragraph]
    return [" ".join(words[start:start + MAX_WORDS]) for start in range(0, len(words), MAX_WORDS)]


def chunk_section(paragraphs: list[str]) -> list[str]:
    """Pack whole paragraphs into chunks of up to MAX_WORDS, with OVERLAP_WORDS of overlap."""
    pieces = [piece for paragraph in paragraphs for piece in split_long_paragraph(paragraph)]
    chunks, current = [], []

    for piece in pieces:
        if current and len(" ".join(current + [piece]).split()) > MAX_WORDS:
            chunks.append(" ".join(current))
            overlap = " ".join(chunks[-1].split()[-OVERLAP_WORDS:])
            current = [overlap]
        current.append(piece)

    if current:
        chunks.append(" ".join(current))

    # A very short last chunk adds little on its own, so fold it into the previous one.
    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_WORDS:
        chunks[-2] = chunks[-2] + " " + chunks.pop()

    return chunks


def chunk_city(city: dict, page: dict) -> list[dict]:
    chunks = []

    for section, paragraphs in split_sections(page["text"]):
        for text in chunk_section(paragraphs):
            # The header gives each chunk context on its own, e.g. a "Budget:" list
            # of restaurants is clearly about eating in Ubud.
            header = f"{page['title']}, {city['country']} — {section}"
            chunks.append({
                "city": page["title"],
                "section": section,
                "chunk_index": len(chunks),
                "content": f"{header}\n{text}",
                "source_url": city["source_url"],
            })

    return chunks


def chunk_all():
    cities = load_cities()
    total, words_per_chunk, skipped = 0, [], []

    with CHUNKS_PATH.open("w") as output:
        for city in cities:
            path = page_path(city["title"])
            if not path.exists():
                skipped.append(city["title"])
                continue

            page = json.loads(path.read_text())
            for chunk in chunk_city(city, page):
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                words_per_chunk.append(len(chunk["content"].split()))
                total += 1

    content_bytes = CHUNKS_PATH.stat().st_size
    # Per row in Postgres: the text, a 1024-float vector (4 KB), the tsvector
    # (about half the text), the HNSW index entry (about 5 KB) and row overhead.
    estimated_bytes = content_bytes * 1.5 + total * (4_100 + 5_000 + 100)

    print(f"{total} chunks from {len(cities) - len(skipped)} cities -> {CHUNKS_PATH}")
    print(f"words per chunk: median {statistics.median(words_per_chunk):.0f}, "
          f"max {max(words_per_chunk)}, about {statistics.mean(words_per_chunk) * TOKENS_PER_WORD:.0f} tokens on average")
    print(f"estimated database size: {estimated_bytes / 1e6:.0f} MB")
    if skipped:
        print(f"not downloaded yet ({len(skipped)}): {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}")


# =========================
# Stage 3: embed and load
# =========================

def to_vector_literal(vector: list[float]) -> str:
    # pgvector accepts vectors as text like "[0.1,0.2,...]".
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def load_chunks_by_city() -> dict[str, list[dict]]:
    chunks_by_city = {}
    with CHUNKS_PATH.open() as file:
        for line in file:
            chunk = json.loads(line)
            chunks_by_city.setdefault(chunk["city"], []).append(chunk)
    return chunks_by_city


def loaded_revisions(conn) -> dict[str, int]:
    """Cities already in the database with chunks, and the guide revision they were built from."""
    rows = conn.execute(
        """
        SELECT c.title, c.revision_id
        FROM kb_cities c
        WHERE EXISTS (SELECT 1 FROM kb_chunks k WHERE k.city_id = c.id)
        """
    ).fetchall()
    return {title: revision_id for title, revision_id in rows}


def save_city(conn, city: dict, page: dict, chunks: list[dict], vectors: list[list[float]]):
    # One transaction per city: the old chunks are replaced all at once,
    # so a crash never leaves a city half loaded.
    with conn.transaction():
        city_id = conn.execute(
            """
            INSERT INTO kb_cities (title, country, page_views, revision_id, source_url, fetched_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (title) DO UPDATE SET
                country = EXCLUDED.country,
                page_views = EXCLUDED.page_views,
                revision_id = EXCLUDED.revision_id,
                source_url = EXCLUDED.source_url,
                fetched_at = EXCLUDED.fetched_at
            RETURNING id
            """,
            (page["title"], city["country"], city["page_views"], page["revision_id"],
             city["source_url"], page["fetched_at"]),
        ).fetchone()[0]

        conn.execute("DELETE FROM kb_chunks WHERE city_id = %s", (city_id,))

        # COPY sends all rows in one stream, much faster than one INSERT per chunk.
        with conn.cursor().copy(
            "COPY kb_chunks (city_id, section, chunk_index, content, source_url, embedding) FROM STDIN"
        ) as copy:
            for chunk, vector in zip(chunks, vectors):
                copy.write_row((city_id, chunk["section"], chunk["chunk_index"], chunk["content"],
                                chunk["source_url"], to_vector_literal(vector)))


def embed_all(limit: int | None):
    cities = load_cities()[:limit] if limit else load_cities()
    chunks_by_city = load_chunks_by_city()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    with psycopg.connect(get_database_url()) as conn:
        done = loaded_revisions(conn)
        started = time.time()
        loaded = 0

        for city in cities:
            path = page_path(city["title"])
            if not path.exists():
                print(f"  #{city['rank']} {city['title']}: not downloaded, skipped")
                continue

            page = json.loads(path.read_text())
            if done.get(page["title"]) == page["revision_id"]:
                continue

            chunks = chunks_by_city.get(page["title"], [])
            if not chunks:
                print(f"  #{city['rank']} {city['title']}: no chunks, run the chunk stage first")
                continue

            city_started = time.time()
            vectors = []
            for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[start:start + EMBED_BATCH_SIZE]
                vectors.extend(embeddings.embed_documents([chunk["content"] for chunk in batch]))

            save_city(conn, city, page, chunks, vectors)
            loaded += 1
            print(f"  #{city['rank']} {city['title']}: {len(chunks)} chunks in {time.time() - city_started:.0f}s")

        size = conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
        total = conn.execute("SELECT count(*) FROM kb_chunks").fetchone()[0]

    print(f"Loaded {loaded} cities in {(time.time() - started) / 60:.1f} min. "
          f"kb_chunks now has {total} rows; database size {size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fetch", "chunk", "embed"])
    parser.add_argument("--limit", type=int, help="embed stage: only the first N cities")
    args = parser.parse_args()

    if args.stage == "fetch":
        fetch_all()
    elif args.stage == "chunk":
        chunk_all()
    else:
        embed_all(args.limit)


if __name__ == "__main__":
    main()
