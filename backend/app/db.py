import os

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

load_dotenv()


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


# App tables. The LangGraph checkpointer creates its own tables separately.
# `trips` records which user owns each graph thread. One statement per item,
# because the pool prepares every query and a prepared query can't hold several.
SCHEMA_SQL = [
    """
CREATE TABLE IF NOT EXISTS users (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
)
""",
    """
CREATE TABLE IF NOT EXISTS trips (
    thread_id  TEXT PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
""",
    "CREATE INDEX IF NOT EXISTS trips_user_updated_idx ON trips (user_id, updated_at DESC)",

    # Destination knowledge base (Wikivoyage), used for RAG.
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
CREATE TABLE IF NOT EXISTS kb_cities (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title       TEXT NOT NULL UNIQUE,
    country     TEXT,
    page_views  BIGINT NOT NULL DEFAULT 0,
    revision_id BIGINT,
    source_url  TEXT NOT NULL,
    fetched_at  TIMESTAMPTZ
)
""",
    # `embedding` is a bge-m3 vector (1024 dimensions) for semantic search.
    # `tsv` is filled in by Postgres from `content` for keyword search.
    """
CREATE TABLE IF NOT EXISTS kb_chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    city_id     BIGINT NOT NULL REFERENCES kb_cities(id) ON DELETE CASCADE,
    section     TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (city_id, chunk_index)
)
""",
    "CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx ON kb_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING gin (tsv)",
]


# One pool shared by the graph checkpointer and the app's own queries.
# Created by open_db() at startup, so importing this module never touches the database.
_pool: AsyncConnectionPool | None = None


async def open_db():
    global _pool

    _pool = AsyncConnectionPool(
        conninfo=get_database_url(),
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        # Render drops idle connections, so test each one before handing it out.
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    await _pool.open()

    async with _pool.connection() as conn:
        for statement in SCHEMA_SQL:
            await conn.execute(statement)


async def close_db():
    if _pool:
        await _pool.close()


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool is not open. Call open_db() first.")
    return _pool
