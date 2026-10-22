"""
Hybrid search over the Wikivoyage knowledge base (kb_chunks).

Two searches run inside one SQL query:
- meaning search: pgvector cosine distance between the question's embedding and each chunk
- keyword search: Postgres full-text search on the `tsv` column
Their result lists are merged with Reciprocal Rank Fusion (RRF):
    score = sum over both lists of 1 / (RRF_K + rank)
so only each chunk's position matters, not the two searches' different score scales.
"""

from langchain_ollama import OllamaEmbeddings

from app.db import get_pool

EMBEDDING_MODEL = "bge-m3"
RRF_K = 60
# How many candidates each search contributes before fusion.
CANDIDATES_PER_SEARCH = 40
# The keyword search only counts chunks containing every word of the query. With
# "any word", long questions matched thousands of chunks on common words, and fusing
# that noise pushed good meaning results down (eval: MRR 0.58-0.62 vs 0.77 with this
# on, 60 questions). Exact names like "Bebek Bengil" still match, so keyword search
# still helps where it should. See docs/METRICS.md.
KEYWORD_NEEDS_ALL_WORDS = True

_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


def to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


HYBRID_SEARCH_SQL = """
WITH keyword_query AS (
    -- `all_words` (AND) finds chunks with every word, such as the exact name
    -- "Bebek Bengil". `any_words` (OR) is only used when require_all_words is off.
    SELECT websearch_to_tsquery('english', %(question)s) AS all_words,
           replace(websearch_to_tsquery('english', %(question)s)::text, ' & ', ' | ')::tsquery AS any_words
),
city_filter AS (
    SELECT id FROM kb_cities
    WHERE %(city)s::text IS NULL
       OR lower(title) = lower(%(city)s)
       OR title ILIKE %(city)s || ' (%%'      -- e.g. "Córdoba (city, Argentina)"
),
meaning AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(embedding)s::vector) AS rank
    FROM (
        SELECT id, embedding FROM kb_chunks
        WHERE city_id IN (SELECT id FROM city_filter)
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(candidates)s
    ) nearest
),
keyword AS (
    SELECT id, row_number() OVER (ORDER BY has_all_words DESC, keyword_score DESC) AS rank
    FROM (
        SELECT id,
               tsv @@ all_words AS has_all_words,
               ts_rank_cd(tsv, any_words) AS keyword_score
        FROM kb_chunks, keyword_query
        WHERE city_id IN (SELECT id FROM city_filter) AND tsv @@ any_words
          -- Optionally keep only chunks with every word (see KEYWORD_NEEDS_ALL_WORDS).
          AND (NOT %(require_all_words)s OR tsv @@ all_words)
        ORDER BY has_all_words DESC, keyword_score DESC
        LIMIT %(candidates)s
    ) matches
),
fused AS (
    SELECT id,
           -- A weight of 0 turns one search off (used by the retrieval eval).
           sum(CASE WHEN source = 'meaning' THEN %(meaning_weight)s ELSE %(keyword_weight)s END
               / (%(rrf_k)s + rank)) AS score,
           bool_or(source = 'meaning') AS from_meaning,
           bool_or(source = 'keyword') AS from_keyword
    FROM (
        SELECT id, rank, 'meaning' AS source FROM meaning
        UNION ALL
        SELECT id, rank, 'keyword' AS source FROM keyword
    ) both_lists
    GROUP BY id
)
SELECT k.id AS chunk_id, c.title AS city, c.country, k.section, k.content, k.source_url,
       f.score, f.from_meaning, f.from_keyword
FROM fused f
JOIN kb_chunks k ON k.id = f.id
JOIN kb_cities c ON c.id = k.city_id
WHERE f.score > 0
-- On a tie, prefer chunks both searches agree on, then exact keyword matches.
ORDER BY f.score DESC, (f.from_meaning AND f.from_keyword) DESC, f.from_keyword DESC
LIMIT %(limit)s
"""


async def search_knowledge_base(
    question: str,
    city: str | None = None,
    limit: int = 5,
    mode: str = "hybrid",
    require_all_words: bool | None = None,
) -> list[dict]:
    """
    Returns the `limit` best chunks for the question, optionally only from one city.
    Each result has chunk_id, city, country, section, content, source_url, score, and
    from_meaning / from_keyword showing which search found it.
    mode "meaning" or "keyword" runs one search alone (for the retrieval eval).
    """
    if require_all_words is None:
        require_all_words = KEYWORD_NEEDS_ALL_WORDS
    embedding = await _embeddings.aembed_query(question)

    async with get_pool().connection() as conn:
        async with conn.transaction():
            # pgvector 0.8: keep scanning the HNSW index until enough rows pass
            # the city filter, instead of filtering a fixed top-40 down to nothing.
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            cur = await conn.execute(HYBRID_SEARCH_SQL, {
                "question": question,
                "embedding": to_vector_literal(embedding),
                "city": city,
                "candidates": CANDIDATES_PER_SEARCH,
                "rrf_k": RRF_K,
                "limit": limit,
                "require_all_words": require_all_words,
                "meaning_weight": 0.0 if mode == "keyword" else 1.0,
                "keyword_weight": 0.0 if mode == "meaning" else 1.0,
            })
            return await cur.fetchall()


async def city_in_knowledge_base(city: str) -> bool:
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM kb_cities WHERE lower(title) = lower(%s) OR title ILIKE %s || ' (%%' LIMIT 1",
            (city, city),
        )
        return await cur.fetchone() is not None
