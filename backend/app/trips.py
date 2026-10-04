from app.db import get_pool

# Every query here filters by user_id, so a user can only see or change their own trips.


def make_title(message: str, max_length: int = 80) -> str:
    title = " ".join(message.split())
    return title if len(title) <= max_length else title[: max_length - 3] + "..."


async def user_owns_trip(user_id: int, thread_id: str) -> bool:
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM trips WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        return await cur.fetchone() is not None


async def create_trip(user_id: int, thread_id: str, first_message: str):
    async with get_pool().connection() as conn:
        await conn.execute(
            "INSERT INTO trips (thread_id, user_id, title) VALUES (%s, %s, %s)",
            (thread_id, user_id, make_title(first_message)),
        )


async def touch_trip(user_id: int, thread_id: str):
    async with get_pool().connection() as conn:
        await conn.execute(
            "UPDATE trips SET updated_at = now() WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )


async def list_trips(user_id: int) -> list[dict]:
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM trips
            WHERE user_id = %s
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        rows = await cur.fetchall()

    return [
        {
            "thread_id": row["thread_id"],
            "title": row["title"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        }
        for row in rows
    ]
