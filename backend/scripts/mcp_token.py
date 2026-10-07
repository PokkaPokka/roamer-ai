"""
Makes a long-lived login token for the Roamer MCP server, so Claude Desktop can
read your trips. Put the printed token in the MCP config as ROAMER_TOKEN.

Run from the backend folder:
    python -m scripts.mcp_token --email you@example.com
Treat the token like a password; it works until it expires.
"""

import argparse
import asyncio
from datetime import timedelta

from app.auth import create_access_token, normalize_email
from app.db import close_db, get_pool, open_db


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    await open_db()
    try:
        async with get_pool().connection() as conn:
            cur = await conn.execute("SELECT id FROM users WHERE email = %s", (normalize_email(args.email),))
            user = await cur.fetchone()
    finally:
        await close_db()

    if user is None:
        raise SystemExit(f"No account for {args.email}. Register in the web app first.")

    print(create_access_token(user["id"], lifetime=timedelta(days=args.days)))


if __name__ == "__main__":
    asyncio.run(main())
