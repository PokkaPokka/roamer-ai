import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg.errors import UniqueViolation
from pwdlib import PasswordHash

from app.db import get_pool

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET is missing. Add a long random string to .env")

JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME = timedelta(days=7)

# Argon2, the current recommendation for storing passwords.
password_hash = PasswordHash.recommended()

# Checked against when the email doesn't exist, so a wrong email takes as long
# as a wrong password and doesn't reveal which accounts exist.
DUMMY_HASH = password_hash.hash("not-a-real-password")

bearer_scheme = HTTPBearer(auto_error=False)


# =========================
# Tokens
# =========================

def create_access_token(user_id: int, lifetime: timedelta = TOKEN_LIFETIME) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + lifetime,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        return None


# =========================
# Users
# =========================

def normalize_email(email: str) -> str:
    return email.strip().lower()


async def create_user(email: str, password: str) -> dict | None:
    """Returns the new user, or None if the email is already registered."""
    try:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email",
                (normalize_email(email), password_hash.hash(password)),
            )
            return await cur.fetchone()
    except UniqueViolation:
        return None


async def authenticate_user(email: str, password: str) -> dict | None:
    """Returns the user if the email and password match, otherwise None."""
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s",
            (normalize_email(email),),
        )
        user = await cur.fetchone()

    if user is None:
        password_hash.verify(password, DUMMY_HASH)
        return None

    if not password_hash.verify(password, user["password_hash"]):
        return None

    return {"id": user["id"], "email": user["email"]}


# =========================
# FastAPI dependency
# =========================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """Add `user: dict = Depends(get_current_user)` to a route to require login."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not logged in or session expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise unauthorized

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise unauthorized

    async with get_pool().connection() as conn:
        cur = await conn.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
        user = await cur.fetchone()

    # The token is valid but the account was deleted.
    if user is None:
        raise unauthorized

    return user
