import sqlite3
import uuid
from datetime import date
from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel

import os

router = APIRouter()

# Both the summary cache and the session/usage tables live in this SQLite file.
# DB_PATH is env-overridable so a container can point it at a mounted volume;
# without that the database would sit inside the image layer and be discarded
# every time the container is recreated.
DB_PATH = os.getenv("DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "users.db"
)
DAILY_LIMIT = 5  # summaries per session per day


def _ensure_parent_dir(path: str):
    """Create the directory holding the SQLite file if it is missing.

    DB_PATH can point at a mounted volume path that does not exist yet on a
    fresh host; sqlite3 will not create intermediate directories itself.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def init_db(db_path: str = None):
    path = db_path or DB_PATH
    _ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            session_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            session_id TEXT,
            usage_date TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (session_id, usage_date)
        )
    """)
    conn.commit()
    conn.close()


def _connect():
    """Open a connection, lazily ensuring the schema exists."""
    init_db()
    return sqlite3.connect(DB_PATH)


class EmailInput(BaseModel):
    email: str


def get_or_create_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
        )
    return session_id


@router.post("/save-email")
def save_email(data: EmailInput, request: Request, response: Response):
    session_id = get_or_create_session_id(request, response)

    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO users (session_id, email) VALUES (?, ?)",
        (session_id, data.email)
    )
    conn.commit()
    conn.close()

    return {"status": "saved", "session_id": session_id}


@router.get("/get-email")
def get_email(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"email": None}

    conn = _connect()
    cursor = conn.execute("SELECT email FROM users WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()

    return {"email": row[0] if row else None}


def check_and_increment_usage(session_id: str):
    """
    Call this right before running a summarization.
    Raises an error if the session has hit today's limit.
    """
    today = str(date.today())
    conn = _connect()

    cursor = conn.execute(
        "SELECT count FROM usage WHERE session_id = ? AND usage_date = ?",
        (session_id, today)
    )
    row = cursor.fetchone()
    current_count = row[0] if row else 0

    if current_count >= DAILY_LIMIT:
        conn.close()
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {DAILY_LIMIT} summaries reached. Try again tomorrow."
        )

    conn.execute(
        "INSERT OR REPLACE INTO usage (session_id, usage_date, count) VALUES (?, ?, ?)",
        (session_id, today, current_count + 1)
    )
    conn.commit()
    conn.close()


def refund_usage(session_id: str):
    """Give back a summary credit that was reserved but never delivered.

    check_and_increment_usage() runs before the work starts, so a paper that
    fails to extract or a provider outage would otherwise cost the caller one
    of their daily summaries for nothing. Never drops below zero.
    """
    today = str(date.today())
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT count FROM usage WHERE session_id = ? AND usage_date = ?",
            (session_id, today)
        ).fetchone()
        if not row or row[0] <= 0:
            return
        conn.execute(
            "UPDATE usage SET count = ? WHERE session_id = ? AND usage_date = ?",
            (row[0] - 1, session_id, today)
        )
        conn.commit()
    finally:
        conn.close()


@router.get("/usage-status")
def usage_status(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"used": 0, "limit": DAILY_LIMIT}

    today = str(date.today())
    conn = _connect()
    cursor = conn.execute(
        "SELECT count FROM usage WHERE session_id = ? AND usage_date = ?",
        (session_id, today)
    )
    row = cursor.fetchone()
    conn.close()

    return {"used": row[0] if row else 0, "limit": DAILY_LIMIT}
