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

# Optional backstop for the cookie-only identity above. A client that simply
# discards cookies gets a fresh session id on every request, so the per-session
# limit alone bounds nothing. Setting IP_DAILY_LIMIT > 0 also caps summaries per
# client address per day.
#
# It is off by default on purpose: an address is a poor proxy for a person.
# University networks, offices and mobile carriers put thousands of people
# behind one NAT address, and limiting those to a shared allowance punishes
# ordinary users to inconvenience an attacker who can rent another address for
# pennies. Turn it on only where you know the traffic shape.
IP_DAILY_LIMIT = int(os.getenv("IP_DAILY_LIMIT", "0") or 0)

# Whether to believe X-Forwarded-For / X-Real-IP. Only enable this when the app
# genuinely sits behind a proxy you control: any client can send these headers,
# so trusting them on a directly-exposed app lets a caller forge a new address
# per request and makes the IP limit worse than useless.
TRUST_PROXY_HEADERS = (os.getenv("TRUST_PROXY_HEADERS", "") or "").strip().lower() in {"1", "true", "yes"}


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ip_usage (
            ip TEXT,
            usage_date TEXT,
            count INTEGER DEFAULT 0,
            PRIMARY KEY (ip, usage_date)
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


def _increment(table: str, key_column: str, key: str, limit: int, detail: str):
    """Reserve one unit against a daily counter, or raise 429 if it is spent."""
    today = str(date.today())
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT count FROM {table} WHERE {key_column} = ? AND usage_date = ?",
            (key, today)
        ).fetchone()
        current_count = row[0] if row else 0

        if current_count >= limit:
            raise HTTPException(status_code=429, detail=detail)

        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({key_column}, usage_date, count) VALUES (?, ?, ?)",
            (key, today, current_count + 1)
        )
        conn.commit()
    finally:
        conn.close()


def _decrement(table: str, key_column: str, key: str):
    """Hand back one unit reserved from a daily counter. Never goes negative."""
    today = str(date.today())
    conn = _connect()
    try:
        row = conn.execute(
            f"SELECT count FROM {table} WHERE {key_column} = ? AND usage_date = ?",
            (key, today)
        ).fetchone()
        if not row or row[0] <= 0:
            return
        conn.execute(
            f"UPDATE {table} SET count = ? WHERE {key_column} = ? AND usage_date = ?",
            (row[0] - 1, key, today)
        )
        conn.commit()
    finally:
        conn.close()


def check_and_increment_usage(session_id: str):
    """
    Call this right before running a summarization.
    Raises an error if the session has hit today's limit.
    """
    _increment(
        "usage", "session_id", session_id, DAILY_LIMIT,
        f"Daily limit of {DAILY_LIMIT} summaries reached. Try again tomorrow."
    )


def check_and_increment_ip_usage(ip: str):
    """Per-address backstop for the cookie-only session limit.

    A no-op unless IP_DAILY_LIMIT is configured — see the note on that setting
    for why it is opt-in.
    """
    if IP_DAILY_LIMIT <= 0 or not ip:
        return
    _increment(
        "ip_usage", "ip", ip, IP_DAILY_LIMIT,
        f"Daily limit of {IP_DAILY_LIMIT} summaries reached for this network. Try again tomorrow."
    )


def refund_ip_usage(ip: str):
    """Give back a per-address credit reserved for work that never happened."""
    if IP_DAILY_LIMIT <= 0 or not ip:
        return
    _decrement("ip_usage", "ip", ip)


def client_ip(request: Request) -> str:
    """Best-effort client address.

    Proxy headers are honoured only when TRUST_PROXY_HEADERS says the app is
    actually behind a proxy; otherwise the socket peer is the only value a
    caller cannot forge.
    """
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real = request.headers.get("x-real-ip", "")
        if real:
            return real.strip()
    return request.client.host if request.client else ""


def refund_usage(session_id: str):
    """Give back a summary credit that was reserved but never delivered.

    check_and_increment_usage() runs before the work starts, so a paper that
    fails to extract or a provider outage would otherwise cost the caller one
    of their daily summaries for nothing. Never drops below zero.
    """
    _decrement("usage", "session_id", session_id)


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
