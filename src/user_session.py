import sqlite3
import uuid
from datetime import date
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel

app = FastAPI()

DB_PATH = "users.db"
DAILY_LIMIT = 5  # summaries per session per day


def init_db():
    conn = sqlite3.connect(DB_PATH)
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


init_db()


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


@app.post("/save-email")
def save_email(data: EmailInput, request: Request, response: Response):
    session_id = get_or_create_session_id(request, response)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO users (session_id, email) VALUES (?, ?)",
        (session_id, data.email)
    )
    conn.commit()
    conn.close()

    return {"status": "saved", "session_id": session_id}


@app.get("/get-email")
def get_email(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"email": None}

    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)

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


@app.get("/usage-status")
def usage_status(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return {"used": 0, "limit": DAILY_LIMIT}

    today = str(date.today())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT count FROM usage WHERE session_id = ? AND usage_date = ?",
        (session_id, today)
    )
    row = cursor.fetchone()
    conn.close()

    return {"used": row[0] if row else 0, "limit": DAILY_LIMIT}
