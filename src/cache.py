"""A small SQLite-backed TTL cache for external lookups.

Crossref and Unpaywall are hit on every DOI request. The results barely change
— a paper's authors and journal are fixed, and its open-access status moves
over months, not minutes — so re-fetching them on every summarize call just adds
latency and load on two free public APIs that ask callers to be considerate.

Summaries have their own cache in summarize.py, keyed by paper text. This one is
for the metadata lookups around them.
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


def default_db_path() -> str:
    """The SQLite file shared by the cache, sessions and usage counters.

    Env-overridable so a container can point it at a mounted volume; see the
    DB_PATH note in docker-compose.yml.
    """
    return os.getenv("DB_PATH") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "users.db"
    )


DB_PATH = default_db_path()

# Positive results are stable enough to keep for a month. Misses expire sooner:
# a paper that was paywalled or unindexed last week may not be today, and a
# transient outage should not be remembered as "this DOI has no PDF".
TTL_DAYS = 30
MISS_TTL_DAYS = 1


def _connect():
    path = DB_PATH
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lookup_cache (
            namespace TEXT,
            key TEXT,
            value_json TEXT,
            created_at TEXT,
            PRIMARY KEY (namespace, key)
        )
    """)
    return conn


_SENTINEL = object()


def get(namespace: str, key: str, default=_SENTINEL) -> Any:
    """Return the cached value, or `default` when absent or expired.

    A cached None is a real value (a negative result), which is why the miss
    signal is a separate sentinel rather than None itself.
    """
    if not key:
        return None if default is _SENTINEL else default
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT value_json, created_at FROM lookup_cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        finally:
            conn.close()
    except Exception:
        return None if default is _SENTINEL else default

    if not row:
        return None if default is _SENTINEL else default

    value_json, created_at = row
    try:
        value = json.loads(value_json)
    except Exception:
        return None if default is _SENTINEL else default

    ttl = timedelta(days=MISS_TTL_DAYS if value is None else TTL_DAYS)
    try:
        created = datetime.fromisoformat(created_at)
    except Exception:
        return None if default is _SENTINEL else default
    if datetime.now(timezone.utc) - created > ttl:
        return None if default is _SENTINEL else default
    return value


def put(namespace: str, key: str, value: Any) -> None:
    """Store a value. Failures are swallowed — a cache must never break a request."""
    if not key:
        return
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO lookup_cache (namespace, key, value_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (namespace, key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def clear(namespace: str = None) -> None:
    """Drop cached entries — the whole cache, or one namespace."""
    try:
        conn = _connect()
        try:
            if namespace:
                conn.execute("DELETE FROM lookup_cache WHERE namespace = ?", (namespace,))
            else:
                conn.execute("DELETE FROM lookup_cache")
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
