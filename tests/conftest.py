"""Shared test setup.

The lookup cache is SQLite-backed and shared across requests by design, which
means it is also shared across tests unless isolated. Without this, one test's
cached Crossref/Unpaywall result silently answers another test's lookup — and
the failure shows up somewhere unrelated.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cache  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_lookup_cache(tmp_path, monkeypatch):
    """Give every test its own empty lookup cache."""
    monkeypatch.setattr(cache, "DB_PATH", str(tmp_path / "lookup-cache.db"))
    yield
