"""Tests for the /summarize/stream SSE endpoint.
Run from the project root:  ./venv/bin/python -m pytest tests/ -v
"""
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import api  # noqa: E402
import user_session  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(user_session, "DB_PATH", str(tmp_path / "test.db"))
    yield TestClient(api.app)


FAKE_PAPER = {
    "source": "pdf", "title": "Test Paper", "abstract": "An abstract.",
    "authors": ["Jane Doe"], "year": "2025", "journal": "J of Tests", "cited_by": 3,
    "full_text": "word " * 100,
}


def _parse_events(text):
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def test_stream_requires_input(client):
    res = client.post("/summarize/stream", data={})
    assert res.status_code == 400
    assert "Provide either a PDF file or a DOI" in res.json()["detail"]


def test_stream_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(api, "process_input", lambda pdf_path=None, doi=None, email=None: {
        "source": "error", "title": "", "abstract": "", "full_text": "", "error": "bad pdf",
    })
    res = client.post("/summarize/stream", files={"file": ("paper.pdf", b"junk", "application/pdf")})
    assert res.status_code == 422
    assert "bad pdf" in res.json()["detail"]


def test_stream_success(client, monkeypatch):
    monkeypatch.setattr(api, "process_input", lambda pdf_path=None, doi=None, email=None: dict(FAKE_PAPER))

    def fake_stream(text, title="", abstract=""):
        yield {"type": "section_done", "section": "summary", "content": "A summary."}
        yield {"type": "section_done", "section": "findings", "content": "Some findings."}
        yield {"type": "done", "result": {
            "summary": "A summary.", "methodology": "A method.",
            "research_gaps": "Some gaps.", "findings": "Some findings.",
            "future_work": "Future work.", "full_text": text,
        }}

    monkeypatch.setattr(api, "stream_summarize_paper", fake_stream)

    res = client.post("/summarize/stream", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _parse_events(res.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert events[0]["meta"]["title"] == "Test Paper"
    assert types.count("section_done") == 2
    assert types[-1] == "done"
    assert events[-1]["result"]["summary"] == "A summary."
    assert events[-1]["result"]["full_text"] == FAKE_PAPER["full_text"]


def test_stream_section_failure_surfaces_error(client, monkeypatch):
    monkeypatch.setattr(api, "process_input", lambda pdf_path=None, doi=None, email=None: dict(FAKE_PAPER))

    def failing_stream(text, title="", abstract=""):
        yield {"type": "error", "detail": "Groq error: rate limit exceeded"}

    monkeypatch.setattr(api, "stream_summarize_paper", failing_stream)

    res = client.post("/summarize/stream", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 200  # stream itself starts fine
    events = _parse_events(res.text)
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "error"
    assert "rate limit" in events[-1]["detail"]

