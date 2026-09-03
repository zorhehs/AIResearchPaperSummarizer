"""API tests. Run from the project root:  ./venv/bin/python -m pytest tests/ -v"""
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import api  # noqa: E402
import user_session  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # isolate the usage/db layer to a temp database per test
    monkeypatch.setattr(user_session, "DB_PATH", str(tmp_path / "test.db"))
    yield TestClient(api.app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_served_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AI Research Paper Summarizer" in res.text


def test_summarize_requires_input(client):
    res = client.post("/summarize", data={})
    assert res.status_code == 400
    assert "Provide either a PDF file or a DOI" in res.json()["detail"]


def test_summarize_success(client, monkeypatch):
    monkeypatch.setattr(api, "process_input", lambda pdf_path=None, doi=None, email=None: {
        "source": "pdf", "title": "Test Paper", "abstract": "An abstract.",
        "authors": ["Jane Doe"], "year": "2025", "journal": "J of Tests", "cited_by": 3,
        "full_text": "word " * 100,
    })
    monkeypatch.setattr(api, "summarize_paper", lambda text, session_id=None, title="", abstract="", source="": {
        "title": "Model Title", "authors": ["Model Author"],
        "one_line_summary": "One line.", "field_tags": ["ML"],
        "overview": "An overview.", "problem_statement": "A problem.",
        "approach": "An approach.", "key_findings": [{"finding": "F", "detail": "D"}],
        "results_table": [], "significance": "Significant.",
        "limitations": [], "future_work": [], "key_terms": [],
        "confidence_notes": "", "full_text": text,
    })
    res = client.post("/summarize", files={"file": ("paper.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert res.status_code == 200
    data = res.json()
    assert data["title"] == "Test Paper"  # pipeline-extracted title wins
    assert data["overview"] == "An overview."
    assert data["authors"] == ["Jane Doe"]  # Crossref-verified authors win
    assert data["cited_by"] == 3


def test_summarize_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(api, "process_input", lambda pdf_path=None, doi=None, email=None: {
        "source": "error", "title": "", "abstract": "", "full_text": "", "error": "bad pdf",
    })
    res = client.post("/summarize", files={"file": ("paper.pdf", b"junk", "application/pdf")})
    assert res.status_code == 422
    assert "bad pdf" in res.json()["detail"]


def test_chat_requires_paper(client):
    res = client.post("/chat", json={"question": "hi"})
    assert res.status_code == 400
    assert "No paper loaded" in res.json()["detail"]


def test_chat_requires_question(client):
    res = client.post("/chat", json={"paper_text": "some text", "question": "  "})
    assert res.status_code == 400


def test_chat_success(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda text, q, history: "Because of self-attention.")
    res = client.post("/chat", json={"paper_text": "text", "question": "why?", "chat_history": []})
    assert res.status_code == 200
    assert res.json()["answer"] == "Because of self-attention."


def test_ground_citations_maps_quotes_to_pages():
    from api import _ground_citations
    payload = {"key_findings": [
        {"finding": "A", "quote": "The quick brown fox jumps over the lazy dog"},
        {"finding": "B", "quote": "This quote appears nowhere in the document"},
        {"finding": "C", "quote": ""},
    ]}
    text = "First page filler words here.\nSecond page: the quick brown fox jumps over the lazy dog."
    spans = [[0, 30], [30, len(text)]]
    _ground_citations(payload, text, spans)
    assert payload["key_findings"][0]["citation"] == {"verified": True, "page": 2}
    assert payload["key_findings"][1]["citation"]["verified"] is False
    assert payload["key_findings"][2]["citation"]["verified"] is False
    assert payload["grounded_citations"] == 1


def test_daily_usage_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(user_session, "DB_PATH", str(tmp_path / "usage.db"))
    sid = str(uuid.uuid4())
    for i in range(user_session.DAILY_LIMIT):
        user_session.check_and_increment_usage(sid)  # should not raise
    with pytest.raises(user_session.HTTPException) as exc:
        user_session.check_and_increment_usage(sid)
    assert exc.value.status_code == 429
