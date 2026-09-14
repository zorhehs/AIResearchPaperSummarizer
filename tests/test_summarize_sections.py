"""Tests for the structured PaperSummary generation engine.
Run from the project root:  ./venv/bin/python -m pytest tests/ -v
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import summarize  # noqa: E402

VALID_SUMMARY = {
    "title": "Test Paper",
    "authors": ["Jane Doe"],
    "institutions": ["Test University"],
    "publication_info": "arXiv:2401.00001",
    "one_line_summary": "A one line summary of the paper.",
    "field_tags": ["Machine Learning"],
    "overview": "The paper studies things.",
    "problem_statement": "A gap exists.",
    "approach": "They use a novel method.",
    "key_findings": [{"finding": "It works", "detail": "94.2% accuracy"}],
    "results_table": [{"metric": "Accuracy", "value": "94.2%", "comparison": "+3.1"}],
    "significance": "Important for the field.",
    "limitations": ["Small dataset"],
    "future_work": ["Scale it up"],
    "key_terms": [{"term": "Transformer", "definition": "A neural architecture."}],
    "confidence_notes": "",
}


def test_schema_rejects_missing_required_fields():
    with pytest.raises(summarize.ValidationError):
        summarize.PaperSummary(title="t")


def test_generate_paper_summary_valid_json(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq", lambda messages, **kw: json.dumps(VALID_SUMMARY))
    result = summarize._generate_paper_summary("paper text")
    assert result["title"] == "Test Paper"
    assert result["key_findings"][0]["detail"] == "94.2% accuracy"


def test_generate_paper_summary_tolerates_code_fences(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq",
                        lambda messages, **kw: "```json\n" + json.dumps(VALID_SUMMARY) + "\n```")
    result = summarize._generate_paper_summary("paper text")
    assert result["overview"] == "The paper studies things."


def test_generate_paper_summary_retries_invalid_json(monkeypatch):
    calls = []

    def fake_groq(messages, **kw):
        calls.append((messages, kw))
        if len(calls) == 1:
            return "I cannot do that."  # unparseable → not valid JSON
        return json.dumps(VALID_SUMMARY)

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    result = summarize._generate_paper_summary("paper text")
    assert result["significance"] == "Important for the field."
    assert len(calls) == 2  # one retry

    first, second = calls
    # The retry re-sends the same prompt — it no longer appends the previous
    # answer, which the input cap used to truncate into a useless tail anyway.
    assert second[0] == first[0]
    # ...and it must not wait on a full window: a request that already cost
    # one call should fall back rather than stall for the minute to roll.
    assert first[1]["block"] is True
    assert second[1]["block"] is False


def test_summary_call_uses_json_mode_and_low_reasoning(monkeypatch):
    """These two settings are what keep the summary to one call: JSON mode
    removes the malformed-output retry, low effort trims thinking tokens."""
    seen = {}
    monkeypatch.setattr(summarize, "_ask_groq",
                        lambda messages, **kw: (seen.update(kw), json.dumps(VALID_SUMMARY))[1])
    summarize._generate_paper_summary("paper text")
    assert seen["json_mode"] is True
    assert seen["reasoning_effort"] == "low"


def test_a_retry_with_no_budget_falls_back_instead_of_waiting(monkeypatch):
    calls = []

    def fake_groq(messages, **kw):
        calls.append(kw)
        if len(calls) == 1:
            return "not json"
        raise summarize.BudgetUnavailable("no room")

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    monkeypatch.setattr(summarize, "_ask_ollama",
                        lambda prompt, json_mode=False: json.dumps(VALID_SUMMARY))
    result = summarize._generate_paper_summary("paper text")
    assert result["title"] == "Test Paper"  # served by the local fallback
    assert len(calls) == 2


def test_generate_paper_summary_ollama_fallback(monkeypatch):
    def failing_groq(messages, **kw):
        raise Exception("Groq error: quota exhausted")

    monkeypatch.setattr(summarize, "_ask_groq", failing_groq)
    monkeypatch.setattr(summarize, "_ask_ollama",
                        lambda prompt, json_mode=False: json.dumps(VALID_SUMMARY))
    result = summarize._generate_paper_summary("paper text")
    assert result["problem_statement"] == "A gap exists."


def test_generate_paper_summary_raises_when_all_providers_fail(monkeypatch):
    def down(*a, **k):
        raise Exception("provider down")

    monkeypatch.setattr(summarize, "_ask_groq", down)
    monkeypatch.setattr(summarize, "_ask_ollama", down)
    with pytest.raises(Exception, match="Summary generation failed"):
        summarize._generate_paper_summary("paper text")


def test_parse_json_object_extracts_embedded_json():
    raw = ('Here you go:\n{"title": "T", "one_line_summary": "x", "overview": "o", '
           '"problem_statement": "p", "approach": "a", "significance": "s"}\nHope that helps!')
    data = summarize._parse_json_object(raw)
    assert data["title"] == "T"


def test_cache_roundtrip_and_versioning(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize, "DB_PATH", str(tmp_path / "cache.db"))
    text = "unique paper text " * 10
    payload = {"summary": "S", "key_stats": "- Accuracy: 91.2%", "findings": "- 12% gain"}

    assert summarize._cache_get(text) is None
    summarize._cache_put(text, payload)
    cached = summarize._cache_get(text)
    assert cached == payload

    # bumping the cache version invalidates old entries
    monkeypatch.setattr(summarize, "CACHE_VERSION", "v999")
    assert summarize._cache_get(text) is None
