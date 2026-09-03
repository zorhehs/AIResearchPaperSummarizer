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
    monkeypatch.setattr(summarize, "_ask_groq", lambda messages: json.dumps(VALID_SUMMARY))
    result = summarize._generate_paper_summary("paper text")
    assert result["title"] == "Test Paper"
    assert result["key_findings"][0]["detail"] == "94.2% accuracy"


def test_generate_paper_summary_tolerates_code_fences(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq",
                        lambda messages: "```json\n" + json.dumps(VALID_SUMMARY) + "\n```")
    result = summarize._generate_paper_summary("paper text")
    assert result["overview"] == "The paper studies things."


def test_generate_paper_summary_retries_invalid_json(monkeypatch):
    calls = []

    def fake_groq(messages):
        calls.append(messages)
        if len(calls) == 1:
            return "I cannot do that."  # unparseable → not valid JSON
        return json.dumps(VALID_SUMMARY)

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    result = summarize._generate_paper_summary("paper text")
    assert result["significance"] == "Important for the field."
    assert len(calls) == 2  # one corrective retry
    assert "Return ONLY the JSON object" in calls[1][-1]["content"]


def test_generate_paper_summary_ollama_fallback(monkeypatch):
    def failing_groq(messages):
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
