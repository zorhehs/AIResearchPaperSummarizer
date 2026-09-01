"""Tests for summarize section strategies, prompt context, and the
findings digit-check retry. Run from the project root:
./venv/bin/python -m pytest tests/ -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import summarize  # noqa: E402


def test_key_stats_section_registered():
    assert "key_stats" in summarize.SECTION_STRATEGIES
    assert "key_stats" in summarize._SECTION_ORDER
    # every ordered section has a strategy, and vice versa
    assert set(summarize._SECTION_ORDER) == set(summarize.SECTION_STRATEGIES.keys())


def test_paper_context_block():
    block = summarize._paper_context_block("My Title", "My abstract.")
    assert "Paper title: My Title" in block
    assert "Paper abstract: My abstract." in block
    assert summarize._paper_context_block("", "") == ""
    assert summarize._paper_context_block("T only") .startswith("Paper title: T only")


def test_paper_context_block_truncates_abstract():
    long_abstract = "x" * 5000
    block = summarize._paper_context_block("", long_abstract)
    assert "x" * 1500 in block
    assert "x" * 1501 not in block


def test_findings_retry_when_no_numbers(monkeypatch):
    calls = []

    def fake_groq(messages):
        calls.append(messages)
        if len(calls) == 1:
            return "The method works well in practice."  # no digits
        return "Accuracy: 94.2% (+3.1 over the baseline)."  # has digits

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    result = summarize._ask_section("findings", "paper text")
    assert "94.2%" in result
    assert len(calls) == 2  # retried once for missing numbers


def test_findings_kept_when_numbers_present(monkeypatch):
    calls = []

    def fake_groq(messages):
        calls.append(messages)
        return "Accuracy improved by 12% across 3 datasets."

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    result = summarize._ask_section("findings", "paper text")
    assert "12%" in result
    assert len(calls) == 1  # no retry needed


def test_non_findings_section_not_digit_checked(monkeypatch):
    calls = []

    def fake_groq(messages):
        calls.append(messages)
        return "A qualitative methodology description without numbers."

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    result = summarize._ask_section("methodology", "paper text")
    assert "qualitative" in result
    assert len(calls) == 1  # digit check only applies to findings


def test_section_prompt_includes_paper_context(monkeypatch):
    captured = {}

    def fake_groq(messages):
        captured["messages"] = messages
        return "Accuracy: 99.9%."

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    summarize._ask_section("key_stats", "paper text", title="My Paper", abstract="An abstract.")
    user_content = captured["messages"][1]["content"]
    assert "Paper title: My Paper" in user_content
    assert "Paper abstract: An abstract." in user_content
    assert "key quantitative results" in user_content  # key_stats instruction


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
