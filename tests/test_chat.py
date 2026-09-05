"""Tests for the chat path and context preparation in src/summarize.py.

`answer_question` was previously only reached through /chat with itself mocked
out, so the history cleaning, the input cap and the Groq -> Ollama fallback were
untested.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import summarize  # noqa: E402


@pytest.fixture()
def prompts(monkeypatch):
    """Capture the prompt that reaches the model instead of calling one."""
    seen = {}

    def fake_groq(messages, model=None, max_retries=3, max_tokens=None):
        seen["messages"] = messages
        seen["max_tokens"] = max_tokens
        return "An answer."

    monkeypatch.setattr(summarize, "_ask_groq", fake_groq)
    monkeypatch.setattr(summarize, "_ask_ollama",
                        lambda p, json_mode=True: pytest.fail("Ollama should not be reached"))
    return seen


# ---------------------------------------------------------------------------
# History handling
# ---------------------------------------------------------------------------

def test_history_is_included_and_labelled_by_role(prompts):
    summarize.answer_question("PAPER", "Why?", [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ])
    prompt = prompts["messages"][-1]["content"]
    assert "User: First question" in prompt
    assert "Assistant: First answer" in prompt


def test_error_placeholders_are_stripped_from_history(prompts):
    """Older builds pushed "⚠️ Could not reach the server." into the transcript;
    feeding those back would have the model explain an outage it never saw."""
    summarize.answer_question("PAPER", "Why?", [
        {"role": "assistant", "content": "⚠️ Could not reach the server."},
        {"role": "user", "content": "A real question"},
    ])
    prompt = prompts["messages"][-1]["content"]
    assert "⚠️" not in prompt
    assert "A real question" in prompt


def test_empty_history_entries_are_dropped(prompts):
    summarize.answer_question("PAPER", "Why?", [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": ""},
    ])
    assert "Conversation so far" not in prompts["messages"][-1]["content"]


def test_only_the_recent_exchanges_are_kept(prompts):
    history = [{"role": "user", "content": f"question {i}"} for i in range(20)]
    summarize.answer_question("PAPER", "Why?", history)
    prompt = prompts["messages"][-1]["content"]
    assert "question 19" in prompt
    assert "question 0" not in prompt


def test_no_history_means_no_conversation_block(prompts):
    summarize.answer_question("PAPER", "Why?", None)
    assert "Conversation so far" not in prompts["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# Input bounds and model settings
# ---------------------------------------------------------------------------

def test_paper_text_is_capped_before_being_sent(prompts):
    """Keeps the chat request inside the per-minute token budget."""
    summarize.answer_question("x" * 100000, "Why?")
    prompt = prompts["messages"][-1]["content"]
    # measure the embedded paper, not incidental "x"s in the prompt boilerplate
    longest_run = max(len(run) for run in re.findall(r"x+", prompt))
    assert longest_run == 12000


def test_answers_are_bounded_by_max_tokens(prompts):
    summarize.answer_question("PAPER", "Why?")
    assert prompts["max_tokens"] == 700


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

def test_falls_back_to_ollama_when_groq_fails(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("quota exhausted")))
    monkeypatch.setattr(summarize, "_ask_ollama", lambda p, json_mode=True: "local answer")
    assert summarize.answer_question("PAPER", "Why?") == "local answer"


def test_both_backends_failing_surfaces_an_error(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq",
                        lambda *a, **k: (_ for _ in ()).throw(Exception("groq down")))
    monkeypatch.setattr(summarize, "_ask_ollama",
                        lambda p, json_mode=True: (_ for _ in ()).throw(Exception("Ollama is not running")))
    with pytest.raises(Exception, match="Ollama is not running"):
        summarize.answer_question("PAPER", "Why?")


# ---------------------------------------------------------------------------
# Answer cleaning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Answer: the result is 42.", "the result is 42."),
    ("ANSWER:  spaced out", "spaced out"),
    ("Response: something", "something"),
    ("  plain answer  ", "plain answer"),
    ("line\n\n\n\n\nline", "line\n\nline"),
    ("", ""),
    (None, ""),
])
def test_clean_answer_normalises_model_output(raw, expected):
    assert summarize._clean_answer(raw) == expected


def test_clean_answer_is_applied_to_the_returned_answer(monkeypatch):
    monkeypatch.setattr(summarize, "_ask_groq", lambda *a, **k: "Answer: forty two")
    assert summarize.answer_question("PAPER", "Why?") == "forty two"


# ---------------------------------------------------------------------------
# Context preparation for the summary call
# ---------------------------------------------------------------------------

def test_short_papers_pass_through_whole():
    text = "word " * 100
    assert summarize._prepare_section_context(text) == text


def test_long_papers_are_sampled_from_head_middle_and_tail():
    """Sampling rather than truncating is what keeps the conclusions in scope."""
    size = summarize.SECTION_CONTEXT_CHARS * 3
    text = "H" * (size // 3) + "M" * (size // 3) + "T" * (size // 3)
    out = summarize._prepare_section_context(text)

    assert "H" in out and "M" in out and "T" in out
    assert "[ ... middle section omitted ... ]" in out
    assert len(out) <= summarize.MAX_INPUT_CHARS


def test_sampled_context_never_exceeds_the_hard_cap():
    out = summarize._prepare_section_context("z" * 500000)
    assert len(out) <= summarize.MAX_INPUT_CHARS
