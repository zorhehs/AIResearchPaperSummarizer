"""Tests for the Groq client layer in src/summarize.py.

Every other test in the suite monkeypatches `_ask_groq` away, so the model
rotation, the 429 handling and the per-minute token budget — the code that
actually decides whether a summary succeeds under free-tier pressure — had no
coverage at all. This file exercises them directly against a fake Groq SDK.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import summarize  # noqa: E402


# ---------------------------------------------------------------------------
# A fake Groq SDK: each model gets a scripted list of outcomes.
# ---------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, script, calls):
        self._script = script
        self._calls = calls

    def create(self, model, messages, temperature, **kwargs):
        self._calls.append({"model": model, "messages": messages, **kwargs})
        outcomes = self._script.get(model, ["ok"])
        outcome = outcomes.pop(0) if outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        message = types.SimpleNamespace(content=outcome)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def fake_groq(monkeypatch, script):
    """Install a fake `groq.Groq` and return the list of calls it receives."""
    calls = []

    class FakeGroq:
        def __init__(self, api_key=None):
            self.chat = types.SimpleNamespace(completions=_FakeCompletions(script, calls))

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    # Never actually sleep or block on the token window in tests.
    monkeypatch.setattr(summarize.time, "sleep", lambda s: None)
    monkeypatch.setattr(summarize, "_reserve_budget", lambda tokens: None)
    return calls


MSG = [{"role": "user", "content": "hello"}]

DAILY_QUOTA = Exception(
    "Error code: 429 - rate_limit_exceeded: Limit 100000, Used 100000, "
    "tokens per day (TPD) exceeded. Please try again in 19m39.792s."
)
MINUTE_LIMIT = Exception(
    "Error code: 429 - rate_limit_exceeded: tokens per minute (TPM). "
    "Please try again in 8.0s."
)
LONG_WAIT = Exception("Error code: 429 - rate_limit_exceeded. Please try again in 4m10.0s.")
# A daily exhaustion reported with a short retry window — happens near the reset
# boundary. This is the only shape that isolates the "tokens per day" branch:
# with the usual multi-minute window, the >20s check would rotate anyway.
DAILY_QUOTA_SHORT_WAIT = Exception(
    "Error code: 429 - rate_limit_exceeded: tokens per day (TPD) exceeded. "
    "Please try again in 6.0s."
)
NOT_FOUND = Exception("Error code: 404 - model_not_found: the model does not exist")
BAD_REQUEST = Exception("Error code: 400 - invalid_request_error: something else")


# ---------------------------------------------------------------------------
# Model rotation
# ---------------------------------------------------------------------------

def test_uses_the_primary_model_when_it_works(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["result"]})
    assert summarize._ask_groq(MSG) == "result"
    assert [c["model"] for c in calls] == [summarize.GROQ_MODEL]


def test_daily_quota_moves_straight_to_the_next_model(monkeypatch):
    """A per-day exhaustion cannot be waited out inside a request, so the whole
    point is to give up on that model immediately rather than sleep."""
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [DAILY_QUOTA],
        summarize.GROQ_FALLBACK_MODEL: ["from the fallback"],
    })
    slept = []
    monkeypatch.setattr(summarize.time, "sleep", lambda s: slept.append(s))

    assert summarize._ask_groq(MSG) == "from the fallback"
    assert [c["model"] for c in calls] == [summarize.GROQ_MODEL, summarize.GROQ_FALLBACK_MODEL]
    assert slept == [], "must not wait out a daily quota"


def test_daily_quota_rotates_even_when_the_retry_window_is_short(monkeypatch):
    """Isolates the "tokens per day" branch from the >20s branch that usually
    shadows it. A day's budget does not come back in six seconds, so waiting is
    always wrong here however short the window claims to be."""
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [DAILY_QUOTA_SHORT_WAIT, "should never be reached"],
        summarize.GROQ_FALLBACK_MODEL: ["from the fallback"],
    })
    slept = []
    monkeypatch.setattr(summarize.time, "sleep", lambda s: slept.append(s))

    assert summarize._ask_groq(MSG) == "from the fallback"
    assert slept == [], "a daily quota must never be slept through"
    assert [c["model"] for c in calls].count(summarize.GROQ_MODEL) == 1


def test_rotation_walks_the_whole_model_list(monkeypatch):
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [DAILY_QUOTA],
        summarize.GROQ_FALLBACK_MODEL: [DAILY_QUOTA],
        summarize.GROQ_THIRD_MODEL: ["third model answer"],
    })
    assert summarize._ask_groq(MSG) == "third model answer"
    assert [c["model"] for c in calls] == summarize.GROQ_MODELS


def test_an_unknown_model_id_is_skipped_not_retried(monkeypatch):
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [NOT_FOUND],
        summarize.GROQ_FALLBACK_MODEL: ["ok"],
    })
    assert summarize._ask_groq(MSG) == "ok"
    # exactly one attempt at the bad model, not max_retries of them
    assert [c["model"] for c in calls].count(summarize.GROQ_MODEL) == 1


def test_a_requested_model_is_tried_first_then_the_rotation(monkeypatch):
    calls = fake_groq(monkeypatch, {
        "custom-model": [DAILY_QUOTA],
        summarize.GROQ_MODEL: ["rotated"],
    })
    assert summarize._ask_groq(MSG, model="custom-model") == "rotated"
    assert [c["model"] for c in calls][:2] == ["custom-model", summarize.GROQ_MODEL]


def test_non_rate_limit_errors_move_on_without_retrying(monkeypatch):
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [BAD_REQUEST],
        summarize.GROQ_FALLBACK_MODEL: ["ok"],
    })
    assert summarize._ask_groq(MSG) == "ok"
    assert [c["model"] for c in calls].count(summarize.GROQ_MODEL) == 1


def test_every_model_failing_raises_with_the_last_error(monkeypatch):
    fake_groq(monkeypatch, {m: [DAILY_QUOTA] for m in summarize.GROQ_MODELS})
    with pytest.raises(Exception) as excinfo:
        summarize._ask_groq(MSG)
    assert "Groq error" in str(excinfo.value)
    assert "tokens per day" in str(excinfo.value)


def test_missing_api_key_fails_before_any_network_call(monkeypatch):
    calls = fake_groq(monkeypatch, {})
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(Exception, match="GROQ_API_KEY not set"):
        summarize._ask_groq(MSG)
    assert calls == []


# ---------------------------------------------------------------------------
# Rate-limit waiting
# ---------------------------------------------------------------------------

def test_a_short_minute_limit_is_waited_out_and_retried(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: [MINUTE_LIMIT, "second try"]})
    slept = []
    monkeypatch.setattr(summarize.time, "sleep", lambda s: slept.append(s))

    assert summarize._ask_groq(MSG) == "second try"
    assert [c["model"] for c in calls] == [summarize.GROQ_MODEL, summarize.GROQ_MODEL]
    assert slept == [9.0], "8.0s from the message, plus the 1s safety margin"


def test_a_multi_minute_wait_switches_model_instead_of_blocking(monkeypatch):
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [LONG_WAIT],
        summarize.GROQ_FALLBACK_MODEL: ["ok"],
    })
    slept = []
    monkeypatch.setattr(summarize.time, "sleep", lambda s: slept.append(s))

    assert summarize._ask_groq(MSG) == "ok"
    assert slept == [], "a 4-minute window must not be slept through"
    assert [c["model"] for c in calls] == [summarize.GROQ_MODEL, summarize.GROQ_FALLBACK_MODEL]


def test_retries_are_bounded_by_max_retries(monkeypatch):
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [MINUTE_LIMIT] * 10,
        summarize.GROQ_FALLBACK_MODEL: ["ok"],
    })
    assert summarize._ask_groq(MSG, max_retries=2) == "ok"
    assert [c["model"] for c in calls].count(summarize.GROQ_MODEL) == 2


@pytest.mark.parametrize("message, expected", [
    ("Please try again in 19m39.792s.", 19 * 60 + 39.792 + 1.0),
    ("Please try again in 20.0s.", 21.0),
    ("Please try again in 1m0s.", 61.0),
    ("Please try again in 0.5s", 1.5),
    ("no timing information here", 10.0),
    ("", 10.0),
])
def test_rate_limit_wait_parses_groq_messages(message, expected):
    assert summarize._rate_limit_wait(message) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Input capping and the token budget
# ---------------------------------------------------------------------------

def test_input_is_capped_so_a_request_can_never_be_too_large(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["ok"]})
    huge = [
        {"role": "system", "content": "s" * 15000},
        {"role": "user", "content": "u" * 50000},
    ]
    summarize._ask_groq(huge)
    sent = sum(len(m["content"]) for m in calls[0]["messages"])
    assert sent <= summarize.MAX_INPUT_CHARS
    assert len(calls[0]["messages"]) == 2, "messages are truncated, never dropped"
    assert calls[0]["messages"][0]["role"] == "system"


def test_capping_preserves_earlier_messages_in_full(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["ok"]})
    summarize._ask_groq([
        {"role": "system", "content": "keep me"},
        {"role": "user", "content": "x" * 50000},
    ])
    assert calls[0]["messages"][0]["content"] == "keep me"


def test_max_tokens_is_forwarded_only_when_given(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["ok", "ok"]})
    summarize._ask_groq(MSG)
    assert "max_tokens" not in calls[0]
    summarize._ask_groq(MSG, max_tokens=700)
    assert calls[1]["max_tokens"] == 700


def test_budget_is_reserved_before_the_call(monkeypatch):
    reserved = []
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["ok"]})
    monkeypatch.setattr(summarize, "_reserve_budget", lambda t: reserved.append(t))
    summarize._ask_groq([{"role": "user", "content": "x" * 3800}])
    assert len(reserved) == 1
    # ~1000 input tokens at 3.8 chars/token, plus the output estimate
    assert reserved[0] == pytest.approx(1000 + summarize.OUTPUT_TOKEN_EST, rel=0.05)


def test_reserve_budget_lets_a_request_inside_the_window_straight_through(monkeypatch):
    monkeypatch.setattr(summarize, "_budget", {"minute": int(summarize.time.time() // 60), "tokens": 0})
    monkeypatch.setattr(summarize.time, "sleep", lambda s: pytest.fail("should not block"))
    summarize._reserve_budget(summarize.TPM_LIMIT - 1)
    assert summarize._budget["tokens"] == summarize.TPM_LIMIT - 1


def test_reserve_budget_blocks_until_the_minute_rolls_over(monkeypatch):
    """The window is what stops parallel calls piling up into a 413."""
    minute = int(summarize.time.time() // 60)
    monkeypatch.setattr(summarize, "_budget", {"minute": minute, "tokens": summarize.TPM_LIMIT})

    clock = {"now": minute * 60.0}
    monkeypatch.setattr(summarize.time, "time", lambda: clock["now"])

    slept = []
    def advance(seconds):
        slept.append(seconds)
        clock["now"] += 60  # the next poll lands in a fresh minute
    monkeypatch.setattr(summarize.time, "sleep", advance)

    summarize._reserve_budget(100)
    assert slept, "a full window must make the caller wait"
    assert summarize._budget["tokens"] == 100, "counter resets with the new minute"


def test_est_tokens_never_returns_zero():
    assert summarize._est_tokens(0) == 1
    assert summarize._est_tokens(3800) == 1000
