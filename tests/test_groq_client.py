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
    # Never actually sleep or block on the token window in tests, and start
    # every test with empty per-model windows — _budgets is module state.
    monkeypatch.setattr(summarize.time, "sleep", lambda s: None)
    monkeypatch.setattr(summarize, "_budgets", {})
    monkeypatch.setattr(summarize, "_try_reserve", lambda model, tokens: True)
    monkeypatch.setattr(summarize, "_reserve_budget", lambda tokens, model=None: None)
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


def test_budget_is_reserved_against_the_model_that_is_called(monkeypatch):
    reserved = []
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["ok"]})
    monkeypatch.setattr(summarize, "_try_reserve",
                        lambda model, t: (reserved.append((model, t)), True)[1])
    summarize._ask_groq([{"role": "user", "content": "x" * 3800}])
    assert len(reserved) == 1
    assert reserved[0][0] == summarize.GROQ_MODEL
    # ~1000 input tokens at 3.8 chars/token, plus the output estimate
    assert reserved[0][1] == pytest.approx(1000 + summarize.OUTPUT_TOKEN_EST, rel=0.05)


def test_try_reserve_lets_a_request_inside_the_window_straight_through(monkeypatch):
    monkeypatch.setattr(summarize, "_budgets", {})
    assert summarize._try_reserve("m", summarize.TPM_LIMIT - 1) is True
    assert summarize._budgets["m"]["tokens"] == summarize.TPM_LIMIT - 1


def test_try_reserve_refuses_without_waiting_when_the_window_is_full(monkeypatch):
    monkeypatch.setattr(summarize, "_budgets", {})
    monkeypatch.setattr(summarize.time, "sleep", lambda s: pytest.fail("must never sleep"))
    assert summarize._try_reserve("m", summarize.TPM_LIMIT) is True
    assert summarize._try_reserve("m", 1) is False


def test_windows_are_tracked_per_model(monkeypatch):
    """Groq meters tokens per model, so filling one window must not touch another."""
    monkeypatch.setattr(summarize, "_budgets", {})
    assert summarize._try_reserve("a", summarize.TPM_LIMIT) is True
    assert summarize._try_reserve("a", 1) is False
    assert summarize._try_reserve("b", summarize.TPM_LIMIT) is True


def test_reserve_budget_blocks_until_the_minute_rolls_over(monkeypatch):
    """The window is what stops parallel calls piling up into a 413."""
    minute = int(summarize.time.time() // 60)
    monkeypatch.setattr(summarize, "_budgets", {"m": {"minute": minute, "tokens": summarize.TPM_LIMIT}})

    clock = {"now": minute * 60.0}
    monkeypatch.setattr(summarize.time, "time", lambda: clock["now"])

    slept = []
    def advance(seconds):
        slept.append(seconds)
        clock["now"] += 60  # the next poll lands in a fresh minute
    monkeypatch.setattr(summarize.time, "sleep", advance)

    summarize._reserve_budget(100, "m")
    assert slept, "a full window must make the caller wait"
    assert summarize._budgets["m"]["tokens"] == 100, "counter resets with the new minute"


# ---------------------------------------------------------------------------
# Rotation on a full window, JSON mode, reasoning effort
# ---------------------------------------------------------------------------

def test_a_full_window_rotates_to_the_next_model_instead_of_waiting(monkeypatch):
    """This is the 47-second stall: a retry that could not fit in gpt-oss-20b's
    window used to sleep until the minute rolled, while gpt-oss-120b sat idle
    with a full window of its own."""
    calls = fake_groq(monkeypatch, {summarize.GROQ_FALLBACK_MODEL: ["from 120b"]})
    monkeypatch.setattr(summarize, "_try_reserve",
                        lambda model, t: model != summarize.GROQ_MODEL)
    monkeypatch.setattr(summarize, "_reserve_budget",
                        lambda *a: pytest.fail("must rotate, not wait"))
    assert summarize._ask_groq(MSG) == "from 120b"
    assert [c["model"] for c in calls] == [summarize.GROQ_FALLBACK_MODEL]


def test_non_blocking_call_raises_when_no_model_has_room(monkeypatch):
    calls = fake_groq(monkeypatch, {})
    monkeypatch.setattr(summarize, "_try_reserve", lambda model, t: False)
    with pytest.raises(summarize.BudgetUnavailable):
        summarize._ask_groq(MSG, block=False)
    assert calls == [], "nothing may be sent when nothing fits"


def test_blocking_call_waits_only_after_every_model_with_room_has_failed(monkeypatch):
    """Waiting is the last resort, not the first: models with room are tried
    first, and only then does the call wait on a full window."""
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_FALLBACK_MODEL: [DAILY_QUOTA],
        summarize.GROQ_THIRD_MODEL: [DAILY_QUOTA],
        summarize.GROQ_MODEL: ["after the wait"],
    })
    monkeypatch.setattr(summarize, "_try_reserve",
                        lambda model, t: model != summarize.GROQ_MODEL)
    waited = []
    monkeypatch.setattr(summarize, "_reserve_budget",
                        lambda t, model=None: waited.append(model))
    assert summarize._ask_groq(MSG) == "after the wait"
    assert waited == [summarize.GROQ_MODEL]
    assert [c["model"] for c in calls] == [
        summarize.GROQ_FALLBACK_MODEL, summarize.GROQ_THIRD_MODEL, summarize.GROQ_MODEL]


def test_json_mode_is_requested_when_asked(monkeypatch):
    calls = fake_groq(monkeypatch, {summarize.GROQ_MODEL: ["{}", "{}"]})
    summarize._ask_groq(MSG)
    assert "response_format" not in calls[0]
    summarize._ask_groq(MSG, json_mode=True)
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_reasoning_effort_goes_only_to_models_that_accept_it(monkeypatch):
    """The qwen models take different values and reject "low"; sending it
    would burn the fallback with a 400."""
    calls = fake_groq(monkeypatch, {
        summarize.GROQ_MODEL: [DAILY_QUOTA],
        summarize.GROQ_FALLBACK_MODEL: [DAILY_QUOTA],
        summarize.GROQ_THIRD_MODEL: ["ok"],
    })
    summarize._ask_groq(MSG, reasoning_effort="low")
    by_model = {c["model"]: c for c in calls}
    assert by_model[summarize.GROQ_MODEL].get("reasoning_effort") == "low"
    assert by_model[summarize.GROQ_FALLBACK_MODEL].get("reasoning_effort") == "low"
    assert "reasoning_effort" not in by_model[summarize.GROQ_THIRD_MODEL]


def test_the_third_model_is_one_that_exists():
    """qwen3-32b was removed from Groq; an earlier fix had the ids backwards."""
    assert summarize.GROQ_THIRD_MODEL == "qwen/qwen3.8-27b"


def test_est_tokens_never_returns_zero():
    assert summarize._est_tokens(0) == 1
    assert summarize._est_tokens(3800) == 1000
