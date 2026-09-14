"""Deferred Crossref enrichment.

The title lookup takes 2-6 seconds cold and used to run before the model call
could start. It now runs alongside it and is merged at the end.
"""
import os
import sys
import threading
import time
from concurrent.futures import Future

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pipeline  # noqa: E402
import api  # noqa: E402
import user_session  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_papers", "paper1.pdf")
FOUND = {"authors": ["Ada Lovelace"], "year": "2024", "journal": "J. Tests", "cited_by": 9}


def test_deferred_lookup_does_not_block_process_input(monkeypatch):
    """The whole point: extraction must return in milliseconds even when the
    lookup is slow."""
    release = threading.Event()

    def slow_lookup(title):
        release.wait(5)
        return FOUND

    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search", slow_lookup)
    t0 = time.perf_counter()
    result = pipeline.process_input(pdf_path=SAMPLE, defer_enrichment=True)
    elapsed = time.perf_counter() - t0
    release.set()

    assert elapsed < 1.5, f"process_input waited on the lookup ({elapsed:.1f}s)"
    assert isinstance(result.get(pipeline.ENRICHMENT_KEY), Future)
    assert result["journal"] == ""  # not filled in yet


def test_apply_enrichment_merges_only_missing_fields(monkeypatch):
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search", lambda t: FOUND)
    result = pipeline.process_input(pdf_path=SAMPLE, defer_enrichment=True)
    result["year"] = "1999"  # something the PDF itself supplied

    pipeline.apply_enrichment(result)

    assert result["journal"] == "J. Tests"
    assert result["cited_by"] == 9
    assert result["year"] == "1999", "a value from the PDF must not be overwritten"
    assert pipeline.ENRICHMENT_KEY not in result, "the Future must not leak into the payload"


def test_apply_enrichment_gives_up_on_a_lookup_that_is_still_running(monkeypatch):
    """A slow Crossref must not hold the finished summary hostage."""
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search",
                        lambda t: time.sleep(3) or FOUND)
    result = pipeline.process_input(pdf_path=SAMPLE, defer_enrichment=True)

    t0 = time.perf_counter()
    pipeline.apply_enrichment(result, timeout=0.2)
    assert time.perf_counter() - t0 < 1.0
    assert result["journal"] == ""
    assert pipeline.ENRICHMENT_KEY not in result


def test_apply_enrichment_swallows_lookup_failures(monkeypatch):
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search",
                        lambda t: (_ for _ in ()).throw(RuntimeError("crossref down")))
    result = pipeline.process_input(pdf_path=SAMPLE, defer_enrichment=True)
    pipeline.apply_enrichment(result)  # must not raise
    assert result["journal"] == ""


def test_apply_enrichment_is_a_no_op_without_a_pending_lookup():
    meta = {"title": "x", "journal": ""}
    assert pipeline.apply_enrichment(dict(meta)) == meta


def test_inline_lookup_is_unchanged_by_default(monkeypatch):
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search", lambda t: FOUND)
    result = pipeline.process_input(pdf_path=SAMPLE)
    assert result["journal"] == "J. Tests"
    assert pipeline.ENRICHMENT_KEY not in result


# ---------------------------------------------------------------------------
# Through the API
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(user_session, "DB_PATH", str(tmp_path / "test.db"))
    yield TestClient(api.app)


SUMMARY = {
    "title": "T", "authors": [], "one_line_summary": "x", "field_tags": [],
    "overview": "o", "problem_statement": "p", "approach": "a",
    "key_findings": [], "results_table": [], "significance": "s",
    "limitations": [], "future_work": [], "key_terms": [], "confidence_notes": "",
}


def test_summarize_endpoint_carries_the_late_lookup_into_the_response(client, monkeypatch):
    """Metadata found while the model was working must reach the caller."""
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search", lambda t: FOUND)
    monkeypatch.setattr(api, "summarize_paper",
                        lambda text, **kw: {**SUMMARY, "full_text": text})
    with open(SAMPLE, "rb") as f:
        res = client.post("/summarize", files={"file": ("p.pdf", f, "application/pdf")})
    assert res.status_code == 200
    body = res.json()
    assert body["journal"] == "J. Tests"
    assert body["cited_by"] == 9
    assert pipeline.ENRICHMENT_KEY not in body


def test_stream_endpoint_carries_the_late_lookup_into_done(client, monkeypatch):
    import json
    monkeypatch.setattr(pipeline, "get_metadata_from_crossref_search", lambda t: FOUND)
    monkeypatch.setattr(api, "stream_summarize_paper",
                        lambda text, **kw: iter([{"type": "done", "result": {**SUMMARY, "full_text": text}}]))
    with open(SAMPLE, "rb") as f:
        res = client.post("/summarize/stream", files={"file": ("p.pdf", f, "application/pdf")})
    assert res.status_code == 200
    events = [json.loads(l[6:]) for l in res.text.splitlines() if l.startswith("data: ")]
    done = next(e for e in events if e["type"] == "done")
    assert done["result"]["journal"] == "J. Tests"
    meta = next(e for e in events if e["type"] == "meta")
    assert pipeline.ENRICHMENT_KEY not in meta["meta"], "a Future must never be serialised"
