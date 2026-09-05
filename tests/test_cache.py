"""Tests for the lookup cache and the Crossref/Unpaywall caching built on it."""
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cache  # noqa: E402
import fetch_doi  # noqa: E402
import pipeline  # noqa: E402

MISS = object()


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------

def test_roundtrips_a_value():
    cache.put("ns", "k", {"a": 1, "b": ["x"]})
    assert cache.get("ns", "k") == {"a": 1, "b": ["x"]}


def test_an_absent_key_reports_the_default():
    assert cache.get("ns", "nothing", MISS) is MISS


def test_namespaces_do_not_collide():
    cache.put("a", "same-key", "from a")
    cache.put("b", "same-key", "from b")
    assert cache.get("a", "same-key") == "from a"
    assert cache.get("b", "same-key") == "from b"


def test_a_cached_none_is_a_real_negative_result():
    """Distinguishing "we know there is nothing" from "we have not looked" is
    the whole point of caching misses."""
    cache.put("ns", "k", None)
    assert cache.get("ns", "k", MISS) is None


def test_an_empty_key_is_never_stored():
    cache.put("ns", "", "value")
    assert cache.get("ns", "", MISS) is MISS


def test_writing_again_replaces_the_entry():
    cache.put("ns", "k", "first")
    cache.put("ns", "k", "second")
    assert cache.get("ns", "k") == "second"


def _age_entry(namespace, key, days):
    import sqlite3
    conn = sqlite3.connect(cache.DB_PATH)
    conn.execute(
        "UPDATE lookup_cache SET created_at = ? WHERE namespace = ? AND key = ?",
        ((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), namespace, key),
    )
    conn.commit()
    conn.close()


def test_a_stale_hit_expires():
    cache.put("ns", "k", "value")
    _age_entry("ns", "k", cache.TTL_DAYS + 1)
    assert cache.get("ns", "k", MISS) is MISS


def test_a_fresh_hit_survives():
    cache.put("ns", "k", "value")
    _age_entry("ns", "k", cache.TTL_DAYS - 1)
    assert cache.get("ns", "k") == "value"


def test_negative_results_expire_sooner_than_positive_ones():
    """A paper that was paywalled last month may be open today, so a miss must
    not be remembered for as long as a hit."""
    assert cache.MISS_TTL_DAYS < cache.TTL_DAYS
    cache.put("ns", "missing", None)
    _age_entry("ns", "missing", cache.MISS_TTL_DAYS + 0.5)
    assert cache.get("ns", "missing", MISS) is MISS


def test_clear_empties_one_namespace_only():
    cache.put("keep", "k", 1)
    cache.put("drop", "k", 1)
    cache.clear("drop")
    assert cache.get("keep", "k") == 1
    assert cache.get("drop", "k", MISS) is MISS


def test_a_broken_cache_never_breaks_the_caller(monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", "/nonexistent-dir/nope/cache.db")
    cache.put("ns", "k", "v")                      # must not raise
    assert cache.get("ns", "k", MISS) is MISS


# ---------------------------------------------------------------------------
# Crossref / Unpaywall caching
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload


CROSSREF_WORK = {"message": {
    "title": ["A Paper"], "author": [{"given": "Ada", "family": "Lovelace"}],
    "issued": {"date-parts": [[2024]]}, "container-title": ["Journal"],
    "is-referenced-by-count": 7,
}}


def _count_calls(monkeypatch, module, response):
    calls = []
    monkeypatch.setattr(module.requests, "get",
                        lambda url, **kw: (calls.append(url), response)[1])
    return calls


def test_repeated_crossref_doi_lookups_hit_the_network_once(monkeypatch):
    calls = _count_calls(monkeypatch, pipeline, _Resp(CROSSREF_WORK))
    first = pipeline.get_metadata_from_doi_crossref("10.1234/abc")
    for _ in range(4):
        assert pipeline.get_metadata_from_doi_crossref("10.1234/abc") == first
    assert len(calls) == 1
    assert first["title"] == "A Paper"
    assert first["cited_by"] == 7


def test_different_dois_are_cached_separately(monkeypatch):
    calls = _count_calls(monkeypatch, pipeline, _Resp(CROSSREF_WORK))
    pipeline.get_metadata_from_doi_crossref("10.1234/aaa")
    pipeline.get_metadata_from_doi_crossref("10.1234/bbb")
    assert len(calls) == 2


def test_a_crossref_error_is_not_cached(monkeypatch):
    """A 5xx is Crossref having a bad minute, not a fact about the paper."""
    calls = _count_calls(monkeypatch, pipeline, _Resp({}, status_code=503))
    for _ in range(3):
        assert pipeline.get_metadata_from_doi_crossref("10.1234/abc")["title"] == ""
    assert len(calls) == 3


def test_crossref_title_search_caches_its_misses(monkeypatch):
    """This runs on every PDF upload, so a title that matches nothing must not
    re-query on each one."""
    calls = _count_calls(monkeypatch, pipeline, _Resp({"message": {"items": []}}))
    for _ in range(4):
        assert pipeline.get_metadata_from_crossref_search("Some Unmatched Title") is None
    assert len(calls) == 1


def test_repeated_unpaywall_lookups_hit_the_network_once(monkeypatch):
    payload = {"is_oa": True, "best_oa_location": {"url_for_pdf": "https://x/p.pdf"}}
    calls = _count_calls(monkeypatch, fetch_doi, _Resp(payload))
    for _ in range(4):
        assert fetch_doi.get_pdf_url_from_doi("10.1234/abc", "e@x.com") == "https://x/p.pdf"
    assert len(calls) == 1


def test_a_closed_access_result_is_cached_as_a_negative(monkeypatch):
    calls = _count_calls(monkeypatch, fetch_doi, _Resp({"is_oa": False}))
    for _ in range(4):
        assert fetch_doi.get_pdf_url_from_doi("10.1234/abc", "e@x.com") is None
    assert len(calls) == 1


def test_an_unpaywall_outage_is_not_cached(monkeypatch):
    calls = _count_calls(monkeypatch, fetch_doi, _Resp({}, status_code=503))
    for _ in range(3):
        assert fetch_doi.get_pdf_url_from_doi("10.1234/abc", "e@x.com") is None
    assert len(calls) == 3
