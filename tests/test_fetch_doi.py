"""Tests for the Unpaywall lookup in src/fetch_doi.py.

Every failure mode here has to return None rather than raise, because the
caller's fallback to Crossref metadata is what keeps a DOI usable when no
open-access PDF exists.
"""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import fetch_doi  # noqa: E402

DOI = "10.1371/journal.pone.0121283"
EMAIL = "someone@example.com"


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def stub_get(monkeypatch, response=None, raises=None):
    """Replace requests.get and record the URL it was called with."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if raises:
            raise raises
        return response

    monkeypatch.setattr(fetch_doi.requests, "get", fake_get)
    return calls


OA_PAYLOAD = {
    "is_oa": True,
    "best_oa_location": {"url_for_pdf": "https://example.org/paper.pdf"},
}


def test_returns_the_pdf_url_for_an_open_access_paper(monkeypatch):
    stub_get(monkeypatch, FakeResponse(200, OA_PAYLOAD))
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) == "https://example.org/paper.pdf"


def test_the_doi_is_url_encoded_and_the_email_is_sent(monkeypatch):
    calls = stub_get(monkeypatch, FakeResponse(200, OA_PAYLOAD))
    fetch_doi.get_pdf_url_from_doi("10.1234/a b/c", EMAIL)
    url = calls[0]["url"]
    assert "10.1234%2Fa%20b%2Fc" in url
    assert f"email={EMAIL}" in url


def test_a_missing_email_skips_the_lookup_entirely(monkeypatch):
    """Unpaywall requires a contact address, and the caller falls back to
    Crossref — so this must not fire a request that is guaranteed to fail."""
    calls = stub_get(monkeypatch, FakeResponse(200, OA_PAYLOAD))
    for empty in ("", "   ", None):
        assert fetch_doi.get_pdf_url_from_doi(DOI, empty) is None
    assert calls == []


def test_a_closed_access_paper_yields_no_url(monkeypatch):
    stub_get(monkeypatch, FakeResponse(200, {"is_oa": False,
                                             "best_oa_location": {"url_for_pdf": "x"}}))
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) is None


def test_open_access_with_no_location_yields_no_url(monkeypatch):
    stub_get(monkeypatch, FakeResponse(200, {"is_oa": True, "best_oa_location": None}))
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) is None


def test_a_location_without_a_pdf_yields_no_url(monkeypatch):
    """Some OA locations are landing pages with no direct PDF."""
    stub_get(monkeypatch, FakeResponse(200, {"is_oa": True,
                                             "best_oa_location": {"url": "https://example.org/html"}}))
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) is None


@pytest.mark.parametrize("status", [404, 422, 500, 503])
def test_error_statuses_yield_no_url(monkeypatch, status):
    stub_get(monkeypatch, FakeResponse(status, {}))
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) is None


@pytest.mark.parametrize("error", [
    requests.exceptions.Timeout("timed out"),
    requests.exceptions.ConnectionError("unreachable"),
])
def test_network_failures_return_none_rather_than_raising(monkeypatch, error):
    stub_get(monkeypatch, raises=error)
    assert fetch_doi.get_pdf_url_from_doi(DOI, EMAIL) is None


def test_the_request_is_bounded_by_a_timeout(monkeypatch):
    """An unbounded call here would hang a summarize request indefinitely."""
    calls = stub_get(monkeypatch, FakeResponse(200, OA_PAYLOAD))
    fetch_doi.get_pdf_url_from_doi(DOI, EMAIL)
    assert calls[0].get("timeout")


def test_a_user_agent_identifies_this_client(monkeypatch):
    calls = stub_get(monkeypatch, FakeResponse(200, OA_PAYLOAD))
    fetch_doi.get_pdf_url_from_doi(DOI, EMAIL)
    assert "AI-Research-Summarizer" in calls[0]["headers"]["User-Agent"]
