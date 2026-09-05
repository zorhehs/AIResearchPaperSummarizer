import os

import requests
from typing import Optional


def get_pdf_url_from_doi(doi: str, email: str) -> Optional[str]:
    """Resolve a DOI to an open-access PDF URL via Unpaywall.

    Unpaywall requires a contact address on every request; it identifies the
    caller, so it has to come from the deploying operator's configuration
    rather than being baked in here. Returns None when it is missing, which
    lets the caller fall back to Crossref metadata.
    """
    if not (email or "").strip():
        return None
    encoded_doi = requests.utils.quote(doi, safe="")
    url = f"https://api.unpaywall.org/v2/{encoded_doi}?email={email}"
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "AI-Research-Summarizer/1.0"})
    except requests.RequestException:
        # unreachable / timeout — let the caller fall back to Crossref
        return None

    if response.status_code != 200:
        return None

    data = response.json()

    if not data.get("is_oa"):
        return None

    best_location = data.get("best_oa_location")
    if not best_location:
        return None

    return best_location.get("url_for_pdf")


if __name__ == "__main__":
    test_dois = [
        "10.1371/journal.pone.0121283",
        "10.1016/j.cell.2015.05.001",
        "10.9999/totally.fake.doi.12345",
    ]

    for doi in test_dois:
        result = get_pdf_url_from_doi(doi, os.getenv("UNPAYWALL_EMAIL", ""))
        print(f"{doi} -> {result}")
