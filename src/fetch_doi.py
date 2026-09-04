import requests
from typing import Optional


def get_pdf_url_from_doi(doi: str, email: str = "zorhehs@gmail.com") -> Optional[str]:
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
        result = get_pdf_url_from_doi(doi)
        print(f"{doi} -> {result}")
