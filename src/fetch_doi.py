import requests
from typing import Optional

def get_pdf_url_from_doi(doi: str, email: str = "zorhehs@gmail.com") -> Optional[str]:
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    response = requests.get(url)

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
