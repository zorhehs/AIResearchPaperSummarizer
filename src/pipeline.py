import os
import requests
from dotenv import load_dotenv

load_dotenv()

from extract import extract_text, PDFExtractionError
from clean import clean_text
from metadata import extract_metadata
from fetch_doi import get_pdf_url_from_doi

UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "zorhehs@gmail.com")


def get_metadata_from_doi_crossref(doi):
    url = "https://api.crossref.org/works/" + doi
    response = requests.get(url)
    if response.status_code != 200:
        return {"title": "", "abstract": "", "authors": [], "year": "", "journal": "", "cited_by": None}

    data = response.json().get("message", {})
    title = data.get("title", [""])
    title = title[0] if title else ""
    abstract = data.get("abstract", "") or ""
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in data.get("author", [])[:12]
    ]
    year = ""
    for field in ("published-print", "published-online", "created"):
        parts = data.get(field, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    journal = (data.get("container-title") or [""])
    journal = journal[0] if journal else ""
    cited_by = data.get("is-referenced-by-count")
    return {"title": title, "abstract": abstract, "authors": authors, "year": year, "journal": journal, "cited_by": cited_by}


def download_pdf(pdf_url, save_path="temp_downloaded.pdf"):
    response = requests.get(pdf_url, timeout=15)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def process_input(pdf_path=None, doi=None, email=None):
    if pdf_path:
        try:
            raw = extract_text(pdf_path)
        except PDFExtractionError as e:
            return {
                "source": "error",
                "title": "",
                "abstract": "",
                "full_text": "",
                "error": str(e),
            }

        cleaned = clean_text(raw)
        meta = extract_metadata(cleaned)
        meta.update({"source": "pdf", "full_text": cleaned, "authors": [], "year": "", "journal": "", "cited_by": None})
        return meta

    elif doi:
        pdf_url = get_pdf_url_from_doi(doi, email or UNPAYWALL_EMAIL)

        if pdf_url:
            try:
                local_path = download_pdf(pdf_url)
                raw = extract_text(local_path)
                cleaned = clean_text(raw)
                meta = extract_metadata(cleaned)
                os.remove(local_path)
                meta.update({"source": "doi_pdf", "full_text": cleaned, "authors": [], "year": "", "journal": "", "cited_by": None})
                return meta
            except Exception as e:
                print("PDF download/parsing failed:", e)

        meta = get_metadata_from_doi_crossref(doi)
        meta.update({"source": "doi_metadata_only", "full_text": meta.get("abstract", "")})
        return meta

    else:
        raise ValueError("Provide either pdf_path or doi")


if __name__ == "__main__":
    import sys
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else "tests/sample_papers/paper1.pdf"

    print("===== TEST 1: Local PDF =====")
    result1 = process_input(pdf_path=test_pdf)
    if result1["source"] == "error":
        print("ERROR:", result1["error"])
    else:
        print("Source:", result1["source"])
        print("Title:", result1["title"])
        print("Text length:", len(result1["full_text"]))
