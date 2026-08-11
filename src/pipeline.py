import os
import requests

from extract import extract_text, PDFExtractionError
from clean import clean_text
from metadata import extract_metadata
from fetch_doi import get_pdf_url_from_doi


def get_metadata_from_doi_crossref(doi):
    url = "https://api.crossref.org/works/" + doi
    response = requests.get(url)
    if response.status_code != 200:
        return {"title": "", "abstract": ""}

    data = response.json().get("message", {})
    title = data.get("title", [""])
    title = title[0] if title else ""
    abstract = data.get("abstract", "") or ""
    return {"title": title, "abstract": abstract}


def download_pdf(pdf_url, save_path="temp_downloaded.pdf"):
    response = requests.get(pdf_url, timeout=15)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def process_input(pdf_path=None, doi=None, email="zorhehs@gmail.com"):
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
        return {
            "source": "pdf",
            "title": meta["title"],
            "abstract": meta["abstract"],
            "full_text": cleaned,
        }

    elif doi:
        pdf_url = get_pdf_url_from_doi(doi, email)

        if pdf_url:
            try:
                local_path = download_pdf(pdf_url)
                raw = extract_text(local_path)
                cleaned = clean_text(raw)
                meta = extract_metadata(cleaned)
                os.remove(local_path)
                return {
                    "source": "doi_pdf",
                    "title": meta["title"],
                    "abstract": meta["abstract"],
                    "full_text": cleaned,
                }
            except Exception as e:
                print("PDF download/parsing failed:", e)

        meta = get_metadata_from_doi_crossref(doi)
        return {
            "source": "doi_metadata_only",
            "title": meta["title"],
            "abstract": meta["abstract"],
            "full_text": meta["abstract"],
        }

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
