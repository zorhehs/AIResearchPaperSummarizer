import logging
import os
import re
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

from extract import (
    extract_page_texts,
    ocr_page_texts,
    ocr_available,
    PDFExtractionError,
    extract_pdf_metadata,
)
from clean import clean_text
from metadata import extract_metadata
from fetch_doi import get_pdf_url_from_doi
import cache

_MISS = object()

# Module logger rather than print(): under uvicorn these lines are otherwise
# unattributed stdout, with no level to filter on and no timestamp.
log = logging.getLogger(__name__)

# Unpaywall wants a contact address identifying whoever is calling it. There
# is deliberately no default: an unset value skips the Unpaywall lookup and
# falls through to Crossref, rather than sending every deployment's traffic
# under one person's address.
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "").strip()

# Scanned papers are OCR'd automatically when Tesseract is available. Set
# ENABLE_OCR=0 to keep the old behaviour (an honest error instead of a slow
# fallback) on deployments where the extra seconds per page are not welcome.
ENABLE_OCR = (os.getenv("ENABLE_OCR", "1") or "1").strip().lower() not in {"0", "false", "no"}

# Below this many characters a PDF is treated as having no usable text layer.
MIN_USABLE_TEXT = 50


def _parse_crossref_item(data: dict) -> dict:
    title = data.get("title", [""])
    title = title[0] if title else ""
    abstract = data.get("abstract", "") or ""
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in data.get("author", [])[:12]
    ]
    year = ""
    for field in ("published-print", "published-online", "issued", "created"):
        parts = data.get(field, {}).get("date-parts", [[None]])
        if parts and parts[0] and parts[0][0]:
            year = str(parts[0][0])
            break
    journal = (data.get("container-title") or [""])
    journal = journal[0] if journal else ""
    cited_by = data.get("is-referenced-by-count")
    return {"title": title, "abstract": abstract, "authors": authors, "year": year, "journal": journal, "cited_by": cited_by}


EMPTY_CROSSREF = {"title": "", "abstract": "", "authors": [], "year": "", "journal": "", "cited_by": None}


def get_metadata_from_doi_crossref(doi):
    # A published paper's authors, year and journal do not change; only the
    # citation count drifts, and not on a timescale worth a request per summary.
    cached = cache.get("crossref_doi", doi, _MISS)
    if cached is not _MISS:
        return cached

    encoded_doi = requests.utils.quote(doi, safe="")
    url = "https://api.crossref.org/works/" + encoded_doi
    response = requests.get(url, timeout=20, headers={"User-Agent": "AI-Research-Summarizer/1.0"})
    if response.status_code != 200:
        # Not cached: a 404 here may just be Crossref having a bad minute.
        return dict(EMPTY_CROSSREF)

    data = response.json().get("message", {})
    meta = _parse_crossref_item(data)
    cache.put("crossref_doi", doi, meta)
    return meta


def _title_similarity(a: str, b: str) -> float:
    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def get_metadata_from_crossref_search(title: str, rows: int = 3):
    """Look a paper up on Crossref by its title.

    Returns the best match's metadata only when the matched title closely
    resembles the given title (similarity >= 0.6), so we never attach the
    wrong paper's authors/journal. Returns None on any failure.
    """
    if not title:
        return None

    cached = cache.get("crossref_title", title, _MISS)
    if cached is not _MISS:
        return cached

    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": title,
                "rows": rows,
                "select": "title,author,issued,container-title,is-referenced-by-count",
            },
            timeout=15,
            headers={"User-Agent": "AI-Research-Summarizer/1.0"},
        )
        if response.status_code != 200:
            return None
        items = response.json().get("message", {}).get("items", [])
    except Exception:
        return None

    best, best_score = None, 0.0
    for item in items:
        candidate_title = (item.get("title") or [""])[0]
        score = _title_similarity(title, candidate_title)
        if score > best_score:
            best, best_score = item, score
    if best is None or best_score < 0.6:
        # Cache the miss too: a title that matches nothing today will still
        # match nothing tomorrow, and this runs on every PDF upload.
        cache.put("crossref_title", title, None)
        return None

    meta = _parse_crossref_item(best)
    meta.pop("abstract", None)  # the PDF has the real abstract
    cache.put("crossref_title", title, meta)
    return meta


def _enrich_from_crossref(meta: dict) -> dict:
    """Fill missing authors/year/journal/citations via a Crossref title search."""
    try:
        found = get_metadata_from_crossref_search(meta.get("title", ""))
    except Exception:
        found = None
    if not found:
        return meta
    for key in ("authors", "year", "journal", "cited_by"):
        if not meta.get(key) and found.get(key):
            meta[key] = found[key]
    return meta


def _finalize_pdf_meta(meta: dict, pdf_path: str) -> dict:
    """Complete PDF-derived metadata with embedded info and Crossref lookup."""
    embedded = extract_pdf_metadata(pdf_path)

    # Prefer the PDF's embedded title when the text heuristics found nothing
    if not meta.get("title") and embedded["title"]:
        meta["title"] = embedded["title"]

    authors = meta.get("authors") or []
    if not authors and embedded["author"]:
        authors = [
            a.strip(" .-*")
            for a in re.split(r"[,;]|\band\b|&", embedded["author"])
            if a.strip(" .-*")
        ]
    meta["authors"] = authors
    meta["year"] = meta.get("year") or ""
    meta["journal"] = meta.get("journal") or ""
    meta["cited_by"] = meta.get("cited_by")

    if meta.get("title"):
        meta = _enrich_from_crossref(meta)
    return meta


def download_pdf(pdf_url, save_path="temp_downloaded.pdf"):
    response = requests.get(pdf_url, timeout=15)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path


def _build_page_text(page_texts: list) -> tuple:
    """Clean each PDF page separately and join them, keeping a span index so
    any offset in the combined text can be mapped back to a page number.

    Returns (cleaned_full_text, page_spans) where page_spans[i] = [start, end)
    gives the character range of physical page i+1 inside cleaned_full_text.
    """
    cleaned_pages = [clean_text(p) for p in page_texts]
    cleaned = "\n\n".join(cleaned_pages)
    spans, offset = [], 0
    for cp in cleaned_pages:
        spans.append([offset, offset + len(cp)])
        offset += len(cp) + 2  # account for the "\n\n" join
    return cleaned, spans


def normalize_doi(raw: str) -> str:
    """Clean a user-supplied DOI: pull it out of URLs (doi.org/..., dx.doi.org/...),
    strip 'doi:' labels, whitespace and trailing punctuation."""
    s = (raw or "").strip()
    m = re.search(r"10\.\d{4,9}/\S+", s, flags=re.I)
    if m:
        s = m.group(0)
    else:
        s = re.sub(r"^doi\s*:\s*", "", s, flags=re.I).strip()
    return s.rstrip(".,;")


def process_input(pdf_path=None, doi=None, email=None):
    if pdf_path:
        try:
            page_texts = extract_page_texts(pdf_path)
        except PDFExtractionError as e:
            return {
                "source": "error",
                "title": "",
                "abstract": "",
                "full_text": "",
                "error": str(e),
            }

        cleaned, page_spans = _build_page_text(page_texts)
        source = "pdf"

        # No text layer — almost always a scan. Try OCR before giving up.
        if len(cleaned.strip()) < MIN_USABLE_TEXT and ENABLE_OCR and ocr_available():
            try:
                ocr_texts = ocr_page_texts(pdf_path)
                ocr_cleaned, ocr_spans = _build_page_text(ocr_texts)
                if len(ocr_cleaned.strip()) >= MIN_USABLE_TEXT:
                    cleaned, page_spans, source = ocr_cleaned, ocr_spans, "pdf_ocr"
            except Exception as e:
                log.warning("OCR failed for %s: %s", pdf_path, e)

        if len(cleaned.strip()) < MIN_USABLE_TEXT:
            if not ENABLE_OCR:
                why = "OCR is disabled on this server (ENABLE_OCR=0)."
            elif not ocr_available():
                why = ("This looks like a scanned PDF, and OCR is unavailable here "
                       "(Tesseract is not installed or its language data is missing).")
            else:
                why = "This looks like a scanned PDF, but OCR could not read any text from it."
            return {
                "source": "error",
                "title": "",
                "abstract": "",
                "full_text": "",
                "error": f"'{os.path.basename(pdf_path)}' has no extractable text. {why}",
            }

        meta = _finalize_pdf_meta(extract_metadata(cleaned), pdf_path)
        meta.update({"source": source, "full_text": cleaned, "page_spans": page_spans})
        return meta

    elif doi:
        doi = normalize_doi(doi)
        if not doi:
            return {
                "source": "error",
                "title": "", "abstract": "", "full_text": "",
                "error": "That doesn't look like a valid DOI. It should look like 10.1234/abcdef — or upload the PDF directly.",
            }

        contact_email = email or UNPAYWALL_EMAIL
        if not contact_email:
            log.info("UNPAYWALL_EMAIL is not set — skipping Unpaywall, using Crossref only.")
            pdf_url = None
        else:
            try:
                pdf_url = get_pdf_url_from_doi(doi, contact_email)
            except Exception as e:
                # Unpaywall unreachable → still try Crossref below instead of failing
                log.warning("Unpaywall lookup failed for %s: %s", doi, e)
                pdf_url = None

        if pdf_url:
            try:
                local_path = download_pdf(pdf_url)
                page_texts = extract_page_texts(local_path)
                cleaned, page_spans = _build_page_text(page_texts)
                meta = _finalize_pdf_meta(extract_metadata(cleaned), local_path)
                os.remove(local_path)
                meta.update({"source": "doi_pdf", "full_text": cleaned, "page_spans": page_spans})
                return meta
            except Exception as e:
                log.warning("PDF download/parsing failed for %s: %s", pdf_url, e)

        try:
            meta = get_metadata_from_doi_crossref(doi)
        except Exception as e:
            log.warning("Crossref lookup failed for %s: %s", doi, e)
            meta = {"title": "", "abstract": "", "authors": [], "year": "", "journal": "", "cited_by": None}
        abstract = (meta.get("abstract") or "").strip()
        if abstract:
            meta.update({
                "source": "doi_metadata_only",
                "full_text": abstract,
                "page_spans": [[0, len(abstract)]],
            })
            return meta

        return {
            "source": "error",
            "title": "",
            "abstract": "",
            "full_text": "",
            "error": "This DOI could not be resolved or does not provide usable article text. Please upload the PDF directly or try a different DOI.",
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
