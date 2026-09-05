import os
import re

import fitz  # PyMuPDF

# Scanned papers carry no text layer, so they need OCR. MuPDF shells out to
# Tesseract, which has to be able to find its language data; TESSDATA_PREFIX is
# often unset even when Tesseract is installed, so look in the usual places.
_TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",           # Homebrew (Apple silicon)
    "/usr/local/share/tessdata",              # Homebrew (Intel) / source builds
    "/usr/share/tesseract-ocr/5/tessdata",    # Debian/Ubuntu, Tesseract 5
    "/usr/share/tesseract-ocr/4.00/tessdata", # Debian/Ubuntu, Tesseract 4
    "/usr/share/tessdata",
)

# OCR is slow relative to a text-layer read and runs per page, so cap the work.
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "20") or 20)
OCR_DPI = int(os.getenv("OCR_DPI", "200") or 200)


def _tessdata_dir() -> str:
    """Return a usable tessdata directory, or "" if none can be found."""
    configured = (os.getenv("TESSDATA_PREFIX") or "").strip()
    if configured and os.path.isdir(configured):
        return configured
    for candidate in _TESSDATA_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return ""


def ocr_available() -> bool:
    """Whether this install can actually OCR a page.

    Both halves matter: MuPDF has to be built with Tesseract support, and the
    language data has to be findable. Checked by attempting a real OCR call on
    a throwaway page, because a missing tessdata directory only surfaces at
    that point.
    """
    tessdata = _tessdata_dir()
    if not tessdata:
        return False
    try:
        doc = fitz.open()
        try:
            page = doc.new_page(width=50, height=50)
            page.get_textpage_ocr(dpi=72, full=True, tessdata=tessdata)
        finally:
            doc.close()
        return True
    except Exception:
        return False


def ocr_page_texts(pdf_path: str, max_pages: int = None) -> list:
    """OCR each page of a PDF that has no usable text layer.

    Returns one string per page, empty for any page OCR could not read, so the
    caller's page-span indexing still lines up with the physical pages.
    """
    tessdata = _tessdata_dir()
    if not tessdata:
        raise PDFExtractionError("OCR is unavailable: no Tesseract language data found.")

    limit = max_pages if max_pages is not None else OCR_MAX_PAGES
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Could not open PDF '{pdf_path}': {e}")

    texts = []
    try:
        for i, page in enumerate(doc):
            if i >= limit:
                texts.append("")
                continue
            try:
                textpage = page.get_textpage_ocr(dpi=OCR_DPI, full=True, tessdata=tessdata)
                texts.append(page.get_text(textpage=textpage))
            except Exception:
                texts.append("")  # a page that fails must not sink the document
    finally:
        doc.close()
    return texts



class PDFExtractionError(Exception):
    """Raised when a PDF can't be opened or contains no extractable text."""
    pass


def extract_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Could not open PDF '{pdf_path}': {e}")

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    stripped = full_text.strip()
    if len(stripped) < 50:
        raise PDFExtractionError(
            f"'{pdf_path}' appears to have no extractable text "
            f"(only {len(stripped)} characters found). "
            f"This may be a scanned PDF that needs OCR, which isn't supported yet."
        )

    return full_text


def extract_page_texts(pdf_path: str) -> str:
    """Return the text of each PDF page (one list entry per physical page).

    Raises PDFExtractionError when the file can't be opened.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFExtractionError(f"Could not open PDF '{pdf_path}': {e}")
    texts = [page.get_text() for page in doc]
    doc.close()
    return texts


def extract_pdf_metadata(pdf_path: str) -> dict:
    """Return the PDF's embedded document metadata ({'title', 'author'}).

    Most paper PDFs carry a proper title/author in their document info dict,
    which is far more reliable than guessing from extracted text. Placeholder
    values some PDF producers write (e.g. 'Untitled') are ignored.
    """
    try:
        doc = fitz.open(pdf_path)
        try:
            meta = doc.metadata or {}
        finally:
            doc.close()
    except Exception:
        return {"title": "", "author": ""}

    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()

    if title.lower() in {"untitled", "unknown", "document", "microsoft word - "}:
        title = ""
    if title and not re.search(r"[a-zA-Z]", title):
        title = ""

    return {"title": title, "author": author}


if __name__ == "__main__":
    text = extract_text("tests/sample_papers/paper1.pdf")
    print(text[:2000])
    print(f"\n\nTotal length: {len(text)} characters")
