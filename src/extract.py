import re

import fitz  # PyMuPDF


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
