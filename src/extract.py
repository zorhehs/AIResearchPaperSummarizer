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


if __name__ == "__main__":
    text = extract_text("tests/sample_papers/paper1.pdf")
    print(text[:2000])
    print(f"\n\nTotal length: {len(text)} characters")
