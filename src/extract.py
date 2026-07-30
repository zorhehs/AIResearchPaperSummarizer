import fitz  # PyMuPDF

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

if __name__ == "__main__":
    text = extract_text("tests/sample_papers/paper2.pdf")
    print(text[:2000])
    print(f"\n\nTotal length: {len(text)} characters")
