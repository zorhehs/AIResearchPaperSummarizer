"""OCR fallback for scanned PDFs. Run from the project root:
    ./venv/bin/python -m pytest tests/ -v
"""
import os
import sys

import fitz
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import extract  # noqa: E402
import pipeline  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_papers", "paper1.pdf")

needs_ocr = pytest.mark.skipif(
    not extract.ocr_available(),
    reason="Tesseract or its language data is not installed on this machine",
)


@pytest.fixture(scope="module")
def scanned_pdf(tmp_path_factory):
    """A real paper rendered to images — i.e. a PDF with no text layer at all."""
    out_path = tmp_path_factory.mktemp("ocr") / "scanned.pdf"
    src = fitz.open(SAMPLE)
    out = fitz.open()
    for n in range(2):
        pix = src[n].get_pixmap(dpi=150)
        page = out.new_page(width=src[n].rect.width, height=src[n].rect.height)
        page.insert_image(page.rect, pixmap=pix)
    out.save(str(out_path))
    out.close()
    src.close()
    return str(out_path)


def test_fixture_really_has_no_text_layer(scanned_pdf):
    """Guard the fixture itself: if this PDF had text, the OCR tests below
    would pass without OCR ever running."""
    assert sum(len(t.strip()) for t in extract.extract_page_texts(scanned_pdf)) == 0


@needs_ocr
def test_ocr_reads_a_scanned_pdf(scanned_pdf):
    texts = extract.ocr_page_texts(scanned_pdf)
    assert len(texts) == 2
    assert all(len(t.strip()) > 100 for t in texts)


@needs_ocr
def test_ocr_respects_the_page_cap(scanned_pdf):
    texts = extract.ocr_page_texts(scanned_pdf, max_pages=1)
    assert len(texts) == 2          # one entry per physical page, always
    assert texts[0].strip()          # first page read
    assert texts[1] == ""            # second page skipped by the cap


@needs_ocr
def test_pipeline_falls_back_to_ocr(scanned_pdf):
    result = pipeline.process_input(pdf_path=scanned_pdf)
    assert result["source"] == "pdf_ocr"
    assert len(result["full_text"]) > 500
    assert len(result["page_spans"]) == 2
    assert result["title"]


@needs_ocr
def test_page_spans_survive_ocr(scanned_pdf):
    """Citation grounding maps an offset back to a page through these spans, so
    they have to keep tiling the text correctly after an OCR read."""
    result = pipeline.process_input(pdf_path=scanned_pdf)
    text, spans = result["full_text"], result["page_spans"]
    assert spans[0][0] == 0
    for (_, end), (nxt_start, _) in zip(spans, spans[1:]):
        assert nxt_start == end + 2  # the "\n\n" join between pages
    assert spans[-1][1] <= len(text)


def test_scanned_pdf_error_is_honest_when_ocr_is_off(scanned_pdf, monkeypatch):
    monkeypatch.setattr(pipeline, "ENABLE_OCR", False)
    result = pipeline.process_input(pdf_path=scanned_pdf)
    assert result["source"] == "error"
    assert "ENABLE_OCR=0" in result["error"]
    assert "scanned.pdf" in result["error"]


def test_scanned_pdf_error_when_ocr_unavailable(scanned_pdf, monkeypatch):
    monkeypatch.setattr(pipeline, "ocr_available", lambda: False)
    result = pipeline.process_input(pdf_path=scanned_pdf)
    assert result["source"] == "error"
    assert "OCR is unavailable" in result["error"]


def test_text_pdfs_never_pay_the_ocr_cost(monkeypatch):
    """A normal paper must not touch OCR at all."""
    monkeypatch.setattr(pipeline, "ocr_available", lambda: pytest.fail("OCR should not run"))
    result = pipeline.process_input(pdf_path=SAMPLE)
    assert result["source"] == "pdf"
