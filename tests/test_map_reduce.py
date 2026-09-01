import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from map_reduce import CHUNK_SIZE, OVERLAP, chunk_text, get_condensed_text


def test_chunk_text_basic():
    text = "a" * 45000
    chunks = chunk_text(text)
    assert len(chunks) == 3
    # each chunk is at most CHUNK_SIZE
    for c in chunks:
        assert len(c) <= CHUNK_SIZE
    # chunks cover the full text with overlap
    assert chunks[0][-OVERLAP:] == chunks[1][:OVERLAP]
    assert chunks[1][-OVERLAP:] == chunks[2][:OVERLAP]
    # overlap region does not push the final chunk past the text end
    assert sum(len(c) for c in chunks) > len(text)
    assert "".join(chunks).count(text[-1000:]) >= 1


def test_chunk_text_short():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("") == []


def test_chunk_text_exact_multiple():
    text = "b" * 40000
    chunks = chunk_text(text)
    # 0-20000, 19200-39200, 38400-40000 (overlap 800, no paragraph breaks)
    assert len(chunks) == 3
    assert len(chunks[-1]) == 1600


def test_chunk_text_cuts_at_paragraph_boundaries():
    # 400 numbered paragraphs of 60 chars each = 24000 chars → 2 chunks
    paragraphs = [f"P{i:03d} " + "a" * 50 + "\n\n" for i in range(400)]
    text = "".join(paragraphs)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    search_from = 0
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= CHUNK_SIZE
        # paragraph markers are unique, so each chunk's head locates its start
        start = text.find(chunk[:60], search_from)
        assert start != -1, "chunk content not found in order"
        assert start >= search_from, "chunk starts must advance"
        search_from = start + 1
        # every non-final cut lands just after a paragraph break
        if i < len(chunks) - 1:
            assert chunk.endswith("\n\n"), "chunk split mid-paragraph"


def test_chunk_text_falls_back_to_line_breaks():
    # single newlines only (typical of PDF text extraction)
    lines = [f"line {i:04d} " + "a" * 40 for i in range(600)]
    text = "\n".join(lines)
    assert "\n\n" not in text
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert chunk.endswith("\n"), "chunk split mid-line"
        assert len(chunk) <= CHUNK_SIZE


def test_chunk_text_falls_back_to_sentence_ends():
    # no newlines at all — only sentence ends
    text = "Sentence one is here. Sentence two follows it. " * 500
    assert "\n" not in text
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert chunk.endswith(". "), "chunk split mid-sentence"
        assert len(chunk) <= CHUNK_SIZE


def test_chunk_text_start_positions_advance():
    # ~72k chars, every line unique so chunk positions are unambiguous
    lines = [
        f"para {i:03d} " + "".join(chr(97 + (i * 7 + j * 13) % 26) for j in range(2000))
        for i in range(30)
    ]
    text = "\n\n".join(lines)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    search_from = 0
    prev_end = 0
    for chunk in chunks:
        assert len(chunk) > 0
        start = text.find(chunk[:80], search_from)
        assert start != -1, "chunk content not found in order"
        # next chunk may overlap by up to OVERLAP chars, but must never leave
        # a gap (start > prev_end) or move coverage backward (start < prev_end - OVERLAP)
        assert prev_end - OVERLAP <= start <= prev_end
        search_from = start + 1
        prev_end = start + len(chunk)
    # chunks (with their overlaps) cover the entire text
    assert prev_end >= len(text)


def test_get_condensed_text_passthrough():
    short = "x" * 5000
    assert get_condensed_text(short) == short
