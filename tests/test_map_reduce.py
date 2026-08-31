import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from map_reduce import chunk_text, get_condensed_text


def test_chunk_text_basic():
    text = "a" * 45000
    chunks = chunk_text(text)
    assert len(chunks) == 3
    # each chunk is at most CHUNK_SIZE
    for c in chunks:
        assert len(c) <= 20000
    # chunks cover the full text with overlap
    assert chunks[0][19900:20000] == chunks[1][:100]
    assert chunks[1][19900:20000] == chunks[2][:100]


def test_chunk_text_short():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("") == []


def test_chunk_text_exact_multiple():
    text = "b" * 40000
    chunks = chunk_text(text)
    # 0-20000, 19900-39900, 39800-40000
    assert len(chunks) == 3
    assert len(chunks[-1]) == 200


def test_get_condensed_text_passthrough():
    short = "x" * 5000
    assert get_condensed_text(short) == short
