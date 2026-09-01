import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_SIZE = 20000
# Overlap in chars between consecutive chunks. Large enough to bridge a
# paragraph boundary so context is not lost at the cut.
OVERLAP = 800
MAX_RETRIES = 3
RETRY_DELAY = 10
MAX_WORKERS = 5

_SEPARATORS = ("\n\n", "\n", ". ")


def _find_cut(text: str, lo: int, hi: int):
    """Best split point in [lo, hi): paragraph break > line break > sentence
    end. Returns the index just AFTER the separator, or None for a hard cut."""
    for sep in _SEPARATORS:
        idx = text.rfind(sep, lo, hi)
        if idx != -1:
            return idx + len(sep)
    return None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """Split text into chunks of at most `chunk_size` chars.

    Cuts are aligned to natural boundaries wherever possible - paragraph
    breaks, then line breaks (what most PDF extractors emit), then sentence
    ends - so sentences, tables, and references are never split mid-way.
    Falls back to a hard cut for dense text.
    """
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # Prefer a natural boundary within the back half of the chunk
            cut = _find_cut(text, start + chunk_size // 2, end)
            if cut and cut > start:
                end = cut
        chunks.append(text[start:end])
        if end >= n:
            break
        # Next chunk starts `overlap` chars before the cut, realigned to the
        # nearest natural boundary at or after that point.
        nxt = end - overlap
        if nxt <= start:
            nxt = end - overlap if end - overlap > start else end
        else:
            cut = _find_cut(text, nxt, end)
            if cut and cut > nxt:
                nxt = cut
        start = max(nxt, start + 1)
    return chunks


def _call_groq_with_retry(prompt: str) -> str:
    """Call the shared Groq helper (with model fallback) wrapped in retry/backoff."""
    from summarize import _ask_groq

    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _ask_groq([{"role": "user", "content": prompt}])
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    (error: {e}, retrying in {delay}s... attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2


def summarize_chunk(chunk: str, chunk_num: int, total_chunks: int) -> str:
    print(f"  -> summarizing chunk {chunk_num}/{total_chunks}...")
    prompt = f"""This is part {chunk_num} of {total_chunks} of a research paper.
Extract the key information from THIS SECTION ONLY as terse bullet points (one per line, starting with "- ").
For each bullet capture one concrete fact: a claim, method, dataset, or result.
ALWAYS preserve exact numbers, metric names, percentages, dataset names, and model names verbatim.
Do NOT write flowing prose. Do NOT add commentary. 5-10 bullets maximum.

Text:
{chunk}
"""
    return _call_groq_with_retry(prompt)


def reduce_summaries(chunk_summaries: list) -> str:
    print("  -> combining chunk summaries...")
    combined = "\n\n".join(chunk_summaries)
    prompt = f"""Below are extracted facts from consecutive sections of a research paper, in order.
Synthesize them into one coherent brief of the ENTIRE paper (250-400 words) that preserves logical flow.
Rules:
- Preserve ALL exact figures verbatim (accuracies, percentages, dataset sizes, model names).
- Organize as: what the paper does, how, and what it achieves, with the key numbers inline.
- Drop duplicates and trivia; never invent a number that is not in the facts.

Extracted facts:
{combined}
"""
    return _call_groq_with_retry(prompt)


def get_condensed_text(full_text: str) -> str:
    if len(full_text) <= CHUNK_SIZE:
        print("  -> paper is short enough, skipping map-reduce")
        return full_text

    chunks = chunk_text(full_text)
    print(f"  -> paper split into {len(chunks)} chunks ({len(chunks) + 1} API calls total)")

    # Chunks are independent → summarize them in parallel. Rate limiting is
    # handled inside _ask_groq's retry/backoff, so no artificial sleeps here.
    chunk_summaries = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as executor:
        futures = {
            executor.submit(summarize_chunk, chunk, i + 1, len(chunks)): i
            for i, chunk in enumerate(chunks)
        }
        for fut in as_completed(futures):
            chunk_summaries[futures[fut]] = fut.result()

    return reduce_summaries(chunk_summaries)


if __name__ == "__main__":
    from extract import extract_text
    from clean import clean_text

    raw = extract_text("tests/sample_papers/paper1.pdf")
    cleaned = clean_text(raw)

    print(f"Original length: {len(cleaned)} chars")
    condensed = get_condensed_text(cleaned)
    print(f"Condensed length: {len(condensed)} chars")
    print("\n===== CONDENSED TEXT =====\n")
    print(condensed)
