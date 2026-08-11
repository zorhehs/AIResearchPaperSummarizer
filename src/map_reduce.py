import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 20000
OVERLAP = 100
MAX_RETRIES = 3
RETRY_DELAY = 10


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _call_groq_with_retry(prompt: str) -> str:
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    (error: {e}, retrying in {delay}s... attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2


def summarize_chunk(chunk: str, chunk_num: int, total_chunks: int) -> str:
    print(f"  -> summarizing chunk {chunk_num}/{total_chunks}...")
    prompt = f"""This is part {chunk_num} of {total_chunks} of a research paper.
Summarize the key points of THIS SECTION ONLY in 100-150 words.

Text:
{chunk}
"""
    result = _call_groq_with_retry(prompt)
    time.sleep(1)
    return result


def reduce_summaries(chunk_summaries: list) -> str:
    print("  -> combining chunk summaries...")
    combined = "\n\n".join(chunk_summaries)
    prompt = f"""Below are summaries of consecutive sections of a research paper, in order.
Combine them into one coherent summary of the ENTIRE paper, preserving logical flow.

Section summaries:
{combined}
"""
    return _call_groq_with_retry(prompt)


def get_condensed_text(full_text: str) -> str:
    if len(full_text) <= CHUNK_SIZE:
        print("  -> paper is short enough, skipping map-reduce")
        return full_text

    chunks = chunk_text(full_text)
    print(f"  -> paper split into {len(chunks)} chunks ({len(chunks) + 1} API calls total)")

    chunk_summaries = [
        summarize_chunk(chunk, i + 1, len(chunks))
        for i, chunk in enumerate(chunks)
    ]

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
