import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash"

CHUNK_SIZE = 20000  # bigger chunks = fewer API calls, respects the 20/day free limit
OVERLAP = 200
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


def _call_gemini_with_retry(prompt: str) -> str:
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except genai_errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"    DAILY QUOTA EXCEEDED. Free tier allows only 20 requests/day for {MODEL}.")
                print(f"    Wait until tomorrow, or reduce test runs. Raising error.")
            raise
        except genai_errors.ServerError:
            if attempt == MAX_RETRIES:
                raise
            print(f"    (server busy, retrying in {delay}s... attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2


def summarize_chunk(chunk: str, chunk_num: int, total_chunks: int) -> str:
    print(f"  -> summarizing chunk {chunk_num}/{total_chunks}...")
    prompt = f"""This is part {chunk_num} of {total_chunks} of a research paper.
Summarize the key points of THIS SECTION ONLY in 100-150 words.

Text:
{chunk}
"""
    result = _call_gemini_with_retry(prompt)
    time.sleep(2)
    return result


def reduce_summaries(chunk_summaries: list) -> str:
    print("  -> combining chunk summaries...")
    combined = "\n\n".join(chunk_summaries)
    prompt = f"""Below are summaries of consecutive sections of a research paper, in order.
Combine them into one coherent summary of the ENTIRE paper, preserving logical flow.

Section summaries:
{combined}
"""
    return _call_gemini_with_retry(prompt)


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
