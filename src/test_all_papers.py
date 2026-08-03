"""
Runs the full pipeline (extract -> clean -> metadata -> summarize) across
every PDF in tests/sample_papers/. Uses mock_gemini by default to test
logic/wiring without burning API quota. Set USE_REAL_API=True to run for real
(only when you have quota available).
"""

import os
import glob

from extract import extract_text
from clean import clean_text
from metadata import extract_metadata
from mock_gemini import fake_ask_gemini

USE_REAL_API = False  # flip to True only when quota is available


def run_pipeline_on_paper(pdf_path: str) -> dict:
    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)
    meta = extract_metadata(cleaned)

    if USE_REAL_API:
        from summarize import summarize_paper
        result = summarize_paper(cleaned)
    else:
        # mock path — verifies wiring/logic without real API calls
        result = {
            "summary": fake_ask_gemini(cleaned[:8000]),
            "methodology": fake_ask_gemini(cleaned[:8000]),
            "research_gaps": fake_ask_gemini(cleaned[:8000]),
            "findings": fake_ask_gemini(cleaned[:8000]),
            "future_work": fake_ask_gemini(cleaned[:8000]),
        }

    result["title"] = meta["title"]
    result["char_count"] = len(cleaned)
    return result


if __name__ == "__main__":
    papers = sorted(glob.glob("tests/sample_papers/*.pdf"))
    print(f"Found {len(papers)} papers to test.\n")

    for path in papers:
        name = os.path.basename(path)
        print(f"===== {name} =====")
        try:
            result = run_pipeline_on_paper(path)
            print(f"Title: {result['title']}")
            print(f"Length: {result['char_count']} chars")
            print(f"Mode: {'REAL API' if USE_REAL_API else 'MOCK (no API used)'}")
        except Exception as e:
            print(f"FAILED: {e}")
        print()
