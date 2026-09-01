"""
Runs the full pipeline (extract -> clean -> metadata -> summarize) across
every PDF in tests/sample_papers/. Uses the real Groq-based summarize_paper().
"""

import os
import glob

from extract import extract_text
from clean import clean_text
from metadata import extract_metadata
from summarize import summarize_paper


def run_pipeline_on_paper(pdf_path: str) -> dict:
    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)
    meta = extract_metadata(cleaned)

    result = summarize_paper(cleaned)
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
            print(f"\nSUMMARY:\n{result['summary']}\n")
        except Exception as e:
            print(f"FAILED: {e}")
        print()
