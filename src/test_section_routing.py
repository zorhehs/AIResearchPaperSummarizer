"""
Verifies section-aware routing logic in summarize_paper() WITHOUT making
any real API calls. Prints which text would be sent to each prompt.
"""

import sys
sys.path.insert(0, "src") if "src" not in sys.path else None

from extract import extract_text
from clean import clean_text
from sections import get_sections
from summarize import _get_relevant_text


def dry_run(pdf_path: str):
    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)
    sections = get_sections(cleaned)

    has_real_sections = not all(k.startswith("chunk_") for k in sections.keys())

    print(f"\n===== {pdf_path} =====")
    print(f"Sections detected: {list(sections.keys())}")
    print(f"Has real section labels: {has_real_sections}")

    if has_real_sections:
        methodology_text = _get_relevant_text(sections, ["methodology", "method", "methods", "approach", "system design", "proposed method", "introduction"], "[FALLBACK]")
        results_text = _get_relevant_text(sections, ["results", "evaluation", "main results", "experiments", "experimental results"], "[FALLBACK]")
        gaps_text = _get_relevant_text(sections, ["limitations", "discussion"], "[FALLBACK]")
        future_text = _get_relevant_text(sections, ["future work", "conclusion", "conclusions"], "[FALLBACK]")

        print(f"\nMethodology prompt would receive: {len(methodology_text)} chars", "(FALLBACK USED)" if methodology_text == "[FALLBACK]" else "")
        print(f"Findings prompt would receive: {len(results_text)} chars", "(FALLBACK USED)" if results_text == "[FALLBACK]" else "")
        print(f"Research gaps prompt would receive: {len(gaps_text)} chars", "(FALLBACK USED)" if gaps_text == "[FALLBACK]" else "")
        print(f"Future work prompt would receive: {len(future_text)} chars", "(FALLBACK USED)" if future_text == "[FALLBACK]" else "")
    else:
        print("\n-> Fallback chunking triggered, all 5 prompts will use full condensed text (same as before)")


if __name__ == "__main__":
    import glob
    for path in sorted(glob.glob("tests/sample_papers/*.pdf")):
        dry_run(path)
