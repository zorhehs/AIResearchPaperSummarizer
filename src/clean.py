
import re
from collections import Counter

def fix_hyphenation(text: str) -> str:
    # Joins words split by hyphen at line break, e.g. "optimiza-\ntion" -> "optimization"
    return re.sub(r'-\n(\w)', r'\1', text)

def remove_repeated_lines(text: str) -> str:
    lines = text.split("\n")
    line_counts = Counter(lines)
    threshold = 3  # lines appearing 3+ times are likely headers/footers/watermarks
    cleaned = [line for line in lines if line_counts[line] < threshold or line.strip() == ""]
    return "\n".join(cleaned)

def remove_references(text: str) -> str:
    markers = ["References", "REFERENCES", "Bibliography"]
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            return text[:idx]
    return text

def clean_text(raw_text: str) -> str:
    text = fix_hyphenation(raw_text)
    text = remove_repeated_lines(text)
    text = remove_references(text)
    return text

if __name__ == "__main__":
    from extract import extract_text

    for paper in ["paper1.pdf", "paper2.pdf"]:
        path = f"tests/sample_papers/{paper}"
        raw = extract_text(path)
        cleaned = clean_text(raw)

        print(f"\n===== {paper} =====")
        print(f"Raw length: {len(raw)} chars")
        print(f"Cleaned length: {len(cleaned)} chars")
        print("\n--- Cleaned preview ---")
        print(cleaned[:1500])
