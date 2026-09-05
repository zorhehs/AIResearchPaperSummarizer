
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

_REFERENCES_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*)?(references|bibliography|works cited)[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def remove_references(text: str) -> str:
    """Drop the reference list from the end of a paper.

    Only a line that is *nothing but* a references heading counts, so an
    in-body mention ("see the references in Section 4") no longer truncates
    the paper there. The heading is searched for in the back half of the text,
    because a bibliography never opens a paper but the phrase may well appear
    in an abstract or introduction.
    """
    if not text:
        return text
    halfway = len(text) // 2
    matches = [m for m in _REFERENCES_HEADING.finditer(text) if m.start() >= halfway]
    if matches:
        return text[: matches[-1].start()]
    return text

def remove_arxiv_watermark(text: str) -> str:
    # arXiv stamp lines like "arXiv:2607.24676v1  [cs.GT]  27 Jul 2026"
    return "\n".join(
        line for line in text.split("\n")
        if not re.match(r'\s*arxiv:\s*\S+', line, re.IGNORECASE)
    )

def clean_text(raw_text: str) -> str:
    text = fix_hyphenation(raw_text)
    text = remove_repeated_lines(text)
    text = remove_references(text)
    text = remove_arxiv_watermark(text)
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
