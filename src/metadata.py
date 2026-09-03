import re

JUNK_PREFIXES = ["RESEARCH ARTICLE", "REVIEW ARTICLE", "ORIGINAL ARTICLE", "ARTICLE"]

# Lines that are banners/stamps/headers rather than the paper title
_JUNK_LINE_PATTERNS = [
    re.compile(r"^arxiv\b", re.I),
    re.compile(r"^doi[:\s]", re.I),
    re.compile(r"https?://\S+", re.I),
    re.compile(r"^\d+$"),
    re.compile(r"^(proceedings|journal of|transactions on|volume \d|vol\.|issue \d)", re.I),
    re.compile(r"^(received|accepted|revised|published|copyright|©|license|issn|isbn)", re.I),
    re.compile(r"^preprint", re.I),
]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_AFFILIATION_RE = re.compile(
    r"\b(department|university|universit|institute|laborator|school of|"
    r"college of|faculty of|academy|center for|centre for)\b",
    re.I,
)


def _is_junk_line(line: str) -> bool:
    if any(p.search(line) for p in _JUNK_LINE_PATTERNS):
        return True
    if _EMAIL_RE.search(line) or _AFFILIATION_RE.search(line):
        return True
    return False


def _looks_like_author_line(line: str) -> bool:
    """Author listings carry superscript affiliation markers; titles don't."""
    return bool(
        re.search(r"\d{1,2}\s*[*†‡§]", line)
        and not re.search(r"\b(?:19|20)\d{2}\b", line)
    )


def extract_title(text: str) -> str:
    lines = text.split("\n")
    title_lines = []
    for line in lines[:12]:
        stripped = line.strip()
        if not stripped or len(stripped) < 8:
            continue
        # skip common junk prefixes that sometimes appear before the real title
        if stripped.upper() in JUNK_PREFIXES:
            continue
        # the abstract heading marks the end of the title block
        if stripped.lower() in {"abstract", "abstract:", "summary"}:
            break
        if _is_junk_line(stripped) or _looks_like_author_line(stripped):
            continue
        title_lines.append(stripped)
        if len(title_lines) == 2:
            break
    # Drop trailing author lines like "Mu Wang1*, Zhenkun Liu1,2" that got pulled in
    while title_lines and re.search(r'\d{1,2}\s*[*†‡]?\s*$', title_lines[-1]):
        title_lines.pop()
    title = " ".join(title_lines)
    return title if 10 <= len(title) <= 300 else ""


def extract_authors(text: str, title: str = "") -> list:
    """Heuristically pull author names from the top of the paper.

    Scans the lines following the title for comma-separated capitalized name
    groups (author lines sit right below the title and usually carry
    superscript affiliation markers). Returns at most 12 names.
    """
    lines = [l.strip() for l in text.split("\n")[:15]]
    start = 0
    if title:
        for i, line in enumerate(lines):
            if line and line in title:
                start = i + 1
                break

    authors = []
    for line in lines[start:]:
        if not line or len(line) < 8 or len(line) > 200:
            continue
        if _is_junk_line(line):
            continue
        names = []
        for token in re.split(r"[,;]|\band\b|&", line):
            token = re.sub(r"[\d*†‡§#]+", "", token).strip(" .-*'")
            if 3 <= len(token) <= 40 and re.match(
                r"^[A-ZÀ-Ý][\w.'’-]*(?:\s+[A-ZÀ-Ý][\w.'’-]*){1,3}$", token
            ):
                names.append(token)
        if len(names) >= 2:
            authors.extend(names)
            break

    seen = set()
    return [a for a in authors if not (a in seen or seen.add(a))][:12]


def extract_year(text: str) -> str:
    head = text[:2000]
    m = re.search(r"©\s*((?:19|20)\d{2})", head)
    if m:
        return m.group(1)
    m = re.search(r"\b((?:19|20)\d{2})\b", head)
    return m.group(1) if m else ""


def extract_abstract(text: str) -> str:
    match = re.search(r'Abstract\s*\n?(.*?)(?=\n1\s|\nIntroduction|\n1\nIntroduction)',
                       text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_metadata(text: str) -> dict:
    title = extract_title(text)
    return {
        "title": title,
        "abstract": extract_abstract(text),
        "authors": extract_authors(text, title=title),
        "year": extract_year(text),
    }


if __name__ == "__main__":
    from extract import extract_text
    from clean import clean_text

    raw = extract_text("tests/sample_papers/paper1.pdf")
    cleaned = clean_text(raw)
    metadata = extract_metadata(cleaned)

    print("TITLE:", metadata["title"])
    print("AUTHORS:", metadata["authors"])
    print("YEAR:", metadata["year"])
    print("\nABSTRACT:", metadata["abstract"][:800])
