
import re


def _looks_like_author_or_affiliation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    lower = stripped.lower()
    if re.search(r"@|\b(arxiv|doi|http|https|www\.|preprint)\b", lower):
        return True
    if re.search(r"\b(dept|department|university|institute|school|faculty|lab|college|center|centers)\b", lower):
        return True
    if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower):
        return True
    if lower.startswith(("abstract", "introduction", "keywords")):
        return True
    if stripped.count(",") >= 2 and len(stripped.split()) <= 12:
        return True
    return False


def extract_title(text: str) -> str:
    abstract_match = re.search(r"(?im)^\s*Abstract\b", text)
    if abstract_match:
        text_before_abstract = text[:abstract_match.start()]
    else:
        text_before_abstract = text

    lines = [line.strip() for line in text_before_abstract.split("\n") if line.strip()]
    title_lines = []

    for line in lines[:8]:
        if line.lower().startswith(("abstract", "introduction", "keywords")):
            break
        if _looks_like_author_or_affiliation(line):
            if title_lines:
                break
            continue

        title_lines.append(line)
        if len(title_lines) >= 2:
            break

    title = " ".join(title_lines).strip()
    if not title:
        return ""

    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"\s+([,.;:])", r"\1", title)
    return title


def extract_abstract(text: str) -> str:
    abstract_match = re.search(r"(?im)^\s*Abstract\b", text)
    if not abstract_match:
        return ""

    start = abstract_match.end()
    section_text = text[start:].lstrip()

    heading_match = re.search(
        r"(?im)^\s*(?:\d+\s*)?(?:Introduction|Related Work|Preliminaries|Problem Setup|Conclusion|Acknowledgements|Keywords)\b",
        section_text,
    )
    if heading_match:
        section_text = section_text[:heading_match.start()]

    lines = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("arxiv:", "doi:", "http://", "https://", "www.", "keywords", "acknowledgements")):
            continue
        if lower.startswith("abstract"):
            continue
        lines.append(stripped)

    abstract = "\n".join(lines).strip()
    return re.sub(r"\n{2,}", "\n", abstract)


def extract_metadata(text: str) -> dict:
    return {
        "title": extract_title(text),
        "abstract": extract_abstract(text)
    }

if __name__ == "__main__":
    from extract import extract_text
    from clean import clean_text

    raw = extract_text("tests/sample_papers/paper1.pdf")
    cleaned = clean_text(raw)
    metadata = extract_metadata(cleaned)

    print("TITLE:", metadata["title"])
    print("\nABSTRACT:", metadata["abstract"])
