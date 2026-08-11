import re

JUNK_PREFIXES = ["RESEARCH ARTICLE", "REVIEW ARTICLE", "ORIGINAL ARTICLE", "ARTICLE"]


def extract_title(text: str) -> str:
    lines = text.split("\n")
    title_lines = []
    for line in lines[:6]:
        stripped = line.strip()
        if stripped == "":
            continue
        # skip common junk prefixes that sometimes appear before the real title
        if stripped.upper() in JUNK_PREFIXES:
            continue
        title_lines.append(stripped)
        if len(title_lines) == 2:
            break
    return " ".join(title_lines)


def extract_abstract(text: str) -> str:
    match = re.search(r'Abstract\s*\n?(.*?)(?=\n1\s|\nIntroduction|\n1\nIntroduction)',
                       text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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
    print("\nABSTRACT:", metadata["abstract"][:800])
