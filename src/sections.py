import re

KNOWN_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "method", "methods", "approach",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "future work", "acknowledgments", "references",
    "preliminaries", "the model", "main results", "proofs"
]

def is_header_line(line: str):
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    cleaned = re.sub(r'^(?:\d+|[IVXLC]+)[\.\)]?\s+', '', stripped)
    normalized = cleaned.lower().strip()

    for header in KNOWN_HEADERS:
        if normalized == header:
            return header
    return None

def split_into_sections(text: str) -> dict:
    lines = text.split("\n")
    sections = {}
    current_section = "preamble"
    buffer = []

    for line in lines:
        header = is_header_line(line)
        if header:
            sections[current_section] = "\n".join(buffer).strip()
            current_section = header
            buffer = []
        else:
            buffer.append(line)

    sections[current_section] = "\n".join(buffer).strip()
    return sections

def chunk_by_length(text: str, chunk_size: int = 3000) -> dict:
    """Fallback: split text into roughly equal-sized numbered chunks
    when header detection doesn't produce a usable structure."""
    chunks = {}
    for i in range(0, len(text), chunk_size):
        chunk_num = (i // chunk_size) + 1
        chunks[f"chunk_{chunk_num}"] = text[i:i + chunk_size].strip()
    return chunks

def get_sections(text: str, imbalance_threshold: float = 0.7) -> dict:
    """Try header-based splitting first. If one section dominates
    (likely means headers weren't detected properly), fall back
    to length-based chunking instead."""
    sections = split_into_sections(text)

    total_len = len(text)
    largest_section_len = max(len(v) for v in sections.values())

    if total_len > 0 and (largest_section_len / total_len) > imbalance_threshold:
        print(f"[fallback triggered] one section held {largest_section_len}/{total_len} chars — using length-based chunking instead")
        return chunk_by_length(text)

    return sections

if __name__ == "__main__":
    from extract import extract_text
    from clean import clean_text

    raw = extract_text("tests/sample_papers/paper1.pdf")
    cleaned = clean_text(raw)
    sections = get_sections(cleaned)

    print("Sections found:", list(sections.keys()))
    print()
    for name, content in sections.items():
        preview = content[:150].replace("\n", " ")
        print(f"--- {name} ({len(content)} chars) ---")
        print(preview)
        print()
