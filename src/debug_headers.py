from extract import extract_text
from clean import clean_text

raw = extract_text("tests/sample_papers/paper1.pdf")
cleaned = clean_text(raw)

lines = cleaned.split("\n")
for i, line in enumerate(lines):
    stripped = line.strip().lower()
    if "introduction" in stripped or "method" in stripped or "conclusion" in stripped:
        print(f"Line {i}: {repr(line)}")
