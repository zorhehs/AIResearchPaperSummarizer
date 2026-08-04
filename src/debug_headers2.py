import re
from extract import extract_text
from clean import clean_text

raw = extract_text("tests/sample_papers/paper1.pdf")
cleaned = clean_text(raw)

lines = cleaned.split("\n")
for i, line in enumerate(lines):
    stripped = line.strip()
    # crude check: short line, starts with a digit (likely a numbered heading)
    if stripped and len(stripped) < 60 and re.match(r'^\d+[\.\)]?\s+\S', stripped):
        print(f"Line {i}: {repr(line)}")
