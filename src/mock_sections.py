# Temporary mock data — mimics what Subhan's Phase 2 function will eventually return.
# Once his real section-detection function is ready, swap this import out.

from extract import extract_text
from clean import clean_text

def get_mock_sections():
    raw = extract_text("tests/sample_papers/paper1.pdf")
    cleaned = clean_text(raw)

    # crude manual split just so we have SOMETHING to build/test prompts against
    return {
        "full_text": cleaned  # for now, treat the whole paper as one blob
    }
