import unittest
from pathlib import Path

from src.clean import clean_text
from src.extract import extract_text
from src.metadata import extract_metadata


class MetadataExtractionTest(unittest.TestCase):
    def test_paper1_title_and_abstract(self) -> None:
        path = Path("tests/sample_papers/paper1.pdf")
        raw_text = extract_text(str(path))
        cleaned_text = clean_text(raw_text)

        metadata = extract_metadata(cleaned_text)

        self.assertIn("Strategy-Proofness", metadata["title"])
        self.assertIn("Committee Selection", metadata["title"])
        self.assertIn("strategyproofness", metadata["abstract"].lower())
        self.assertNotIn("Dael Sinay", metadata["title"])
        self.assertNotIn("arxiv", metadata["abstract"].lower())

    def test_paper2_title_and_abstract(self) -> None:
        path = Path("tests/sample_papers/paper2.pdf")
        raw_text = extract_text(str(path))
        cleaned_text = clean_text(raw_text)

        metadata = extract_metadata(cleaned_text)

        self.assertIn("CAP-DO", metadata["title"])
        self.assertNotIn("Mu Wang", metadata["title"])
        self.assertIn("security", metadata["abstract"].lower())
        self.assertNotIn("arxiv", metadata["abstract"].lower())


if __name__ == "__main__":
    unittest.main()
