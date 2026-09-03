import unittest
from pathlib import Path

from src.clean import clean_text
from src.extract import extract_pdf_metadata, extract_text
from src.metadata import extract_authors, extract_metadata, extract_title, extract_year


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

    def test_extract_title_skips_junk_lines(self) -> None:
        text = (
            "arXiv:2401.12345v2 [cs.AI] 15 Jan 2024\n"
            "JOURNAL OF TESTING THINGS\n"
            "A Genuinely Good Title for a Paper\n"
            "John Doe1*, Jane Smith2\n"
            "Abstract\nWe study things.\n"
        )
        self.assertEqual(extract_title(text), "A Genuinely Good Title for a Paper")

    def test_extract_title_returns_empty_for_garbage(self) -> None:
        self.assertEqual(extract_title("arXiv:2401.12345v2 [cs.AI]\nhttps://example.com\n1\n"), "")

    def test_extract_authors_synthetic(self) -> None:
        text = (
            "Some Long Paper Title About Things\n"
            "John Doe1*, Jane Smith2 and Bob Lee1\n"
            "1 University of Somewhere, 2 Other Institute\n"
            "Abstract\nWe study things.\n"
        )
        authors = extract_authors(text, title="Some Long Paper Title About Things")
        self.assertIn("John Doe", authors)
        self.assertIn("Jane Smith", authors)
        self.assertIn("Bob Lee", authors)
        self.assertNotIn("University of Somewhere", " ".join(authors))

    def test_extract_year(self) -> None:
        self.assertEqual(extract_year("© 2023 IEEE. Some text"), "2023")
        self.assertEqual(extract_year("Published in 2019, this work..."), "2019")
        self.assertEqual(extract_year("no year here"), "")

    def test_pdf_embedded_metadata_shape(self) -> None:
        meta = extract_pdf_metadata("tests/sample_papers/paper1.pdf")
        self.assertIsInstance(meta, dict)
        self.assertIn("title", meta)
        self.assertIn("author", meta)


if __name__ == "__main__":
    unittest.main()
