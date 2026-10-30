"""Fast unit tests: source normalization, evidence preservation, bad input."""

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.csv_reader import read_csv
from ingestion.schemas import normalize_feedback
from ingestion.scraped_reader import read_scraped


class IngestionTests(unittest.TestCase):
    def test_csv_streams_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.csv"
            path.write_text('review_id,text,label\nR1,"Great, thanks!",1\nR2,Bad service,0\n', encoding="utf-8")
            records = list(read_csv(path, tenant_id="demo", source="uci"))
            self.assertEqual([r.review_id for r in records], ["R1", "R2"])
            self.assertEqual(records[0].text, "Great, thanks!")
            self.assertEqual(records[0].attributes["label"], "1")
            self.assertEqual(records[0].tenant_id, "demo")

    def test_scraped_jsonl_aliases_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scraped.jsonl"
            path.write_text(json.dumps({"id": "a12", "review": "Nice app", "url": "https://example.com/a12", "rating": 5}) + "\n", encoding="utf-8")
            record, = list(read_scraped(path, tenant_id="demo", source="scraper"))
            self.assertEqual(record.review_id, "a12")
            self.assertEqual(record.text, "Nice app")
            self.assertEqual(record.attributes["rating"], 5)

    def test_rejects_huge_jsonl_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huge.jsonl"
            path.write_text('{"text":"' + 'a' * 2_000_001 + '"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "oversized JSONL record"):
                list(read_scraped(path, tenant_id="demo", source="scraper"))

    def test_small_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scraped.json"
            path.write_text('[{"id":"1","content":"Useful"}]', encoding="utf-8")
            record, = list(read_scraped(path, tenant_id="demo", source="scraper"))
            self.assertEqual(record.text, "Useful")

    def test_invalid_review_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing or empty text"):
            normalize_feedback({"text": "   "}, tenant_id="demo", source="scraper")

    def test_stable_fallback_id(self):
        record = {"review": "Works fine", "url": "https://example.com/1"}
        first = normalize_feedback(record, tenant_id="demo", source="scraper")
        second = normalize_feedback(record, tenant_id="demo", source="scraper")
        self.assertEqual(first.review_id, second.review_id)
        self.assertTrue(first.review_id.startswith("auto-"))

    def test_date_and_provenance(self):
        record = normalize_feedback({"text":"Great!","created_at":"2026-10-28T12:30:00Z", "source":"external"}, tenant_id="demo", source="trusted")
        self.assertEqual(record.review_date.isoformat(), "2026-10-28")
        self.assertEqual(record.source, "trusted")
        self.assertEqual(record.attributes["source"], "external")


if __name__ == "__main__":
    unittest.main()
