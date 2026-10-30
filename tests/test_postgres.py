"""Live PostgreSQL integration tests; enabled by TEST_DATABASE_URL."""

import os
import unittest
from datetime import date
from pathlib import Path
from uuid import uuid4

from ingestion.csv_reader import read_csv
from ingestion.schemas import normalize_feedback


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL for live PostgreSQL tests")
class PostgresTests(unittest.TestCase):
    """Verify CSV -> SQL -> complete read-back, updates, and tenant separation."""

    @classmethod
    def setUpClass(cls):
        from storage.postgres import connect, create_tables

        cls.conn = connect(os.environ["TEST_DATABASE_URL"])
        create_tables(cls.conn)
        cls.tenant = f"test-{uuid4().hex}"

    @classmethod
    def tearDownClass(cls):
        try:
            cls.conn.execute("DELETE FROM feedback_reviews WHERE tenant_id = %s", (cls.tenant,))
        finally:
            cls.conn.close()

    def test_csv_to_postgres_round_trip_and_idempotence(self):
        from storage.postgres import count_reviews, get_review, upsert_reviews

        csv_path = Path(__file__).resolve().parents[1] / "data/sample_reviews.csv"
        records = list(read_csv(csv_path, tenant_id=self.tenant, source="demo-csv"))
        self.assertEqual(len(records), 2)

        self.assertEqual(upsert_reviews(self.conn, records), (2, 0))
        self.assertEqual(upsert_reviews(self.conn, records), (0, 2))
        self.assertEqual(count_reviews(self.conn, tenant_id=self.tenant, source="demo-csv"), 2)

        saved = get_review(self.conn, tenant_id=self.tenant, source="demo-csv", review_id="r-001")
        self.assertEqual(saved["text"], records[0].text)
        self.assertEqual(saved["company"], "Example Shop")
        self.assertEqual(saved["product_id"], "shop-app")
        self.assertEqual(saved["review_date"], date(2026, 10, 1))
        self.assertEqual(saved["source"], "demo-csv")
        self.assertEqual(saved["tenant_id"], self.tenant)
        self.assertEqual(saved["attributes"], records[0].attributes)
        self.assertEqual(len(saved["content_hash"]), 64)
        self.assertIsNotNone(saved["created_at"])
        self.assertIsNotNone(saved["updated_at"])

        changed = records[0].model_copy(update={"text": "Updated review", "attributes": {"rating": 2}})
        self.assertEqual(upsert_reviews(self.conn, [changed]), (1, 0))
        updated = get_review(self.conn, tenant_id=self.tenant, source="demo-csv", review_id="r-001")
        self.assertEqual(updated["text"], "Updated review")
        self.assertEqual(updated["attributes"], {"rating": 2})
        self.assertNotEqual(saved["content_hash"], updated["content_hash"])

    def test_source_and_tenant_isolation(self):
        from storage.postgres import get_review, upsert_reviews

        record = normalize_feedback({"review_id": "shared", "text": "Source A", "rating": 4},
                                    tenant_id=self.tenant, source="source-a")
        record_b = record.model_copy(update={"source": "source-b", "text": "Source B"})
        upsert_reviews(self.conn, [record, record_b])
        self.assertEqual(get_review(self.conn, tenant_id=self.tenant, source="source-a", review_id="shared")["text"], "Source A")
        self.assertEqual(get_review(self.conn, tenant_id=self.tenant, source="source-b", review_id="shared")["text"], "Source B")
        self.assertIsNone(get_review(self.conn, tenant_id="other-tenant", source="source-a", review_id="shared"))
        self.assertEqual(get_review(self.conn, tenant_id=self.tenant, source="source-a", review_id="shared")["attributes"]["rating"], 4)

    def test_same_batch_duplicate_identity_last_wins(self):
        from storage.postgres import get_review, upsert_reviews

        first = normalize_feedback({"review_id": "dup", "text": "First", "rating": 2},
                                   tenant_id=self.tenant, source="batch-duplicates")
        second = first.model_copy(update={"text": "Latest", "attributes": {"rating": 5}})
        self.assertEqual(upsert_reviews(self.conn, [first, second], batch_size=2), (1, 1))
        saved = get_review(self.conn, tenant_id=self.tenant, source="batch-duplicates", review_id="dup")
        self.assertEqual(saved["text"], "Latest")
        self.assertEqual(saved["attributes"], {"rating": 5})
        self.assertEqual(upsert_reviews(self.conn, [second], batch_size=1), (0, 1))

    def test_many_small_batches_replay(self):
        from storage.postgres import count_reviews, upsert_reviews
        records = [normalize_feedback({"review_id": f"batch-{i}", "text": f"Message {i}"},
                                      tenant_id=self.tenant, source="batch-replay") for i in range(13)]
        self.assertEqual(upsert_reviews(self.conn, iter(records), batch_size=3), (13, 0))
        self.assertEqual(upsert_reviews(self.conn, iter(records), batch_size=4), (0, 13))
        self.assertEqual(count_reviews(self.conn, tenant_id=self.tenant, source="batch-replay"), 13)

    def test_scraped_jsonl_metadata_round_trip(self):
        from ingestion.scraped_reader import read_scraped
        from storage.postgres import get_review, upsert_reviews

        path = Path(__file__).resolve().parents[1] / "data/sample_scraped.jsonl"
        records = list(read_scraped(path, tenant_id=self.tenant, source="scraper"))
        self.assertEqual(upsert_reviews(self.conn, records), (2, 0))
        saved = get_review(self.conn, tenant_id=self.tenant, source="scraper", review_id="web-001")
        self.assertEqual(saved["text"], "The app crashes every morning.")
        self.assertEqual(saved["url"], "https://example.org/reviews/1")
        self.assertEqual(saved["attributes"], {"rating": 1})
