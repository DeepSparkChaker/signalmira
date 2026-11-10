"""No database or GPU needed."""
import unittest
from scripts.beir_pg_full_eval import unique_documents, guard_database


class _Cursor:
    def __init__(self, row):
        self.row = row
    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, name):
        self.name = name
    def execute(self, *args):
        return _Cursor({"db": self.name})


class FullBenchmarkTests(unittest.TestCase):
    def test_doc_dedup_preserves_rank(self):
        rows = [{"review_id": "d1"}, {"review_id": "d1"}, {"review_id": "d2"}]
        self.assertEqual(unique_documents(rows, 2), ["d1", "d2"])
    def test_db_guard_rejects_demo(self):
        with self.assertRaises(RuntimeError):
            guard_database(_Connection("signalmira"))
    def test_db_guard_accepts_isolated(self):
        guard_database(_Connection("signalmira_beir"))


if __name__ == "__main__":
    unittest.main()
