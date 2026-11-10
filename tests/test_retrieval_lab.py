"""Pure local smoke tests; no database, model, GPU, or Streamlit required."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from frontend.retrieval_lab import LabConfig, SAMPLE_QUERY, search_lab


class FakeEmbedder:
    spec = SimpleNamespace(model_key="a" * 64)
    def encode_query(self, text):
        assert text == SAMPLE_QUERY
        return [0.0] * 1023 + [1.0]


class FakeReranker:
    async def arank_with_status(self, query, candidates, *, top_k):
        assert query == SAMPLE_QUERY
        assert len(candidates) == 3
        result = [dict(row, rerank_score=float(10 - row["rrf_rank"])) for row in reversed(candidates)]
        return SimpleNamespace(results=result[:top_k], status="reranked", reason=None, latency_ms=4.5)


class FakeFallback:
    async def arank_with_status(self, query, candidates, *, top_k):
        return SimpleNamespace(results=candidates[:top_k], status="rrf_fallback", reason="timeout", latency_ms=10.0)


DATA = [
    {"chunk_id": "x", "review_id": "a", "chunk_text": "Good battery", "rrf_score": 0.2},
    {"chunk_id": "y", "review_id": "b", "chunk_text": "Battery drain", "rrf_score": 0.1},
    {"chunk_id": "z", "review_id": "c", "chunk_text": "Bad battery", "rrf_score": 0.05},
]


class LabTests(unittest.TestCase):
    def test_qwen_only_when_configured(self):
        config = LabConfig(reranker_model="off")
        self.assertEqual(config.reranker_model, "off")
        self.assertEqual(config.reranker_revision, "")
        with self.assertRaises(ValueError):
            LabConfig(reranker_model="qwen")
        with self.assertRaises(ValueError):
            LabConfig(reranker_model="bge")

    @patch("frontend.retrieval_lab._search_vector", return_value=DATA)
    def test_rrf_only(self, search):
        result = search_lab(object(), "demo", SAMPLE_QUERY, FakeEmbedder(), None,
                            lexical_backend="fts")
        self.assertEqual(result["status"], "rrf_only")
        self.assertEqual([r["review_id"] for r in result["results"]], ["a", "b", "c"])
        self.assertEqual(result["results"][0]["rrf_rank"], 1)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(search.call_args.kwargs["model_key"], "a" * 64)

    @patch("frontend.retrieval_lab._search_vector", return_value=DATA)
    def test_rerank_preserves_rrf_and_metadata(self, search):
        result = search_lab(object(), "demo", SAMPLE_QUERY, FakeEmbedder(), FakeReranker())
        self.assertEqual(result["status"], "reranked")
        self.assertEqual(result["results"][0]["chunk_id"], "z")
        self.assertEqual(result["results"][0]["rrf_rank"], 3)
        self.assertEqual(result["rrf_top8"][0]["chunk_id"], "x")
        self.assertEqual(result["rerank_ms"], 4.5)

    @patch("frontend.retrieval_lab._search_vector", return_value=DATA)
    def test_timeout_returns_rrf(self, search):
        result = search_lab(object(), "demo", SAMPLE_QUERY, FakeEmbedder(), FakeFallback())
        self.assertEqual(result["status"], "rrf_fallback")
        self.assertEqual(result["reason"], "timeout")
        self.assertEqual([r["chunk_id"] for r in result["results"]], ["x", "y", "z"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            search_lab(object(), "demo", "", FakeEmbedder(), None)
        with self.assertRaises(ValueError):
            search_lab(object(), "demo", SAMPLE_QUERY, FakeEmbedder(), None, mode="foo")


if __name__ == "__main__":
    unittest.main()
