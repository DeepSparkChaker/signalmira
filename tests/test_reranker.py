"""No torch, network, GPU, model downloads, or database required."""
import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from retrieval.reranker import Reranker, RerankerConfig, RerankerBusy

SHA = "a" * 40


def docs():
    return [
        {"chunk_id": "positive", "chunk_text": "Battery life is very good.", "rrf_score": 0.03},
        {"chunk_id": "negative", "chunk_text": "Battery drains quickly.", "rrf_score": 0.02},
        {"chunk_id": "irrelevant", "chunk_text": "The screen is bright.", "rrf_score": 0.01},
    ]


class FakeModel:
    def __init__(self, output=(0.2, 5.0, -1.0), gate=None):
        self.output = output
        self.gate = gate
        self.pairs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        if self.gate is not None:
            self.gate.wait(timeout=3)
        return np.asarray(self.output[:len(pairs)], dtype=float)


class RerankerTests(unittest.TestCase):
    def test_default_is_qwen_and_bge_is_explicit(self):
        calls = []
        def loader(settings):
            calls.append(settings.model_id)
            return FakeModel()
        q = Reranker(RerankerConfig(revision=SHA), model_loader=loader)
        self.assertEqual(q.settings.model, "qwen")
        self.assertFalse(q.loaded)
        self.assertEqual(calls, [])  # NO model loaded in constructor
        try:
            ranked = q.rank("battery complaints", docs())
            self.assertEqual(ranked[0]["chunk_id"], "negative")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], "Qwen/Qwen3-Reranker-0.6B")
        finally:
            q.close()
        b = Reranker(RerankerConfig(revision=SHA, model="bge"), model_loader=loader)
        self.assertFalse(b.loaded)
        try:
            b.warmup()
            self.assertEqual(calls[-1], "BAAI/bge-reranker-v2-m3")
            self.assertEqual(calls.count("BAAI/bge-reranker-v2-m3"), 1)
        finally:
            b.close()

    def test_scores_metadata_dedup_and_input(self):
        fake = FakeModel()
        service = Reranker(RerankerConfig(revision=SHA), model_loader=lambda _: fake)
        try:
            ranked = service.rank("battery complaint", docs() + [docs()[0]], top_k=3)
            self.assertEqual([x["chunk_id"] for x in ranked],
                             ["negative", "positive", "irrelevant"])
            self.assertEqual(ranked[0]["rrf_score"], 0.02)
            self.assertEqual(ranked[0]["rerank_score"], 5.0)
            self.assertEqual(fake.pairs[1][1], "Battery drains quickly.")
            self.assertEqual(len(fake.pairs), 3)
            with self.assertRaises(ValueError):
                service.rank("", docs())
            with self.assertRaises(ValueError):
                service.rank("question", [{"chunk_id": 1, "text": "wrong key"}])
            with self.assertRaises(ValueError):
                service.rank("question", docs() * 9)
        finally:
            service.close()

    def test_qwen_failure_never_loads_bge_and_falls_back(self):
        calls = []
        def broken(settings):
            calls.append(settings.model)
            raise RuntimeError("GPU error")
        service = Reranker(RerankerConfig(revision=SHA), model_loader=broken)
        try:
            outcome = asyncio.run(service.arank_with_status("battery", docs()))
            self.assertEqual(outcome.status, "rrf_fallback")
            self.assertEqual(outcome.reason, "model_error")
            self.assertEqual([d["chunk_id"] for d in outcome.results],
                             ["positive", "negative", "irrelevant"])
            self.assertEqual(calls, ["qwen"])
        finally:
            service.close()

    def test_timeout_does_not_release_busy_worker_early(self):
        gate = threading.Event()
        service = Reranker(RerankerConfig(revision=SHA, timeout_s=0.02),
                           model_loader=lambda _: FakeModel(gate=gate))
        try:
            async def run():
                first = await service.arank_with_status("battery", docs())
                second = await service.arank_with_status("battery", docs())
                return first, second
            first, second = asyncio.run(run())
            self.assertEqual(first.reason, "timeout")
            self.assertEqual(second.reason, "busy")
        finally:
            gate.set()
            service.close()

    def test_ranking_stable_ties_and_nonfinite(self):
        ties = Reranker(RerankerConfig(revision=SHA),
                        model_loader=lambda _: FakeModel(output=(1., 1., 1.)))
        try:
            self.assertEqual([x["chunk_id"] for x in ties.rank("battery", docs())],
                             ["positive", "negative", "irrelevant"])
        finally:
            ties.close()
        invalid = Reranker(RerankerConfig(revision=SHA),
                           model_loader=lambda _: FakeModel(output=(float("nan"), 2., 3.)))
        try:
            result = asyncio.run(invalid.arank_with_status("battery", docs()))
            self.assertEqual(result.status, "rrf_fallback")
        finally:
            invalid.close()

    def test_invalid_revision_and_bge_qwen_instruction(self):
        with self.assertRaises(ValueError):
            RerankerConfig(revision="main")
        with self.assertRaises(ValueError):
            RerankerConfig(revision=SHA, model="bge", qwen_instruction="Only complaints")


if __name__ == "__main__":
    unittest.main()
