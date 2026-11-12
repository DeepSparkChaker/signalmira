"""Offline, GPU-free regression tests for the final SignalMira extraction patch."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from enrichment.opinion_schema import EvidenceSpan, Opinion, ReviewOpinions
from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor, OpinionBusy
from llm_providers.llm_factory import get_llm

SHA64 = "a" * 64
SHA40 = "b" * 40
TEXT = "Camera excellent, battery terrible."


def fake_output(text=TEXT):
    quote = "battery terrible"
    start = text.index(quote)
    return ReviewOpinions(overall_sentiment="mixed", opinions=[
        Opinion(aspect="battery", sentiment="negative",
                evidence=EvidenceSpan(quote=quote, start=start, end=start + len(quote))),
    ])


class FakeLLM:
    def __init__(self, fn=None):
        self.fn = fn or (lambda _messages: fake_output())
        self.calls = 0
        self.method = None

    def with_structured_output(self, schema, *, method):
        self.method = method
        self.schema = schema
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.fn(messages)


class EvidenceTests(unittest.TestCase):
    def test_exact_span(self):
        result = fake_output().align(TEXT)
        self.assertEqual(result.opinions[0].evidence.quote, "battery terrible")

    def test_unicode_offsets(self):
        source = "البطارية ضعيفة، caméra magnifique"
        quote = "caméra magnifique"
        span = EvidenceSpan(quote=quote, start=0, end=len(quote))
        fixed = span.align(source)
        self.assertEqual(source[fixed.start:fixed.end], quote)

    def test_ambiguous_quote_fails(self):
        source = "Bad battery. Bad battery."
        span = EvidenceSpan(quote="Bad", start=1, end=4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            span.align(source)

    def test_invented_quote_fails(self):
        with self.assertRaises(ValueError):
            EvidenceSpan(quote="invented", start=0, end=8).align(TEXT)

    def test_duplicate_opinion_fails(self):
        x = fake_output()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            x.model_copy(update={"opinions": x.opinions + x.opinions}).verify(TEXT)

    def test_wrong_schema_field_fails(self):
        with self.assertRaises(ValueError):
            ReviewOpinions.model_validate({"overall_sentiment": "mixed", "opinions": [], "invented": 1})


class ConfigTests(unittest.TestCase):
    def test_default_url_is_raw_not_markdown(self):
        self.assertEqual(OpinionConfig("qwen3.5:4b", SHA64).url,
                         "http://127.0.0.1:11434/api/chat")

    def test_reject_wrong_ollama_digest(self):
        with self.assertRaises(ValueError):
            OpinionConfig("qwen3.5:4b", SHA40)

    def test_reject_ssrf_endpoint(self):
        with self.assertRaises(ValueError):
            OpinionConfig("qwen3.5:4b", SHA64, url="http://example.org/api/chat")

    def test_reject_paid_without_permission(self):
        with self.assertRaises(PermissionError):
            OpinionConfig("some-model", "release-id", provider="openai")

    def test_config_hash_changed_by_provider(self):
        a = OpinionConfig("qwen3.5:4b", SHA64)
        b = OpinionConfig("qwen3.5:4b", SHA40, provider="vllm")
        self.assertNotEqual(a.config_hash, b.config_hash)

    def test_vllm_deployment_attestation(self):
        c = OpinionConfig("Qwen/Qwen3.5-4B", SHA40, provider="vllm")
        extractor = LangChainOpinionExtractor(c)
        try:
            with patch.dict(os.environ, {"VLLM_DEPLOYED_REVISION": "x" * 40}):
                with self.assertRaises(RuntimeError):
                    extractor.verify_model_digest()
            with patch.dict(os.environ, {"VLLM_DEPLOYED_REVISION": SHA40}):
                extractor.verify_model_digest()
        finally:
            extractor.close()


class ExtractorTests(unittest.TestCase):
    def test_works_with_injected_langchain_model(self):
        llm = FakeLLM()
        ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64), llm=llm)
        try:
            result = ext.extract(TEXT)
            self.assertEqual(result.opinions[0].aspect, "battery")
            self.assertEqual(llm.method, "json_schema")
            self.assertEqual(llm.calls, 1)
        finally:
            ext.close()

    def test_reject_long_input_before_inference(self):
        llm = FakeLLM()
        ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64,
                                                      max_review_chars=100), llm=llm)
        try:
            with self.assertRaises(ValueError):
                ext.extract("x" * 101)
            self.assertEqual(llm.calls, 0)
        finally:
            ext.close()

    def test_unverifiable_llm_text_rejected(self):
        wrong = ReviewOpinions(overall_sentiment="negative", opinions=[
            Opinion(aspect="battery", sentiment="negative",
                    evidence=EvidenceSpan(quote="not in source", start=0, end=13))
        ])
        ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64),
                                        llm=FakeLLM(lambda _: wrong))
        try:
            with self.assertRaises(ValueError):
                ext.extract(TEXT)
        finally:
            ext.close()

    def test_model_failure_releases_slot(self):
        failure = FakeLLM(lambda _: (_ for _ in ()).throw(RuntimeError("inference failed")))
        ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64), llm=failure)
        try:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    ext.extract(TEXT)
        finally:
            ext.close()

    def test_busy_during_inflight_request(self):
        entered, release = threading.Event(), threading.Event()
        def slow(_):
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return fake_output()
        ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64), llm=FakeLLM(slow))
        errors = []
        def work():
            try:
                ext.extract(TEXT)
            except Exception as exc:
                errors.append(exc)
        th = threading.Thread(target=work)
        th.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaises(OpinionBusy):
                ext.extract(TEXT)
        finally:
            release.set()
            th.join(timeout=3)
            ext.close()
        self.assertFalse(errors, errors)
        self.assertFalse(th.is_alive())

    def test_async_call(self):
        async def go():
            ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64), llm=FakeLLM())
            try:
                value = await ext.aextract(TEXT)
                self.assertEqual(value.opinions[0].sentiment, "negative")
            finally:
                await ext.aclose()
        asyncio.run(go())

    def test_async_cancellation_does_not_leak_slot(self):
        entered, finish = threading.Event(), threading.Event()
        def slow(_):
            entered.set()
            if not finish.wait(timeout=3):
                raise RuntimeError("Test worker timed out")
            return fake_output()
        async def go():
            ext = LangChainOpinionExtractor(OpinionConfig("qwen3.5:4b", SHA64), llm=FakeLLM(slow))
            try:
                task = asyncio.create_task(ext.aextract(TEXT))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                with self.assertRaises(OpinionBusy):
                    ext.extract(TEXT)
            finally:
                finish.set()
                await ext.aclose()
        asyncio.run(go())


class FactoryTests(unittest.TestCase):
    def test_ollama_overrides_are_forwarded(self):
        calls = []
        mod = types.ModuleType("langchain_ollama")
        mod.ChatOllama = lambda **kw: (calls.append(kw), kw)[1]
        with patch.dict(sys.modules, {"langchain_ollama": mod}):
            value = get_llm(provider_override="ollama", model_override="qwen3.5:4b",
                            context_tokens=8192, max_output_tokens=2048,
                            timeout_s=77, ollama_base_url="http://127.0.0.1:11434")
        self.assertEqual(value["model"], "qwen3.5:4b")
        self.assertEqual(value["client_kwargs"]["timeout"], 77)
        self.assertEqual(value["num_ctx"], 8192)

    def test_paid_api_rejected_by_factory(self):
        with self.assertRaises(PermissionError):
            get_llm(provider_override="openai", model_override="some-model")

    def test_vllm_insecure_url_rejected(self):
        with patch.dict(os.environ, {"VLLM_BASE_URL": "http://example.org/v1",
                                         "VLLM_API_KEY": "x"}):
            with self.assertRaises(ValueError):
                get_llm(provider_override="vllm")

    def test_vllm_private_endpoint(self):
        mod = types.ModuleType("langchain_openai")
        mod.ChatOpenAI = lambda **kw: kw
        with patch.dict(sys.modules, {"langchain_openai": mod}), patch.dict(os.environ, {
            "VLLM_BASE_URL": "https://internal.example.com/v1", "VLLM_API_KEY": "test-key"
        }):
            data = get_llm(provider_override="vllm", model_override="Qwen/Qwen3.5-4B")
        self.assertEqual(data["base_url"], "https://internal.example.com/v1")


if __name__ == "__main__":
    unittest.main()
