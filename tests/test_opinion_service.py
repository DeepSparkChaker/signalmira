"""Service tests; use the real security module in the SignalMira project."""
from __future__ import annotations

import asyncio
import sys
import types
import unittest

try:
    from security.tenant_context import validate_tenant  # noqa: F401
except ModuleNotFoundError:
    # Standalone patch CI only; SignalMira's real tenant validator is used in-project.
    security = types.ModuleType("security")
    security.__path__ = []
    context = types.ModuleType("security.tenant_context")
    def validate_tenant(value):
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError("Invalid tenant")
        return value
    context.validate_tenant = validate_tenant
    sys.modules.setdefault("security", security)
    sys.modules.setdefault("security.tenant_context", context)

from enrichment.opinion_service import analyze_review, analyze_many, aanalyze_review
from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor
from enrichment.opinion_schema import ReviewOpinions


class _LLM:
    def with_structured_output(self, schema, *, method):
        assert method == "json_schema"
        return self

    def invoke(self, messages):
        return ReviewOpinions(overall_sentiment="neutral", opinions=[])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.ext = LangChainOpinionExtractor(
            OpinionConfig("qwen3.5:4b", "a" * 64), llm=_LLM())

    def tearDown(self):
        self.ext.close()

    def test_one_review_and_provenance(self):
        answer = analyze_review(self.ext, tenant="demo", review_id="r1", text="Okay.")
        self.assertEqual(answer.review_id, "r1")
        self.assertEqual(answer.model_id, "qwen3.5:4b")
        self.assertEqual(len(answer.input_hash), 64)
        self.assertEqual(answer.as_dict()["data"]["overall_sentiment"], "neutral")

    def test_stream_bounded_review_rows(self):
        rows = ({"review_id": str(i), "text": "Okay."} for i in range(500))
        output = list(analyze_many(self.ext, tenant="demo", rows=rows, max_rows=3))
        self.assertEqual([x.review_id for x in output], ["0", "1", "2"])

    def test_async_service_entrypoint(self):
        async def run():
            result = await aanalyze_review(self.ext, tenant="demo",
                                           review_id="r2", text="Okay.")
            self.assertEqual(result.review_id, "r2")
        asyncio.run(run())

    def test_bad_id_rejected_before_llm(self):
        with self.assertRaises(ValueError):
            analyze_review(self.ext, tenant="demo", review_id="", text="Okay.")


if __name__ == "__main__":
    unittest.main()
