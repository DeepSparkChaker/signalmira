"""Offline integration tests; no LLM weights, GPU or DB needed."""
import asyncio
import unittest
import sys
import types
try:
    import security.tenant_context
except ModuleNotFoundError:
    security = types.ModuleType("security")
    context = types.ModuleType("security.tenant_context")
    context.validate_tenant = lambda tenant: tenant if tenant and tenant.isidentifier() else (_ for _ in ()).throw(ValueError("tenant"))
    security.tenant_context = context
    sys.modules["security"] = security
    sys.modules["security.tenant_context"] = context
from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor, OpinionBusy
from enrichment.opinion_service import analyze_review, analyze_many
from enrichment.opinion_schema import ReviewOpinions

TEXT = 'La caméra est excellente, mais la batterie est mauvaise.'

class FakeStructured:
    def invoke(self, messages):
        text = messages[-1][1].split('untrusted data):\n', 1)[1]
        quote = 'la batterie est mauvaise' if 'la batterie est mauvaise' in text else text
        return {"overall_sentiment": "mixed", "opinions": [{
            "aspect": "battery", "sentiment": "negative", "evidence": {
                "quote": quote, "start": 0, "end": len(quote),
            }
        }]}

class FakeModel:
    def with_structured_output(self, schema, method):
        assert schema is ReviewOpinions and method == 'json_schema'
        return FakeStructured()

class TestOpinionV2(unittest.TestCase):
    def new(self):
        return LangChainOpinionExtractor(OpinionConfig(model='test:1', model_revision='b'*64), llm=FakeModel())

    def test_unicode_alignment(self):
        worker = self.new()
        try:
            output = worker.extract(TEXT)
            self.assertEqual(output.opinions[0].evidence.start, TEXT.index('la batterie'))
        finally: worker.close()

    def test_service_metadata_and_streaming(self):
        worker = self.new()
        try:
            records = analyze_many(worker, tenant='demo', rows=(
                {'review_id': str(i), 'text': TEXT} for i in range(10)), max_rows=3)
            result = list(records)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0].tenant_id, 'demo')
            self.assertEqual(len(result[0].input_hash), 64)
            self.assertEqual(len(result[0].config_hash), 64)
        finally: worker.close()

    def test_failed_validation_frees_slot(self):
        worker = self.new()
        try:
            with self.assertRaises(ValueError): worker.extract('  ')
            self.assertTrue(worker.extract(TEXT).opinions)
        finally: worker.close()

    def test_async(self):
        worker = self.new()
        try:
            output = asyncio.run(worker.aextract(TEXT))
            self.assertTrue(output.opinions)
        finally: worker.close()

    def test_new_hash_distinct_from_old(self):
        worker = self.new()
        try: self.assertNotEqual(worker.config.config_hash, worker.config._original.config_hash)
        finally: worker.close()

    def test_busy(self):
        worker = self.new()
        from enrichment import langchain_opinions as module
        module._PROCESS_SLOT.acquire()
        try:
            with self.assertRaises(OpinionBusy): worker.extract(TEXT)
        finally:
            module._PROCESS_SLOT.release()
            worker.close()

if __name__ == '__main__': unittest.main()
