"""Fast offline tests; no checkpoint, database, network or Ollama required.

Run from project root: uv run python -m unittest discover -s tests -p 'test_sentiment_opinions.py' -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# The complete patch is also runnable standalone before being copied into the repo.
_PATCH_ROOT = Path(__file__).resolve().parents[1]
if (_PATCH_ROOT / 'enrichment').exists() and str(_PATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PATCH_ROOT))

from pydantic import ValidationError
from enrichment.opinion_schema import EvidenceSpan, Opinion, ReviewOpinions
from enrichment.opinions import OpinionConfig, OpinionExtractor
from enrichment.sentiment import SentimentConfig

SHA1 = 'a'*40
SHA2 = 'b'*40
TEXT = 'The camera is excellent but the battery drains overnight.'


def sample_output(start_adjust=0):
    quote = 'battery drains overnight'
    start = TEXT.index(quote)
    return {'overall_sentiment': 'mixed', 'opinions': [
        {'aspect': 'battery', 'target': None, 'sentiment': 'negative',
         'evidence': {'quote': quote, 'start': start + start_adjust,
                      'end': start + start_adjust + len(quote)},
         'reported_problem': 'Battery drains overnight',
         'customer_suggestion': None},
    ]}


class EvidenceTests(unittest.TestCase):
    def test_valid_span(self):
        result = ReviewOpinions.model_validate(sample_output()).verify(TEXT)
        self.assertEqual(result.opinions[0].aspect, 'battery')

    def test_wrong_offset_repaired_only_for_unique_exact_quote(self):
        result = ReviewOpinions.model_validate(sample_output(start_adjust=4)).align(TEXT)
        self.assertEqual(result.opinions[0].evidence.start, TEXT.index('battery drains'))

    def test_hallucinated_quote_rejected(self):
        data = sample_output()
        data['opinions'][0]['evidence']['quote'] = 'the battery exploded'
        with self.assertRaises(ValueError):
            ReviewOpinions.model_validate(data).align(TEXT)

    def test_ambiguous_duplicated_quote_rejected(self):
        text = 'bad battery, bad battery'
        doc = ReviewOpinions.model_validate({
            'overall_sentiment':'negative', 'opinions': [
                {'aspect':'battery','sentiment':'negative',
                 'evidence':{'quote':'bad battery','start':2,'end':13}}
            ]})
        with self.assertRaises(ValueError):
            doc.align(text)

    def test_valid_customer_suggestion(self):
        text = 'The device is hot. Please improve the cooling.'
        suggestion = 'Please improve the cooling.'
        quote = 'device is hot'
        doc = ReviewOpinions.model_validate({
            'overall_sentiment':'negative', 'opinions': [{
                'aspect':'temperature', 'sentiment':'negative',
                'evidence':{'quote':quote,'start':text.index(quote),'end':text.index(quote)+len(quote)},
                'customer_suggestion':{'quote':suggestion,'start':text.index(suggestion),
                                      'end':text.index(suggestion)+len(suggestion)}
            }]})
        doc.verify(text)

    def test_fake_suggestion_rejected(self):
        data = sample_output()
        data['opinions'][0]['customer_suggestion'] = {
            'quote':'Replace the phone tomorrow.', 'start':0, 'end':27}
        with self.assertRaises(ValueError):
            ReviewOpinions.model_validate(data).align(TEXT)

    def test_extraneous_fields_rejected(self):
        data = sample_output()
        data['root_cause'] = 'guessed hardware defect'
        with self.assertRaises(ValidationError):
            ReviewOpinions.model_validate(data)

    def test_more_than_16_opinions_rejected(self):
        data = sample_output()
        data['opinions'] = data['opinions'] * 17
        with self.assertRaises(ValidationError):
            ReviewOpinions.model_validate(data)

    def test_unicode_offsets(self):
        text = 'البطارية سيئة جدًا.'
        quote = 'سيئة'
        start = text.index(quote)
        evidence = EvidenceSpan(quote=quote, start=start, end=start+len(quote))
        evidence.verify(text)


class ExtractorTests(unittest.TestCase):
    def test_mock_ollama_response_and_schema(self):
        captured = {}
        def transport(req, url, timeout):
            captured.update(req)
            return {'message': {'content': json.dumps(sample_output(start_adjust=1))}}
        extractor = OpinionExtractor(OpinionConfig(model='qwen3:4b',model_revision='a'*64), transport)
        with patch("enrichment.opinions.verify_ollama_digest") as verify:
            result = extractor.extract(TEXT)
            verify.assert_called_once()
        self.assertEqual(result.opinions[0].evidence.quote, 'battery drains overnight')
        self.assertEqual(captured['format']['type'], 'object')
        self.assertFalse(captured['stream'])
        self.assertEqual(captured['options']['temperature'], 0)

    def test_untrusted_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            OpinionConfig(model='qwen',model_revision='a'*64,url='http://example.com/api/chat')

    def test_bounded_review(self):
        extractor = OpinionExtractor(OpinionConfig(model='qwen3:4b',model_revision='a'*64,max_review_chars=100),
                                    lambda *_: self.fail('Transport must not run'))
        with self.assertRaises(ValueError):
            extractor.extract('x'*101)

    def test_prompt_or_model_revision_changes_hash(self):
        a = OpinionConfig(model='qwen3:4b',model_revision='a'*64)
        b = OpinionConfig(model='qwen3:4b',model_revision='b'*64)
        self.assertNotEqual(a.config_hash,b.config_hash)


class SentimentTests(unittest.TestCase):
    def test_sha_is_required(self):
        with self.assertRaises(ValueError):
            SentimentConfig(revision='latest')

    def test_device_default_cpu(self):
        self.assertEqual(SentimentConfig(revision=SHA1).device,'cpu')

    def test_hash_change_with_version(self):
        a = SentimentConfig(revision=SHA1)
        b = SentimentConfig(revision=SHA2)
        self.assertNotEqual(a.config_hash,b.config_hash)

    def test_batch_limits(self):
        with self.assertRaises(ValueError):
            SentimentConfig(revision=SHA1,batch_size=64)


if __name__ == '__main__':
    unittest.main()
