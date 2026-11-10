"""No database or model required. Run: uv run python -m unittest discover -s tests -p 'test_hybrid_sql.py'"""
import unittest
from unittest.mock import patch

from retrieval.hybrid import build_sql, build_params
from retrieval.lexical import _sql as lexical_sql, check_backend


class SQLSafetyTests(unittest.TestCase):
    def test_opt_in_required(self):
        with self.assertRaises(PermissionError):
            check_backend('bm25')
        check_backend('bm25', True)
        check_backend('fts')

    def test_sql_is_parameterized(self):
        for backend in ('bm25', 'fts'):
            for mode in ('hnsw', 'exact'):
                for filters in ({}, {'source': "a' OR true --"},
                                {'product_id': "x' OR true --"},
                                {'source': "x'", 'product_id': "y'"}):
                    with self.subTest(backend=backend, mode=mode, filters=filters):
                        sql = build_sql(backend=backend, mode=mode, **filters)
                        with patch('retrieval.hybrid.checked_vector', side_effect=lambda v: v):
                            with patch('retrieval.hybrid.validate_tenant', side_effect=lambda x: x):
                                params = build_params('demo', 'battery problem', object(),
                                                      'f' * 64, backend=backend,
                                                      mode=mode, **filters)
                        self.assertEqual(sql.count('%s'), len(params))
                        self.assertIn('c.tenant_id=%s', sql)
                        self.assertIn('e.tenant_id=%s', sql)
                        self.assertNotIn("OR true --", sql)
                        self.assertNotIn("x' OR true --", sql)
                        self.assertTrue('ORDER BY c.chunk_text <@>' in sql if backend == 'bm25'
                                        else 'websearch_to_tsquery' in sql)

    def test_lexical_placeholder_counts(self):
        for backend in ('bm25', 'fts'):
            for kw in ({}, {'source': 'x'}, {'product_id': 'y'},
                       {'source': 'x', 'product_id': 'y'}):
                sql = lexical_sql(backend, **{'source': kw.get('source'),
                                             'product_id': kw.get('product_id')})
                self.assertEqual(sql.count('%s'), 4 + len(kw))


if __name__ == '__main__':
    unittest.main()
