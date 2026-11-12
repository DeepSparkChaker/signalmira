"""CPU-only unit checks; no Ollama, Streamlit, Docker or database required."""
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

# The ZIP can be tested standalone before installation. In SignalMira the
# actual tenant_context module is imported instead of this temporary stub.
try:
    import security.tenant_context  # noqa: F401
except ModuleNotFoundError:
    security = types.ModuleType('security')
    security.__path__ = []
    tenant_context = types.ModuleType('security.tenant_context')
    tenant_context.validate_tenant = lambda tenant: tenant if tenant else (_ for _ in ()).throw(ValueError('tenant'))
    tenant_context.tenant_transaction = lambda conn, tenant: None
    security.tenant_context = tenant_context
    sys.modules['security'] = security
    sys.modules['security.tenant_context'] = tenant_context

from enrichment import aspect_groups as ag

K = 'a' * 64


@contextmanager
def fake_transaction(conn, tenant):
    conn.scope.append(tenant)
    try:
        yield
    finally:
        conn.scope.pop()


class Result:
    def __init__(self, one=None, many=None):
        self.one = one
        self.many = many or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class Conn:
    def __init__(self, *, one=None, many=None):
        self.queries = []
        self.scope = []
        self.one, self.many = one, many

    def execute(self, sql, params=()):
        self.queries.append((sql, params, tuple(self.scope)))
        return Result(self.one, self.many)


def test_validation():
    assert ag._aspects(['battery_duration', 'battery_life']) == ['battery_duration', 'battery_life']
    for invalid in ([], ['battery']*2, [' '], ['foo\n'*40], ['battery']*51):
        with pytest.raises(ValueError):
            ag._aspects(invalid)
    with pytest.raises(ValueError):
        ag._key('short')
    with pytest.raises(ValueError):
        ag._id('not-a-uuid')


def test_facets_use_current_review_hash_and_tenant():
    conn = Conn(many=[{'aspect': 'battery_life', 'review_count': 3}])
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        assert ag.list_aspects(conn, 'demo', K, 'battery')[0]['review_count'] == 3
    sql, params, scope = conn.queries[0]
    assert 'f.content_hash=o.input_hash' in sql
    assert 'o.extractor_key=%s' in sql
    assert 'COUNT(DISTINCT (o.source, o.review_id))' in sql
    assert params == ('demo', K, 'battery')
    assert scope == ('demo',)


def test_find_reviews_is_parameterized_and_bounded():
    conn = Conn()
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        ag.find_reviews(conn, 'demo', K, ['battery_life'],
                        source="evil' OR TRUE --", product_id='pixel-9',
                        negative_only=True, limit=3)
    query, params, scope = conn.queries[-1]
    assert 'o.aspect=ANY(%s::text[])' in query
    assert query.count("o.sentiment='negative'") == 2
    assert 'f.content_hash' in query
    assert "evil' OR TRUE --" not in query
    assert params == ('demo', "evil' OR TRUE --", 'pixel-9', K, ['battery_life'], 3, K, ['battery_life'])
    assert scope == ('demo',)
    with pytest.raises(ValueError):
        ag.find_reviews(conn, 'demo', K, ['battery_life'], limit=101)


def test_weekly_counts_distinct_reviews_and_not_opinion_rows():
    conn = Conn()
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        ag.weekly_trend(conn, 'demo', K, ['battery_life'])
    text = '\n'.join(q[0] for q in conn.queries)
    assert 'bool_or' in text
    assert 'GROUP BY week, f.source, f.review_id' in text
    assert 'COUNT(*) FILTER (WHERE negative)' in text
    assert "SET LOCAL TIME ZONE 'UTC'" in text


def test_create_group_writes_a_version_snapshot_in_same_scope():
    conn = Conn()
    def reply(sql, params=()):
        conn.queries.append((sql, params, tuple(conn.scope)))
        if 'SELECT DISTINCT o.aspect' in sql:
            return Result(many=[{'aspect': 'battery_life'}])
        if 'INSERT INTO public.sm_aspect_groups' in sql:
            return Result(one={'group_id': '00000000-0000-0000-0000-000000000001',
                               'version': 1, 'name': 'Battery', 'extractor_key': K,
                               'aspects': ['battery_life']})
        return Result()
    conn.execute = reply
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        result = ag.create_group(conn, 'demo', K, 'Battery', ['battery_life'])
    assert result['version'] == 1
    assert len(conn.queries) == 3
    assert all(scope == ('demo',) for _, _, scope in conn.queries)
    assert 'sm_aspect_group_versions' in conn.queries[-1][0]


def test_unknown_aspects_rejected_before_save():
    conn = Conn(many=[])
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        with pytest.raises(ValueError, match='Unknown or stale'):
            ag.create_group(conn, 'demo', K, 'Battery', ['invented'])
    assert len(conn.queries) == 1


def test_update_conflict_preserves_history():
    conn = Conn(one=None)
    def reply(sql, params=()):
        conn.queries.append((sql, params, tuple(conn.scope)))
        if 'SELECT extractor_key' in sql:
            return Result(one={'extractor_key': K})
        if 'SELECT DISTINCT o.aspect' in sql:
            return Result(many=[{'aspect': 'battery_life'}])
        return Result(one=None)
    conn.execute = reply
    with patch.object(ag, 'tenant_transaction', fake_transaction):
        with pytest.raises(ag.GroupConflict):
            ag.update_group(conn, 'demo', '00000000-0000-0000-0000-000000000001',
                            1, ['battery_life'])
    assert not any('INSERT INTO public.sm_aspect_group_versions' in q for q, _, _ in conn.queries)


def test_migration_enforces_rls_and_immutable_history():
    sql = (Path(__file__).resolve().parents[1] / 'migrations/021_aspect_groups.sql').read_text()
    assert sql.count('FORCE ROW LEVEL SECURITY') == 2
    assert 'sm_aspect_group_versions' in sql
    assert "current_setting('app.tenant_id', true)" in sql
    assert 'DROP TABLE' not in sql.upper()
