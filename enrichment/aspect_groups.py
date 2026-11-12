"""Tenant-scoped saved-aspect analytics over EXISTING, validated opinion rows.

No Streamlit, model loading or network inference. Reusable by API and analyst tools.
A trusted caller authenticates the tenant before calling; RLS is defense in depth.
"""
from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from security.tenant_context import tenant_transaction, validate_tenant

MAX_ASPECTS = 50
MAX_RESULTS = 100


class GroupConflict(ValueError):
    """Group was changed elsewhere; reload before editing."""


def _key(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
        raise ValueError('extractor_key must be a 64-character SHA256 hex string')
    return value.lower()


def _aspects(values: list[str] | tuple[str, ...]) -> list[str]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_ASPECTS:
        raise ValueError('Select 1..50 existing aspects')
    if any(not isinstance(a, str) or not a.strip() or len(a) > 100 or a != a.strip() for a in values):
        raise ValueError('Aspect names must be nonempty, exact stored labels (max 100 characters)')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate aspect labels')
    return sorted(values)


def _name(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 120:
        raise ValueError('Group name must contain 1..120 characters')
    return value.strip()


def _id(value: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError('Invalid saved group ID') from exc


def _rows(conn, query: str, params: tuple) -> list[dict]:
    return [dict(r) for r in conn.execute(query, params).fetchall()]


def list_extractor_runs(conn, tenant: str) -> list[dict]:
    """Only model/prompt versions attached to CURRENT review content are selectable."""
    tenant = validate_tenant(tenant)
    with tenant_transaction(conn, tenant):
        return _rows(conn, """
            SELECT r.extractor_key, r.model_name, r.model_revision,
                   COUNT(*) AS review_count
            FROM public.feedback_opinion_runs r
            JOIN public.feedback_reviews f
              ON f.tenant_id=r.tenant_id AND f.source=r.source
             AND f.review_id=r.review_id AND f.content_hash=r.input_hash
            WHERE r.tenant_id=%s
            GROUP BY r.extractor_key, r.model_name, r.model_revision
            ORDER BY review_count DESC, r.extractor_key
            LIMIT 30
        """, (tenant,))


def list_aspects(conn, tenant: str, extractor_key: str, search: str = '') -> list[dict]:
    """Exact raw names and distinct CURRENT reviews, no alias guessing."""
    tenant, extractor_key = validate_tenant(tenant), _key(extractor_key)
    if not isinstance(search, str) or len(search) > 80:
        raise ValueError('Aspect search must be at most 80 characters')
    with tenant_transaction(conn, tenant):
        return _rows(conn, """
            SELECT o.aspect,
                   COUNT(DISTINCT (o.source, o.review_id)) AS review_count
            FROM public.feedback_opinions o
            JOIN public.feedback_reviews f
              ON f.tenant_id=o.tenant_id AND f.source=o.source
             AND f.review_id=o.review_id AND f.content_hash=o.input_hash
            WHERE o.tenant_id=%s AND o.extractor_key=%s
              AND o.aspect ILIKE ('%%' || %s || '%%')
            GROUP BY o.aspect
            ORDER BY review_count DESC, o.aspect
            LIMIT 500
        """, (tenant, extractor_key, search))


def list_groups(conn, tenant: str, extractor_key: str) -> list[dict]:
    tenant, extractor_key = validate_tenant(tenant), _key(extractor_key)
    with tenant_transaction(conn, tenant):
        return _rows(conn, """
            SELECT group_id, name, extractor_key, aspects, version, updated_at
            FROM public.sm_aspect_groups
            WHERE tenant_id=%s AND extractor_key=%s
            ORDER BY lower(name), group_id
            LIMIT 200
        """, (tenant, extractor_key))


def get_group(conn, tenant: str, group_id: str) -> dict:
    tenant, group_id = validate_tenant(tenant), _id(group_id)
    with tenant_transaction(conn, tenant):
        row = conn.execute("""
            SELECT group_id, name, extractor_key, aspects, version
            FROM public.sm_aspect_groups
            WHERE tenant_id=%s AND group_id=%s
        """, (tenant, group_id)).fetchone()
    if row is None:
        raise KeyError('Saved aspect group not found for tenant')
    return dict(row)


def _ensure_existing(conn, tenant: str, extractor_key: str, aspects: list[str]) -> None:
    """Prevent saving names that are not in this tenant's current extraction output."""
    found = _rows(conn, """
        SELECT DISTINCT o.aspect
        FROM public.feedback_opinions o
        JOIN public.feedback_reviews f
          ON f.tenant_id=o.tenant_id AND f.source=o.source
         AND f.review_id=o.review_id AND f.content_hash=o.input_hash
        WHERE o.tenant_id=%s AND o.extractor_key=%s
          AND o.aspect=ANY(%s::text[])
    """, (tenant, extractor_key, aspects))
    missing = set(aspects) - {r['aspect'] for r in found}
    if missing:
        raise ValueError('Unknown or stale aspect names: ' + ', '.join(sorted(missing)))


def create_group(conn, tenant: str, extractor_key: str, name: str,
                 aspects: list[str] | tuple[str, ...]) -> dict:
    """Save a named, immutable v1 membership snapshot."""
    tenant, extractor_key, name, aspects = (validate_tenant(tenant), _key(extractor_key),
                                            _name(name), _aspects(aspects))
    with tenant_transaction(conn, tenant):
        _ensure_existing(conn, tenant, extractor_key, aspects)
        row = conn.execute("""
            INSERT INTO public.sm_aspect_groups
                (tenant_id, extractor_key, name, aspects)
            VALUES (%s,%s,%s,%s::text[])
            RETURNING group_id, name, extractor_key, aspects, version
        """, (tenant, extractor_key, name, aspects)).fetchone()
        conn.execute("""
            INSERT INTO public.sm_aspect_group_versions
                (tenant_id, group_id, version, name, aspects)
            VALUES (%s,%s,%s,%s,%s::text[])
        """, (tenant, row['group_id'], row['version'], row['name'], aspects))
        return dict(row)


def update_group(conn, tenant: str, group_id: str, expected_version: int,
                 aspects: list[str] | tuple[str, ...]) -> dict:
    """Optimistic locking prevents silent overwrites between UI sessions."""
    tenant, group_id, aspects = validate_tenant(tenant), _id(group_id), _aspects(aspects)
    if type(expected_version) is not int or expected_version < 1:
        raise ValueError('Expected group version must be positive integer')
    with tenant_transaction(conn, tenant):
        row = conn.execute("""
            SELECT extractor_key FROM public.sm_aspect_groups
            WHERE tenant_id=%s AND group_id=%s
            FOR UPDATE
        """, (tenant, group_id)).fetchone()
        if row is None:
            raise KeyError('Saved aspect group not found')
        _ensure_existing(conn, tenant, row['extractor_key'], aspects)
        changed = conn.execute("""
            UPDATE public.sm_aspect_groups
            SET aspects=%s::text[], version=version+1, updated_at=clock_timestamp()
            WHERE tenant_id=%s AND group_id=%s AND version=%s
            RETURNING group_id, name, extractor_key, aspects, version
        """, (aspects, tenant, group_id, expected_version)).fetchone()
        if changed is None:
            raise GroupConflict('Group changed since it was opened. Reload before saving.')
        conn.execute("""
            INSERT INTO public.sm_aspect_group_versions
                (tenant_id, group_id, version, name, aspects)
            VALUES (%s,%s,%s,%s,%s::text[])
        """, (tenant, group_id, changed['version'], changed['name'], aspects))
        return dict(changed)


def _where(source: str | None, product_id: str | None,
           date_from: date | None, date_to: date | None) -> tuple[str, list[Any]]:
    clauses, values = [], []
    for val, column in ((source, 'f.source'), (product_id, 'f.product_id')):
        if val is not None:
            if not isinstance(val, str) or not 1 <= len(val) <= 256:
                raise ValueError('Invalid source or product filter')
            clauses.append('AND ' + column + '=%s')
            values.append(val)
    for val, operator in ((date_from, '>='), (date_to, '<=')):
        if val is not None:
            if not isinstance(val, date):
                raise ValueError('Date filter must be a date')
            clauses.append('AND f.review_date::date ' + operator + ' %s')
            values.append(val)
    if date_from and date_to and date_from > date_to:
        raise ValueError('Start date follows end date')
    return '\n'.join(clauses), values


def find_reviews(conn, tenant: str, extractor_key: str, aspects: list[str], *,
                 source: str | None = None, product_id: str | None = None,
                 date_from: date | None = None, date_to: date | None = None,
                 negative_only: bool = False, limit: int = 20) -> list[dict]:
    """Return at most N distinct reviews, one row per matching opinion (<=16/review)."""
    tenant, extractor_key, aspects = validate_tenant(tenant), _key(extractor_key), _aspects(aspects)
    if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
        raise ValueError('limit must be 1..100')
    filters, filter_values = _where(source, product_id, date_from, date_to)
    polarity = "AND o.sentiment='negative'" if negative_only else ''
    query = f"""
        WITH chosen AS MATERIALIZED (
            SELECT f.tenant_id, f.source, f.review_id,
                   f.review_date, f.text, f.product_id
            FROM public.feedback_reviews f
            WHERE f.tenant_id=%s {filters}
              AND EXISTS (
                SELECT 1 FROM public.feedback_opinions o
                WHERE o.tenant_id=f.tenant_id AND o.source=f.source
                  AND o.review_id=f.review_id AND o.input_hash=f.content_hash
                  AND o.extractor_key=%s AND o.aspect=ANY(%s::text[])
                  {polarity}
              )
            ORDER BY f.review_date DESC NULLS LAST, f.source, f.review_id
            LIMIT %s
        )
        SELECT c.source, c.review_id, c.review_date, c.product_id, c.text,
               o.opinion_index, o.aspect, o.target, o.sentiment,
               o.evidence_quote, o.reported_problem, o.suggestion_quote
        FROM chosen c
        JOIN public.feedback_reviews f
          ON f.tenant_id=c.tenant_id AND f.source=c.source AND f.review_id=c.review_id
        JOIN public.feedback_opinions o
          ON o.tenant_id=c.tenant_id AND o.source=c.source
         AND o.review_id=c.review_id AND o.input_hash=f.content_hash
        WHERE o.extractor_key=%s AND o.aspect=ANY(%s::text[]) {polarity}
        ORDER BY c.review_date DESC NULLS LAST, c.source, c.review_id, o.opinion_index
    """
    params = (tenant, *filter_values, extractor_key, aspects, limit, extractor_key, aspects)
    with tenant_transaction(conn, tenant):
        conn.execute("SET LOCAL TIME ZONE 'UTC'")
        return _rows(conn, query, params)


def weekly_trend(conn, tenant: str, extractor_key: str, aspects: list[str], *,
                 source: str | None = None, product_id: str | None = None,
                 date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    """Distinct-review weekly numbers, not duplicated per opinion. Excludes undated reviews.

    denominator: reviews mentioning ANY chosen aspect that week.
    negative: at least one negative chosen-aspect opinion, may overlap positive.
    """
    tenant, extractor_key, aspects = validate_tenant(tenant), _key(extractor_key), _aspects(aspects)
    filters, filter_values = _where(source, product_id, date_from, date_to)
    query = f"""
        WITH per_review AS (
            SELECT date_trunc('week', f.review_date::timestamp)::date AS week,
                   f.source, f.review_id,
                   bool_or(o.sentiment='negative') AS negative,
                   bool_or(o.sentiment='positive') AS positive,
                   bool_or(o.sentiment='mixed') AS mixed
            FROM public.feedback_opinions o
            JOIN public.feedback_reviews f
              ON f.tenant_id=o.tenant_id AND f.source=o.source
             AND f.review_id=o.review_id AND f.content_hash=o.input_hash
            WHERE o.tenant_id=%s AND o.extractor_key=%s
              AND o.aspect=ANY(%s::text[]) AND f.review_date IS NOT NULL
              {filters}
            GROUP BY week, f.source, f.review_id
        )
        SELECT week, COUNT(*) AS review_count,
               COUNT(*) FILTER (WHERE negative) AS negative_reviews,
               COUNT(*) FILTER (WHERE positive) AS positive_reviews,
               COUNT(*) FILTER (WHERE mixed) AS mixed_reviews,
               ROUND(100.0 * COUNT(*) FILTER (WHERE negative) / NULLIF(COUNT(*),0), 2)
                   AS negative_pct
        FROM per_review
        GROUP BY week ORDER BY week
        LIMIT 520
    """
    with tenant_transaction(conn, tenant):
        conn.execute("SET LOCAL TIME ZONE 'UTC'")
        return _rows(conn, query, (tenant, extractor_key, aspects, *filter_values))


def saved_review(conn, tenant: str, source: str, review_id: str,
                 extractor_key: str | None = None) -> dict | None:
    """Inspect ONE already stored review, and its latest matching CURRENT opinions."""
    tenant = validate_tenant(tenant)
    if any(not isinstance(s, str) or not 1 <= len(s) <= 256 for s in (source, review_id)):
        raise ValueError('Provide a source and review_id, each 1..256 characters')
    if extractor_key is not None:
        extractor_key = _key(extractor_key)
    with tenant_transaction(conn, tenant):
        record = conn.execute("""
            SELECT f.source, f.review_id, f.text, f.review_date, f.product_id,
                   f.content_hash
            FROM public.feedback_reviews f
            WHERE f.tenant_id=%s AND f.source=%s AND f.review_id=%s
        """, (tenant, source, review_id)).fetchone()
        if record is None:
            return None
        ops = _rows(conn, """
            SELECT o.aspect, o.sentiment, o.evidence_quote,
                   o.reported_problem, o.suggestion_quote,
                   o.extractor_key, r.model_name, r.model_revision
            FROM public.feedback_opinions o
            JOIN public.feedback_opinion_runs r
              ON r.tenant_id=o.tenant_id AND r.source=o.source
             AND r.review_id=o.review_id AND r.input_hash=o.input_hash
             AND r.extractor_key=o.extractor_key
            WHERE o.tenant_id=%s AND o.source=%s AND o.review_id=%s
              AND o.input_hash=%s
              AND (%s::text IS NULL OR o.extractor_key=%s)
            ORDER BY o.extractor_key, o.opinion_index
            LIMIT 64
        """, (tenant, source, review_id, record['content_hash'], extractor_key, extractor_key))
    return {**dict(record), 'opinions': ops}


def analyze_saved_group(conn, tenant: str, group_id: str, *,
                        source: str | None = None, product_id: str | None = None,
                        date_from: date | None = None, date_to: date | None = None,
                        negative_only: bool = False, limit: int = 20) -> dict:
    """Agent-ready READ tool: tenant from authenticated context, never LLM text."""
    group = get_group(conn, tenant, group_id)
    kw = dict(source=source, product_id=product_id,
              date_from=date_from, date_to=date_to)
    return {
        'group': group,
        'weekly': weekly_trend(conn, tenant, group['extractor_key'], group['aspects'], **kw),
        'reviews': find_reviews(conn, tenant, group['extractor_key'], group['aspects'],
                                negative_only=negative_only, limit=limit, **kw),
        'definition': 'weekly negative_pct = distinct reviews with a negative selected-aspect opinion / distinct reviews mentioning selected aspects; UTC weeks; excludes undated',
    }
