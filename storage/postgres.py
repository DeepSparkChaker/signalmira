"""Bounded-memory PostgreSQL ingestion: COPY to staging, then atomic upsert."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from itertools import batched

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ingestion.schemas import FeedbackRecord


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback_reviews (
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    review_id TEXT NOT NULL,
    text TEXT NOT NULL,
    company TEXT,
    product_id TEXT,
    review_date DATE,
    url TEXT,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, source, review_id)
)
"""

# Session-local: automatically emptied after every committed batch.
CREATE_STAGE = """
CREATE TEMP TABLE IF NOT EXISTS feedback_ingest_stage (
    tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
    text TEXT NOT NULL, company TEXT, product_id TEXT, review_date DATE,
    url TEXT, attributes JSONB NOT NULL, content_hash TEXT NOT NULL
) ON COMMIT DELETE ROWS
"""

COPY_STAGE = """
COPY feedback_ingest_stage (
    tenant_id, source, review_id, text, company, product_id,
    review_date, url, attributes, content_hash
) FROM STDIN
"""

# One set-based statement per batch, not one SQL query per review.
UPSERT_STAGE = """
WITH applied AS (
    INSERT INTO feedback_reviews AS f (
        tenant_id, source, review_id, text, company, product_id,
        review_date, url, attributes, content_hash
    )
    SELECT tenant_id, source, review_id, text, company, product_id,
           review_date, url, attributes, content_hash
    FROM feedback_ingest_stage WHERE TRUE
    ON CONFLICT (tenant_id, source, review_id) DO UPDATE SET
        text = EXCLUDED.text,
        company = EXCLUDED.company,
        product_id = EXCLUDED.product_id,
        review_date = EXCLUDED.review_date,
        url = EXCLUDED.url,
        attributes = EXCLUDED.attributes,
        content_hash = EXCLUDED.content_hash,
        updated_at = NOW()
    WHERE f.content_hash IS DISTINCT FROM EXCLUDED.content_hash
    RETURNING 1
)
SELECT COUNT(*) AS n FROM applied
"""


def connect(database_url: str) -> psycopg.Connection:
    """Open one connection; works locally or with a Cloud SQL PostgreSQL DSN."""
    dsn = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def create_tables(conn: psycopg.Connection) -> None:
    """Initialize local schema; production changes should use migrations."""
    with conn.transaction():
        conn.execute(CREATE_TABLE)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS feedback_reviews_date_idx "
            "ON feedback_reviews (tenant_id, review_date)"
        )


def fingerprint(record: FeedbackRecord) -> str:
    """Hash original text and all normalized metadata to detect updates."""
    payload = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def upsert_reviews(
    conn: psycopg.Connection,
    records: Iterable[FeedbackRecord],
    *,
    batch_size: int = 500,
) -> tuple[int, int]:
    """Stream fixed-size batches; return (written, skipped/unchanged).

    Last duplicate identity wins within a batch. A failed batch rolls back;
    earlier committed batches stay available and can be retried safely.
    """
    if not 1 <= batch_size <= 10_000:
        raise ValueError("batch_size must be between 1 and 10000")

    # Create outside the batch transaction so ON COMMIT clears the rows,
    # not the temporary table, after each successful batch.
    conn.execute(CREATE_STAGE)
    written = skipped = 0
    for batch in batched(records, batch_size):
        unique = {
            (r.tenant_id, r.source, r.review_id): r for r in batch
        }
        with conn.transaction():
            with conn.cursor() as cur:
                with cur.copy(COPY_STAGE) as copy:
                    for record in unique.values():
                        copy.write_row((
                            record.tenant_id, record.source, record.review_id,
                            record.text, record.company, record.product_id,
                            record.review_date, record.url, Jsonb(record.attributes),
                            fingerprint(record),
                        ))
                applied = cur.execute(UPSERT_STAGE).fetchone()["n"]
        written += applied
        skipped += len(batch) - applied
    return written, skipped


def get_review(
    conn: psycopg.Connection, *, tenant_id: str, source: str, review_id: str
) -> dict | None:
    """Read every persisted column, including original text and JSONB metadata."""
    return conn.execute(
        "SELECT * FROM feedback_reviews "
        "WHERE tenant_id = %s AND source = %s AND review_id = %s",
        (tenant_id, source, review_id),
    ).fetchone()


def count_reviews(conn: psycopg.Connection, *, tenant_id: str, source: str) -> int:
    """Count one tenant/source without mixing organizations."""
    return conn.execute(
        "SELECT COUNT(*) AS n FROM feedback_reviews "
        "WHERE tenant_id = %s AND source = %s",
        (tenant_id, source),
    ).fetchone()["n"]
