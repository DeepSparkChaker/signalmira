"""Incremental tenant-scoped NER using ONE multilingual model by default.

Kept as a compatible CLI entry point; language detection runs independently via
`python -m enrichment.language sync --tenant TENANT` for observability only.
Use --profile accurate *only* for explicitly verified English-only workflows.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from enrichment.auto_nlp import _ner_engine
from security.tenant_context import tenant_transaction, validate_tenant

QUERY = """
SELECT f.source, f.review_id, f.text, f.content_hash,
       n.content_hash AS ner_hash, n.config_hash AS ner_config
FROM public.feedback_reviews f
LEFT JOIN public.feedback_ner_state n
  ON n.tenant_id=f.tenant_id AND n.source=f.source AND n.review_id=f.review_id
WHERE f.tenant_id=%s AND (f.source,f.review_id)>(%s,%s)
ORDER BY f.source, f.review_id
LIMIT %s
"""


def sync_auto(conn: Any, tenant: str, *, page_size: int = 128,
              limit: int = 0, device: str = "auto", offline: bool = True,
              profile: str = "multilingual", ner_factory=None) -> dict:
    """Process changed reviews; preserve atomic source-hash validation on writes.

    One inference checkpoint per worker/run. Never use tenant majority or
    language confidence to route individual reviews. Caller must authorize tenant
    and provide an autocommit=True database connection. Model weights are shared;
    source records and results always retain their tenant identity.
    """
    tenant = validate_tenant(tenant)
    if profile not in {"multilingual", "accurate"}:
        raise ValueError("profile must be multilingual or accurate")
    if not 1 <= page_size <= 1000 or limit < 0:
        raise ValueError("Bad page_size or limit")
    from enrichment.ner import _write_batch
    ner_factory = ner_factory or (lambda name: _ner_engine(name, device, offline))
    conn.execute("""CREATE TEMP TABLE IF NOT EXISTS ner_stage (
        tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
        content_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
        model_revision TEXT NOT NULL, mentions JSONB NOT NULL,
        PRIMARY KEY (tenant_id,source,review_id)
    ) ON COMMIT DELETE ROWS""")
    engine = None
    signature = None
    processed = skipped = entities = scanned = 0
    after = ("", "")
    while True:
        fetch = min(page_size, limit - scanned) if limit else page_size
        if fetch <= 0:
            break
        with tenant_transaction(conn, tenant):
            rows = conn.execute(QUERY, (tenant, *after, fetch)).fetchall()
        if not rows:
            break
        after = (rows[-1]["source"], rows[-1]["review_id"])
        scanned += len(rows)
        if engine is None:
            engine = ner_factory(profile)
            signature = engine.config_hash  # resolved/pinned model configuration
        pending = [r for r in rows if r["ner_hash"] != r["content_hash"]
                   or r["ner_config"] != signature]
        skipped += len(rows) - len(pending)
        if pending:
            evidence = list(zip(
                pending, engine.extract_many(r["text"] for r in pending), strict=True
            ))
            with tenant_transaction(conn, tenant):
                n, e = _write_batch(conn, tenant, engine, evidence)
            processed += n
            entities += e
    return {"scanned": scanned, "reviews_processed": processed,
            "unchanged": skipped, "entities": entities,
            "profiles_loaded": [profile] if engine is not None else []}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--profile", choices=("multilingual", "accurate"),
                        default="multilingual")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    from storage.postgres import connect
    with connect(os.environ["DATABASE_URL"]) as conn:
        print(json.dumps(sync_auto(conn, args.tenant,
              page_size=args.page_size, limit=args.limit,
              device=args.device, offline=args.offline,
              profile=args.profile), indent=2))


if __name__ == "__main__":
    main()
