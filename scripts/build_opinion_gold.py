"""Sample REAL SignalMira reviews into an UNLABELED opinion-evaluation JSONL.

No model calls, no database writes, bounded memory (O(sample size)).
Do not describe these rows as a gold dataset until a human has labeled them.

Examples:
  uv run python -m scripts.build_opinion_gold --tenant demo --count 150
  uv run python -m scripts.build_opinion_gold --csv data/reviews.csv --count 150
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LANGUAGES = ("en", "fr", "es", "ar", "zh")


def _reservoir(pool: list[dict], row: dict, seen: int, cap: int, rng: random.Random) -> None:
    """Maintain a uniform random sample of up to cap from seen rows."""
    if len(pool) < cap:
        pool.append(row)
    else:
        j = rng.randrange(seen)
        if j < cap:
            pool[j] = row


def sample(rows: Iterator[dict], count: int = 150, languages: tuple[str, ...] = LANGUAGES,
           seed: int = 42, max_scanned: int = 0) -> tuple[list[dict], dict]:
    """Prefer equal language quotas, filling shortage with a global reservoir.

    A stable input ordering + fixed seed makes repeat runs reproducible.
    This is *not* a balanced sample when the tenant lacks requested languages.
    """
    if not 1 <= count <= 10000 or not languages or len(set(languages)) != len(languages):
        raise ValueError("count must be 1..10000 and languages unique/nonempty")
    if max_scanned < 0:
        raise ValueError("max_scanned cannot be negative")
    rng = random.Random(seed)
    quotas = {lang: count // len(languages) + (i < count % len(languages))
              for i, lang in enumerate(languages)}
    selected = {lang: [] for lang in languages}
    seen = Counter()
    backup: list[dict] = []
    scanned = 0
    # Sampling the entire stream takes O(number of source records) time, not RAM.
    for row in rows:
        if max_scanned and scanned >= max_scanned:
            break
        scanned += 1
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        lang = (row.get("language") or "und").lower()
        if lang not in languages:
            lang = "und"
        source = str(row.get("source") or "unknown")
        review_id = str(row.get("review_id") or "")
        if not review_id:
            continue
        item = {
            "id": f"{source}:{review_id}", "source": source,
            "review_id": review_id, "tenant_id": row.get("tenant_id") or "demo",
            "language": lang, "text": text,
            "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "gold": None,
        }
        seen[lang] += 1
        _reservoir(backup, item, sum(seen.values()), count * 3, rng)
        if lang in quotas and quotas[lang]:
            _reservoir(selected[lang], item, seen[lang], quotas[lang], rng)
    result: list[dict] = []
    used: set[str] = set()
    for lang in languages:
        for item in selected[lang]:
            if item["id"] not in used:
                result.append(item)
                used.add(item["id"])
    rng.shuffle(backup)
    for item in backup:
        if len(result) >= count:
            break
        if item["id"] not in used:
            result.append(item)
            used.add(item["id"])
    rng.shuffle(result)
    # Freeze a dev/test separation before annotations or model experiments.
    # Keep ~20% per observed language in dev, the remainder as untouched test.
    for lang in set(item["language"] for item in result):
        group = [item for item in result if item["language"] == lang]
        group.sort(key=lambda item: hashlib.sha256(
            f"{seed}:{item['id']}".encode("utf-8")).hexdigest())
        dev_size = max(1, round(len(group) * 0.2)) if len(group) > 4 else 0
        for item in group[:dev_size]:
            item["split"] = "dev"
        for item in group[dev_size:]:
            item["split"] = "test"
    return result[:count], {"scanned": scanned, "selected": len(result[:count]),
                            "source_languages": dict(seen),
                            "sample_languages": dict(Counter(x["language"] for x in result[:count])),
                            "seed": seed, "requested_languages": list(languages)}


def csv_rows(filename: Path) -> Iterator[dict]:
    # No CSV is buffered; caller must supply authentic, licensed reviews.
    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        if not {"review_id", "text"} <= fields:
            raise ValueError("CSV must contain review_id,text; source,language optional")
        for row in reader:
            row["source"] = row.get("source") or "csv"
            yield row


@contextmanager
def postgres_rows(tenant: str, source: str | None = None):
    """Server-side cursor under the existing tenant-scoped transaction."""
    from security.tenant_context import tenant_transaction, validate_tenant
    from storage.postgres import connect
    tenant = validate_tenant(tenant)
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is not set")
    with connect(os.environ["DATABASE_URL"]) as conn:
        with tenant_transaction(conn, tenant):
            # The stored input hash is checked so stale language labels aren't trusted.
            sql = """
                SELECT f.tenant_id, f.source, f.review_id, f.text,
                       COALESCE(NULLIF(l.language, 'und'),
                                NULLIF(f.attributes->>'language',''), 'und') AS language
                FROM public.feedback_reviews AS f
                LEFT JOIN public.feedback_language AS l
                  ON l.tenant_id=f.tenant_id AND l.source=f.source
                 AND l.review_id=f.review_id AND l.input_hash=f.content_hash
                WHERE f.tenant_id=%s AND f.text IS NOT NULL
            """
            params: list[str] = [tenant]
            if source is not None:
                sql += " AND f.source=%s"
                params.append(source)
            sql += " ORDER BY f.source, f.review_id"
            # Named cursor fetches pages rather than materializing all reviews.
            with conn.cursor(name="sm_opinion_gold_sample") as cur:
                cur.execute(sql, params)
                def iterate():
                    while batch := cur.fetchmany(500):
                        for row in batch:
                            yield dict(row)
                yield iterate()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; choose another --out")
    # Exclusive create prevents accidental loss of labeled data.
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tenant", help="Trusted local evaluation tenant")
    src.add_argument("--csv", type=Path, help="Existing real review CSV with review_id,text")
    p.add_argument("--source", help="Optional PostgreSQL source filter")
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--langs", default=",".join(LANGUAGES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scanned", type=int, default=0,
                   help="0 = whole tenant; capping introduces sampling bias")
    p.add_argument("--out", type=Path,
                   default=Path("tests/data/opinion_eval/real_reviews_150.jsonl"))
    args = p.parse_args()
    languages = tuple(x.strip().lower() for x in args.langs.split(",") if x.strip())
    if args.csv:
        if args.source:
            p.error("--source only works with --tenant")
        result, stats = sample(csv_rows(args.csv), args.count, languages, args.seed, args.max_scanned)
    else:
        with postgres_rows(args.tenant, args.source) as rows:
            result, stats = sample(rows, args.count, languages, args.seed, args.max_scanned)
    write_jsonl(args.out, result)
    print(json.dumps({"output": str(args.out), "labels": 0,
                      "human_annotation_required": True, **stats}, indent=2))


if __name__ == "__main__":
    main()
