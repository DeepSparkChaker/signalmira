"""Offline multilingual language enrichment for SignalMira, plus real-time detection.

API:
    detect_text(text)                        -> LanguageResult (no DB, no LLM)
    sync(conn, authorized_tenant)            -> incremental metadata write
    tenant_stats(conn, authorized_tenant)    -> reliable current language summary

CLI:
    python -m enrichment.language init
    python -m enrichment.language detect --text 'Le paiement échoue.'
    python -m enrichment.language sync --tenant demo
    python -m enrichment.language stats --tenant demo

Use autocommit=True connections. Every tenant query uses tenant_transaction();
never accept a raw browser-provided tenant without authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from security.tenant_context import tenant_transaction, validate_tenant

VERSION = "language-v1.0"
# Detect ALL supported languages, not just the first five. Otherwise German or
# Portuguese feedback would be forcibly mislabelled as English/French/Spanish.
DETECTOR = "lingua-language-detector"
MIN_LETTERS = 6
MIN_SCORE = 0.57             # conservative heuristic, NOT calibrated certainty
MIN_MARGIN = 0.12            # top-1 minus top-2, not a probability of correctness
MAX_INPUT_CHARS = 12_000    # inspect bounded sample; original text is untouched
MIGRATION = Path(__file__).resolve().parents[1] / "migrations/003_feedback_language.sql"


@dataclass(frozen=True, slots=True)
class LanguageResult:
    language: str                # ISO 639-1, und (unknown), or mul (mixed scripts)
    status: str                  # detected | unknown | mixed
    confidence: float | None
    margin: float | None
    detector: str
    detector_version: str
    config_hash: str

    def __post_init__(self) -> None:
        if self.status not in ("detected", "unknown", "mixed"):
            raise ValueError("Invalid language status")


def _detector_version() -> str:
    try:
        return importlib.metadata.version(DETECTOR)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install detector: uv add lingua-language-detector") from exc


@lru_cache(maxsize=1)
def _engine():
    """One high-accuracy Rust Lingua engine per worker, lazy and thread-safe."""
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as exc:
        raise RuntimeError("Install detector: uv add lingua-language-detector") from exc
    return LanguageDetectorBuilder.from_all_spoken_languages().build()


def config_hash(version: str | None = None) -> str:
    version = version or _detector_version()
    config = (VERSION, DETECTOR, version, MIN_LETTERS, MIN_SCORE,
              MIN_MARGIN, MAX_INPUT_CHARS, "all-spoken-high-accuracy")
    return hashlib.sha256(repr(config).encode("utf-8")).hexdigest()


def _mixed_scripts(text: str) -> bool:
    """Conservatively identify obvious Latin+Arabic/Han script mixtures.

    Does NOT claim to detect English+Spanish code switching in one paragraph.
    Product names in Latin script may appear inside non-Latin text, so require
    a meaningful amount of writing in both scripts.
    """
    latin = arabic = han = 0
    for char in text:
        name = unicodedata.name(char, "")
        if "LATIN" in name and unicodedata.category(char).startswith("L"):
            latin += 1
        elif "ARABIC" in name and unicodedata.category(char).startswith("L"):
            arabic += 1
        elif "CJK UNIFIED IDEOGRAPH" in name:
            han += 1
    return latin >= 22 and (arabic >= 12 or han >= 12)


def detect_text(text: str, *, engine: Any = None,
                detector_version: str | None = None) -> LanguageResult:
    """Detect from current text, never from tenant majority or source metadata.

    Scores are Lingua-relative confidences, *not* calibrated accuracy odds.
    Unknown is a useful honest result for short/ambiguous feedback.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    version = detector_version or _detector_version()
    key = config_hash(version)
    sampled = text[:MAX_INPUT_CHARS]
    if not sampled.strip() or sum(ch.isalpha() for ch in sampled) < MIN_LETTERS:
        return LanguageResult("und", "unknown", None, None, DETECTOR, version, key)
    if _mixed_scripts(sampled):
        return LanguageResult("mul", "mixed", None, None, DETECTOR, version, key)

    detector = engine if engine is not None else _engine()
    choices = detector.compute_language_confidence_values(sampled)
    if not choices:
        return LanguageResult("und", "unknown", None, None, DETECTOR, version, key)
    first = choices[0]
    second_score = choices[1].value if len(choices) > 1 else 0.0
    score = float(first.value)
    margin = score - float(second_score)
    if score < MIN_SCORE or margin < MIN_MARGIN:
        return LanguageResult("und", "unknown", score, max(0.0, margin), DETECTOR, version, key)
    lang = first.language.iso_code_639_1.name.lower()
    return LanguageResult(lang, "detected", score, max(0.0, margin), DETECTOR, version, key)


def initialize(conn: Any) -> None:
    """Run additive migration with owner/admin connection, not customer credentials."""
    with conn.transaction():
        conn.execute(MIGRATION.read_text(encoding="utf-8"))


PENDING = """
SELECT f.source, f.review_id, f.text, f.content_hash
FROM public.feedback_reviews AS f
LEFT JOIN public.feedback_language AS l
  ON l.tenant_id=f.tenant_id AND l.source=f.source AND l.review_id=f.review_id
WHERE f.tenant_id=%s
  AND (f.source,f.review_id)>(%s,%s)
  AND (l.review_id IS NULL OR l.input_hash IS DISTINCT FROM f.content_hash
       OR l.config_hash IS DISTINCT FROM %s)
ORDER BY f.source,f.review_id
LIMIT %s
"""

UPSERT = """
INSERT INTO public.feedback_language AS old
 (tenant_id, source, review_id, language, status, confidence, margin,
  input_hash, config_hash, detector, detector_version)
SELECT f.tenant_id, r.source, r.review_id, r.language, r.status,
       r.confidence, r.margin, r.input_hash, r.config_hash, r.detector,
       r.detector_version
FROM jsonb_to_recordset(%s::jsonb) AS r(
    source TEXT, review_id TEXT, language TEXT, status TEXT,
    confidence DOUBLE PRECISION, margin DOUBLE PRECISION,
    input_hash TEXT, config_hash TEXT, detector TEXT, detector_version TEXT
)
JOIN public.feedback_reviews AS f
  ON f.tenant_id=%s AND f.source=r.source AND f.review_id=r.review_id
 AND f.content_hash=r.input_hash
ON CONFLICT (tenant_id,source,review_id) DO UPDATE SET
    language=EXCLUDED.language, status=EXCLUDED.status,
    confidence=EXCLUDED.confidence, margin=EXCLUDED.margin,
    input_hash=EXCLUDED.input_hash, config_hash=EXCLUDED.config_hash,
    detector=EXCLUDED.detector, detector_version=EXCLUDED.detector_version,
    detected_at=now()
WHERE old.input_hash IS DISTINCT FROM EXCLUDED.input_hash
   OR old.config_hash IS DISTINCT FROM EXCLUDED.config_hash
RETURNING 1
"""


def sync(conn: Any, tenant: str, *, batch_size: int = 128,
         limit: int = 0, detect_fn=None) -> dict[str, int]:
    """Incrementally enrich; safe with concurrent edits and bounded in memory.

    Each DB operation runs inside a fresh tenant-scoped transaction. Detection
    happens OUTSIDE DB transactions to avoid holding long-lived DB connections.
    Caller must use autocommit=True and supply a previously authorized tenant.
    """
    tenant = validate_tenant(tenant)
    if not 1 <= batch_size <= 1000 or limit < 0:
        raise ValueError("batch_size must be 1..1000; limit must be >=0")
    version = _detector_version()
    key = config_hash(version)
    detect_fn = detect_fn or (lambda txt: detect_text(txt, detector_version=version))
    after = ("", "")
    pending = updated = 0
    while True:
        page_limit = min(batch_size, limit - pending) if limit else batch_size
        if page_limit <= 0:
            break
        with tenant_transaction(conn, tenant):
            rows = conn.execute(PENDING, (tenant, *after, key, page_limit)).fetchall()
        if not rows:
            break
        after = (rows[-1]["source"], rows[-1]["review_id"])
        pending += len(rows)
        payload = []
        for row in rows:
            result = detect_fn(row["text"])
            if result.config_hash != key:
                raise RuntimeError("Detector configuration changed mid-run")
            payload.append({
                "source": row["source"], "review_id": row["review_id"],
                "input_hash": row["content_hash"], **asdict(result),
            })
        # The JOIN requires source hash still equal: stale inference never lands.
        with tenant_transaction(conn, tenant):
            written = conn.execute(UPSERT, (json.dumps(payload, ensure_ascii=False), tenant))
            updated += len(written.fetchall())
        # A concurrent source change is picked up on the next sync run.
    return {"examined": pending, "written": updated,
            "changed_during_detection": pending - updated}


DISTRIBUTION = """
SELECT COALESCE(l.language,'pending') AS language, count(*)::bigint AS n
FROM public.feedback_reviews f
LEFT JOIN public.feedback_language l
 ON l.tenant_id=f.tenant_id AND l.source=f.source AND l.review_id=f.review_id
 AND l.input_hash=f.content_hash AND l.config_hash=%s
WHERE f.tenant_id=%s
GROUP BY COALESCE(l.language,'pending')
ORDER BY n DESC, language
"""


def tenant_stats(conn: Any, tenant: str) -> dict:
    """Read only THIS tenant. Only fresh language results count as detected."""
    tenant = validate_tenant(tenant)
    with tenant_transaction(conn, tenant):
        rows = conn.execute(DISTRIBUTION, (config_hash(), tenant)).fetchall()
    counts = {r["language"]: int(r["n"]) for r in rows}
    total = sum(counts.values())
    en = counts.get("en", 0)
    pending = counts.get("pending", 0)
    unknown = counts.get("und", 0)
    detected = total - pending - unknown
    non_english = detected - en
    # Language is observational metadata, never an automatic model switch.
    # Unknown does NOT mean non-English; detected English does NOT prove the
    # entire tenant is English-only. New topic taxonomies default to BGE-M3.
    mode = ("english_detected_with_uncertainty" if en and not non_english and
            (unknown or pending) else
            "english_detected_only" if en and not non_english else
            "multilingual_or_unverified")
    return {
        "tenant_id": tenant, "total": total, "counts": counts,
        "detected": detected, "english": en,
        "non_english": non_english, "unknown": unknown,
        "pending": pending,
        "topic_profile_recommended": "multilingual_m3",  # fixed default, not inferred
        "tenant_language_mode": mode,
        "routing_provisional": False,
        "unknown_share": round(unknown / total, 4) if total else 0.0,
    }


def _db_conn():
    from storage.postgres import connect
    return connect(os.environ["DATABASE_URL"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    d = sub.add_parser("detect")
    d.add_argument("--text", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--tenant", required=True)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--limit", type=int, default=0)
    st = sub.add_parser("stats")
    st.add_argument("--tenant", required=True)
    args = parser.parse_args()
    if args.action == "detect":
        print(json.dumps(asdict(detect_text(args.text)), ensure_ascii=False, indent=2))
        return
    with _db_conn() as conn:
        if args.action == "init":
            initialize(conn)
            print("feedback_language ready")
        elif args.action == "sync":
            print(json.dumps(sync(conn, args.tenant,
                 batch_size=args.batch_size, limit=args.limit), indent=2))
        else:
            print(json.dumps(tenant_stats(conn, args.tenant), indent=2))


if __name__ == "__main__":
    main()
