"""Minimal, safe, idempotent integration into existing topic engine.

Makes topic_policy snapshot read fresh detected language (not stale attributes)
and automatically selects a language-appropriate profile on NEW discovery.
Does NOT touch active taxonomy assignments, NED, ingestion or RAG.

Run from project root: python scripts/patch_topic_language.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'enrichment/topic_policy.py'
TOPICS = ROOT / 'enrichment/topics.py'

BEFORE = "coalesce(nullif(left(lower(f.attributes->>'language'), 16),''),'und') AS lang,"
AFTER = "coalesce(l.language, nullif(left(lower(f.attributes->>'language'),16),''),'und') AS lang,"
JOIN_BEFORE = """(s.review_id IS NULL OR s.content_hash IS DISTINCT FROM f.content_hash) AS fresh
        FROM feedback_reviews f
        LEFT JOIN sm_topic_discovery_seen s ON s.tenant_id=f.tenant_id"""
JOIN_AFTER = """(s.review_id IS NULL OR s.content_hash IS DISTINCT FROM f.content_hash) AS fresh
        FROM feedback_reviews f
        LEFT JOIN feedback_language l ON l.tenant_id=f.tenant_id
         AND l.source=f.source AND l.review_id=f.review_id
         AND l.input_hash=f.content_hash
        LEFT JOIN sm_topic_discovery_seen s ON s.tenant_id=f.tenant_id"""
TOP_BEFORE = 'cfg=Settings(profile=args.profile or configured.profile, source=args.source,'
TOP_AFTER = '''# Select once when generating a NEW candidate; never switch active models.
        from enrichment.topic_auto import discovery_profile
        chosen = (args.profile or configured.profile if configured.profile != "accurate"
                  else discovery_profile(conn, args.tenant, args.profile))
        cfg=Settings(profile=chosen, source=args.source,'''


def patch(path: Path, replacements: list[tuple[str, str]]) -> str:
    text = path.read_text(encoding='utf-8')
    original = text
    for before, after in replacements:
        if after in text:
            continue
        if text.count(before) != 1:
            raise RuntimeError(f'{path}: expected exactly one matching anchor, got {text.count(before)}')
        text = text.replace(before, after, 1)
    if text != original:
        backup = path.with_suffix(path.suffix + '.pre_language.bak')
        if not backup.exists():
            backup.write_text(original, encoding='utf-8')
        path.write_text(text, encoding='utf-8')
        return 'patched'
    return 'already patched'


def main():
    if not POLICY.is_file() or not TOPICS.is_file():
        raise SystemExit('Run this script from a SignalMira project containing enrichment/topics.py and topic_policy.py')
    # Validate all anchors BEFORE writing either file.
    for path, replacements in [(POLICY, [(BEFORE, AFTER), (JOIN_BEFORE, JOIN_AFTER)]),
                               (TOPICS, [(TOP_BEFORE, TOP_AFTER)])]:
        text = path.read_text(encoding='utf-8')
        for before, after in replacements:
            if after not in text and text.count(before) != 1:
                raise RuntimeError(f'{path}: integration anchor not found; source differs')
    print('topic_policy:', patch(POLICY, [(BEFORE, AFTER), (JOIN_BEFORE, JOIN_AFTER)]))
    print('topics:', patch(TOPICS, [(TOP_BEFORE, TOP_AFTER)]))


if __name__ == '__main__':
    main()
