"""SignalMira NER: pinned offline-capable extraction and incremental PostgreSQL sync.

Read-only on feedback_reviews; writes feedback_ner_state and feedback_entities.
Examples:
  python -m enrichment.ner --init
  python -m enrichment.ner --text 'Apple Pixel 9 battery is great' --profile accurate
  python -m enrichment.ner --tenant demo --profile accurate --offline --batch-size 8

The caller is responsible for authenticating the tenant, and for persisting uploaded
feedback with ingestion/storage before scheduling the normal database sync.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Iterable, Iterator

MODELS = {
    "fast": "en_core_web_sm",
    "balanced": "fastino/gliner2.5-small-v1",
    "accurate": "fastino/gliner2.5-base-v1",
    "multilingual": "fastino/gliner2.5-multi-v1",
}
LABELS = ("company", "brand", "product", "product feature", "person", "organization")
SPACY_LABELS = {
    "ORG": "organization", "PERSON": "person", "PRODUCT": "product",
    "GPE": "location", "LOC": "location", "FAC": "facility",
}
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")

# Compatible with the previous release: data and table names are unchanged.
DDL = (
    """CREATE TABLE IF NOT EXISTS feedback_ner_state (
        tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
        content_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
        processed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id, source, review_id),
        FOREIGN KEY (tenant_id, source, review_id)
            REFERENCES feedback_reviews ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS feedback_entities (
        tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
        mention_index INT NOT NULL, text TEXT NOT NULL, label TEXT NOT NULL,
        start_char INT NOT NULL, end_char INT NOT NULL,
        confidence DOUBLE PRECISION, model_name TEXT NOT NULL, method TEXT NOT NULL,
        PRIMARY KEY (tenant_id, source, review_id, mention_index),
        FOREIGN KEY (tenant_id, source, review_id)
            REFERENCES feedback_ner_state ON DELETE CASCADE
    )""",
    "ALTER TABLE feedback_ner_state ADD COLUMN IF NOT EXISTS model_revision TEXT",
    "CREATE INDEX IF NOT EXISTS feedback_entities_label_idx ON feedback_entities (tenant_id, label)",
    # Supports *optional* entity-to-review prefilter before vector search.
    """CREATE INDEX IF NOT EXISTS feedback_entities_match_idx
       ON feedback_entities (tenant_id, label, lower(text), source, review_id)""",
)


def _device(requested: str) -> str:
    """Select hardware, not the extraction model; fail when CUDA was required."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto/cpu/cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        cuda = torch.cuda.is_available()
    except (ImportError, OSError):
        cuda = False
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but is not available to PyTorch")
    return "cuda" if cuda else "cpu"


def _read_rules(path: str | None) -> list[dict]:
    """Strictly accept plain phrases, never untrusted regex or Python code."""
    if not path:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) > 1000:
        raise ValueError("Rules must be an array of at most 1000 phrases")
    result = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("Each rule must be an object")
        label, phrase = item.get("label"), item.get("pattern")
        if (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z_ ]{1,50}", label)
                or not isinstance(phrase, str) or not 1 <= len(phrase.strip()) <= 120):
            raise ValueError("Each rule needs a short label and nonempty literal pattern")
        result.append({"label": label, "pattern": phrase, "id": f"rule:{i}"})
    return result


@dataclass(frozen=True)
class Config:
    """Stable NER configuration. device can change without changing the chosen model."""
    profile: str = "accurate"
    device: str = "auto"
    labels: tuple[str, ...] = LABELS
    rules_path: str | None = None
    batch_size: int = 8
    threshold: float = 0.5
    revision: str | None = None   # exact 40-character Hub commit SHA
    offline: bool = False
    cpu_threads: int = 0         # 0 = PyTorch default
    compile_model: bool = False  # opt-in only; increases cold-start cost
    attention_backend: str = "standard"  # or disentangled_flash (optional package)
    max_short_chars: int = 1000

    def __post_init__(self):
        if self.profile not in MODELS or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid profile or device")
        if not 1 <= self.batch_size <= 64 or not 0 < self.threshold <= 1:
            raise ValueError("Invalid batch size or threshold")
        if not self.labels or len(self.labels) > 30 or any(
            not isinstance(x, str) or not x.strip() for x in self.labels
        ) or len(set(self.labels)) != len(self.labels):
            raise ValueError("Provide 1..30 distinct nonempty labels")
        if self.rules_path and self.profile != "fast":
            raise ValueError("Custom rules apply only to spaCy fast mode")
        if self.revision is not None and not SHA.fullmatch(self.revision):
            raise ValueError("Revision must be the actual 40-character Hub commit SHA")
        if self.cpu_threads < 0 or not 128 <= self.max_short_chars <= 12000:
            raise ValueError("Invalid cpu_threads or max_short_chars")
        if self.attention_backend not in {"standard", "disentangled_flash"}:
            raise ValueError("Unknown attention backend")
        if self.compile_model and self.attention_backend != "standard":
            raise ValueError("Do not combine compile with disentangled_flash")


class NER:
    """One long-lived NER model per worker; reused for paste, uploads, and DB batches."""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.name = MODELS[config.profile]
        self.device = "cpu" if config.profile == "fast" else _device(config.device)
        self.rules = _read_rules(config.rules_path)
        self._model = None
        self._revision: str | None = None
        self._config_hash: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        cfg = self.config
        if cfg.profile == "fast":
            import spacy
            model = spacy.load(self.name, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
            if self.rules:
                ruler = model.add_pipe("entity_ruler", last=True,
                                       config={"overwrite_ents": True, "phrase_matcher_attr": "LOWER"})
                ruler.add_patterns(self.rules)
            self._revision = f"spacy:{spacy.__version__}:{model.meta['version']}"
        else:
            import torch
            from huggingface_hub import snapshot_download
            from gliner2 import AutoExtractor
            if cfg.cpu_threads:
                torch.set_num_threads(cfg.cpu_threads)
            # Resolve and load one exact immutable Hub snapshot, optionally fully offline.
            path = Path(snapshot_download(
                repo_id=self.name, revision=cfg.revision, local_files_only=cfg.offline
            ))
            if not SHA.fullmatch(path.name):
                raise RuntimeError("Hugging Face snapshot did not resolve to a commit SHA")
            self._revision = path.name.lower()
            kw = {"map_location": self.device, "quantize": self.device == "cuda"}
            if cfg.compile_model:
                kw["compile"] = True
            if cfg.attention_backend != "standard":
                kw["attention_backend"] = cfg.attention_backend
            model = AutoExtractor.from_pretrained(str(path), **kw)
            if hasattr(model, "eval"):
                model.eval()
        self._model = model
        return model

    @property
    def config_hash(self) -> str:
        """Includes the *resolved weights commit*, not a decorative revision string."""
        if self._config_hash is None:
            self._load()
            cfg = self.config
            payload = {
                "model": self.name, "revision": self._revision, "labels": cfg.labels,
                "rules": self.rules, "threshold": cfg.threshold,
                "backend": cfg.attention_backend, "compile": cfg.compile_model,
                "fp16": self.device == "cuda" and cfg.profile != "fast",
                "max_short_chars": cfg.max_short_chars,
                "pipeline_version": "ner2",
            }
            self._config_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        return self._config_hash

    @property
    def revision(self) -> str:
        self._load()
        assert self._revision is not None
        return self._revision

    def _parse(self, text: str, output: dict) -> list[dict]:
        """Reject invalid span offsets; preserve nested mentions for later linking."""
        mentions = []
        for label, spans in output.get("entities", {}).items():
            for span in spans:
                start, end, value = span.get("start"), span.get("end"), span.get("text")
                if not (isinstance(start, int) and isinstance(end, int)
                        and 0 <= start < end <= len(text) and text[start:end] == value):
                    raise ValueError(f"Invalid NER span from {self.name}: {span!r}")
                score = span.get("confidence")
                if score is not None and not 0 <= float(score) <= 1:
                    raise ValueError("Confidence must be in [0,1]")
                mentions.append({
                    "text": value, "label": label, "start": start, "end": end,
                    "confidence": float(score) if score is not None else None,
                    "model": self.name, "method": "gliner",
                })
        return sorted(mentions, key=lambda e: (e["start"], e["end"], e["label"]))

    def extract_many(self, texts: Iterable[str]) -> Iterator[list[dict]]:
        """Lazily process bounded batches, length-sort within batches and restore order."""
        model = self._load()
        cfg = self.config
        if cfg.profile == "fast":
            for doc in model.pipe(texts, batch_size=cfg.batch_size, n_process=1):
                yield [{
                    "text": ent.text,
                    "label": SPACY_LABELS.get(ent.label_, ent.label_.lower()),
                    "start": ent.start_char, "end": ent.end_char,
                    "confidence": None, "model": self.name,
                    "method": "spacy_rule" if ent.ent_id_.startswith("rule:") else "spacy",
                } for ent in doc.ents]
            return

        import torch
        for group in batched(texts, cfg.batch_size):
            if any(not isinstance(text, str) for text in group):
                raise TypeError("Every review must be text")
            results: list[dict | None] = [None] * len(group)
            short = [i for i, text in enumerate(group)
                     if text.strip() and len(text) <= cfg.max_short_chars]
            # Keep a short single example from padding every other review in a batch.
            short.sort(key=lambda i: len(group[i]))
            with torch.inference_mode():
                if short:
                    raw = model.batch_extract_entities(
                        [group[i] for i in short], list(cfg.labels),
                        batch_size=cfg.batch_size, threshold=cfg.threshold,
                        include_spans=True, include_confidence=True,
                    )
                    if len(raw) != len(short):
                        raise RuntimeError("NER batch length does not match input")
                    for i, value in zip(short, raw):
                        results[i] = value
                for i, text in enumerate(group):
                    if results[i] is None:
                        results[i] = (model.extract_entities_long(
                            text, list(cfg.labels), chunk_size=384, chunk_overlap=64,
                            threshold=cfg.threshold, include_spans=True, include_confidence=True,
                        ) if text.strip() else {"entities": {}})
            for text, result in zip(group, results, strict=True):
                assert result is not None
                yield self._parse(text, result)

    def extract(self, text: str) -> list[dict]:
        """Extract one directly pasted review; no database writes."""
        if not isinstance(text, str):
            raise TypeError("Review must be text")
        if not text.strip():
            return []
        return next(self.extract_many([text]))

    def warmup(self, count: int = 3) -> None:
        """Pay GPU kernel startup before accepting normal work."""
        if count < 0 or count > 20:
            raise ValueError("warmup count must be 0..20")
        self._load()
        if count and self.config.profile != "fast":
            for _ in range(count):
                list(self.extract_many([
                    "Apple iPhone battery drains overnight.",
                    "Google Pixel camera is excellent, but the battery runs down quickly.",
                ]))
            if self.device == "cuda":
                import torch
                torch.cuda.synchronize()


def connect():
    """Autocommit when idle; explicit atomic transactions only around batch writes."""
    import psycopg
    from psycopg.rows import dict_row
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def initialize(conn) -> None:
    """Local bootstrap; use versioned migrations and RLS before SaaS deployment."""
    with conn.transaction():
        for statement in DDL:
            conn.execute(statement)


def _write_batch(conn, tenant: str, ner: NER, items: list[tuple[dict, list[dict]]]) -> tuple[int, int]:
    """Stage mentions via COPY; lock originals; atomically replace current results.

    Returns (reviews written, entities written). If source text changed during
    inference, its stale result is deliberately ignored and retried on next run.
    """
    if not items:
        return 0, 0
    with conn.transaction():
        with conn.cursor() as cur:
            # Temp staging was created by sync() on this same session.
            with cur.copy("""COPY ner_stage
                (tenant_id, source, review_id, content_hash, config_hash, model_revision, mentions)
                FROM STDIN""") as copy:
                for review, mentions in items:
                    copy.write_row((
                        tenant, review["source"], review["review_id"],
                        review["content_hash"], ner.config_hash, ner.revision,
                        json.dumps(mentions, ensure_ascii=False),
                    ))

            # Lock the current originals before changing the state or its entities.
            # Ingestion cannot update these rows before this transaction completes.
            valid = cur.execute("""SELECT r.review_id
                FROM feedback_reviews r JOIN ner_stage s
                  ON (r.tenant_id,r.source,r.review_id) =
                     (s.tenant_id,s.source,s.review_id)
                WHERE r.content_hash = s.content_hash
                ORDER BY r.source, r.review_id FOR UPDATE OF r""").fetchall()

            cur.execute("""INSERT INTO feedback_ner_state
                (tenant_id, source, review_id, content_hash, config_hash, model_revision, processed_at)
                SELECT s.tenant_id, s.source, s.review_id, s.content_hash, s.config_hash, s.model_revision, now()
                FROM ner_stage s JOIN feedback_reviews r
                  ON (r.tenant_id,r.source,r.review_id) =
                     (s.tenant_id,s.source,s.review_id)
                 AND r.content_hash = s.content_hash
                ON CONFLICT (tenant_id, source, review_id) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    config_hash = EXCLUDED.config_hash,
                    model_revision = EXCLUDED.model_revision,
                    processed_at = now()""")
            cur.execute("""DELETE FROM feedback_entities e USING ner_stage s, feedback_reviews r
                WHERE (e.tenant_id,e.source,e.review_id) =
                      (s.tenant_id,s.source,s.review_id)
                  AND (r.tenant_id,r.source,r.review_id) =
                      (s.tenant_id,s.source,s.review_id)
                  AND r.content_hash = s.content_hash""")
            cur.execute("""INSERT INTO feedback_entities
                (tenant_id,source,review_id,mention_index,text,label,start_char,
                 end_char,confidence,model_name,method)
                SELECT s.tenant_id, s.source, s.review_id, (m.ordinality - 1)::int,
                       m.value->>'text', m.value->>'label',
                       (m.value->>'start')::int, (m.value->>'end')::int,
                       (m.value->>'confidence')::double precision,
                       m.value->>'model', m.value->>'method'
                  FROM ner_stage s JOIN feedback_reviews r
                    ON (r.tenant_id,r.source,r.review_id) =
                       (s.tenant_id,s.source,s.review_id)
                   AND r.content_hash = s.content_hash
                  CROSS JOIN LATERAL jsonb_array_elements(s.mentions)
                       WITH ORDINALITY AS m(value, ordinality)""")
            written = cur.rowcount
            # ner_stage has ON COMMIT DELETE ROWS, ensuring no stale staging data.
    return len(valid), written


def sync(conn, tenant: str, ner: NER, *, source: str | None = None,
         page_size: int = 128, limit: int = 0, warmup: int = 2) -> tuple[int, int, int]:
    """Tenant-scoped keyset sync; skip unchanged records, including empty NER output."""
    if not tenant or not 1 <= page_size <= 1000 or limit < 0:
        raise ValueError("Invalid tenant, page_size, or limit")
    # Model resolution before the database advisory lock avoids locking while downloading.
    ner.warmup(warmup)
    signature = ner.config_hash
    locked = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s), hashtext(%s)) AS locked",
                          ("signalmira_ner", tenant)).fetchone()["locked"]
    if not locked:
        raise RuntimeError("Another NER job is running for this tenant")
    processed = skipped = entity_count = 0
    last_source = last_id = ""
    try:
        conn.execute("""CREATE TEMP TABLE IF NOT EXISTS ner_stage (
            tenant_id TEXT NOT NULL, source TEXT NOT NULL, review_id TEXT NOT NULL,
            content_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
            model_revision TEXT NOT NULL, mentions JSONB NOT NULL, PRIMARY KEY (tenant_id, source, review_id)
        ) ON COMMIT DELETE ROWS""")
        while True:
            statement = """SELECT source, review_id, text, content_hash FROM feedback_reviews
                           WHERE tenant_id=%s AND (source,review_id)>(%s,%s)"""
            params = [tenant, last_source, last_id]
            if source is not None:
                statement += " AND source=%s"
                params.append(source)
            statement += " ORDER BY source,review_id LIMIT %s"
            params.append(page_size)
            rows = conn.execute(statement, params).fetchall()
            if not rows:
                break
            last_source, last_id = rows[-1]["source"], rows[-1]["review_id"]
            # Bounded index range for just this page; includes first and last keys.
            states = {(s["source"], s["review_id"]): s for s in conn.execute(
                """SELECT source, review_id, content_hash, config_hash
                   FROM feedback_ner_state
                   WHERE tenant_id=%s AND (source, review_id) >= (%s,%s)
                     AND (source, review_id) <= (%s,%s)""",
                (tenant, rows[0]["source"], rows[0]["review_id"], last_source, last_id),
            )}
            pending = [r for r in rows if not (
                (old := states.get((r["source"], r["review_id"])))
                and old["content_hash"] == r["content_hash"]
                and old["config_hash"] == signature
            )]
            skipped += len(rows) - len(pending)
            if limit:
                pending = pending[:max(0, limit - processed)]
            if pending:
                # NER runs in sub-batches (e.g. 8); SQL writes once per page (e.g. 128).
                staged = list(zip(pending,
                    ner.extract_many(r["text"] for r in pending), strict=True))
                written, inserted = _write_batch(conn, tenant, ner, staged)
                processed += written
                entity_count += inserted
            if limit and processed >= limit:
                break
        return processed, skipped, entity_count
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s), hashtext(%s))",
                     ("signalmira_ner", tenant))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="Create NER tables and indexes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tenant", help="Trusted server-side tenant for PostgreSQL sync")
    group.add_argument("--text", help="Paste one review")
    group.add_argument("--stdin", action="store_true", help="Read one review from stdin")
    parser.add_argument("--source", help="Optional database source filter")
    parser.add_argument("--profile", choices=MODELS, default="accurate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--rules-json", help="spaCy literal patterns in a local JSON file")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--revision", help="Actual 40-character Hub commit SHA (recommended)")
    parser.add_argument("--offline", action="store_true", help="No Hub network requests")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Optional torch.compile experiment")
    parser.add_argument("--attention-backend", choices=("standard", "disentangled_flash"),
                        default="standard")
    parser.add_argument("--limit", type=int, default=0, help="0 means all pending reviews")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up iterations for DB jobs")
    args = parser.parse_args()
    if args.init:
        with connect() as conn:
            initialize(conn)
        print("NER tables ready")
        return
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ner = NER(Config(
        profile=args.profile, device=args.device,
        labels=tuple(x.strip() for x in args.labels.split(",")),
        rules_path=args.rules_json, batch_size=args.batch_size, threshold=args.threshold,
        revision=args.revision, offline=args.offline, cpu_threads=args.cpu_threads,
        compile_model=args.compile, attention_backend=args.attention_backend,
    ))
    if args.text is not None or args.stdin:
        import sys
        text = sys.stdin.read() if args.stdin else args.text
        print(json.dumps(ner.extract(text), ensure_ascii=False, indent=2))
        print(f"# model={ner.name} revision={ner.revision} device={ner.device}", file=sys.stderr)
    elif args.tenant:
        with connect() as conn:
            processed, skipped, count = sync(
                conn, args.tenant, ner, source=args.source,
                page_size=args.page_size, limit=args.limit, warmup=args.warmup,
            )
        print(f"Processed: {processed}; unchanged: {skipped}; entities: {count}; "
              f"model: {ner.name}; revision: {ner.revision}")
    else:
        parser.error("Choose --tenant, --text, --stdin, or --init")


if __name__ == "__main__":
    main()
