"""SignalMira optional topic discovery: reproducible per-tenant BERTopic snapshots.

Once: python enrichment/topics.py --init
Run:  python enrichment/topics.py --tenant demo --profile accurate --max-docs 5000

Not called per review. Discovery is a bounded scheduled/on-demand batch;
keep dashboard taxonomies separate from these evolving exploratory topics.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import date
from uuid import uuid4

EMBEDDERS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "accurate": "BAAI/bge-base-en-v1.5",
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
DDL = (
    """CREATE TABLE IF NOT EXISTS topic_runs (
        tenant_id TEXT NOT NULL, run_id UUID NOT NULL,
        model_name TEXT NOT NULL, sample_count INT NOT NULL,
        start_date DATE, source_filter TEXT, random_seed INT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant_id,run_id)
    )""",
    """CREATE TABLE IF NOT EXISTS topic_labels (
        tenant_id TEXT NOT NULL, run_id UUID NOT NULL, topic_id INT NOT NULL,
        name TEXT NOT NULL, review_count INT NOT NULL,
        PRIMARY KEY (tenant_id,run_id,topic_id),
        FOREIGN KEY (tenant_id,run_id) REFERENCES topic_runs ON DELETE CASCADE
    )""",
    """CREATE TABLE IF NOT EXISTS topic_assignments (
        tenant_id TEXT NOT NULL,run_id UUID NOT NULL,
        source TEXT NOT NULL,review_id TEXT NOT NULL,topic_id INT NOT NULL,
        PRIMARY KEY (tenant_id,run_id,source,review_id),
        FOREIGN KEY (tenant_id,run_id,topic_id)
          REFERENCES topic_labels ON DELETE CASCADE,
        FOREIGN KEY (tenant_id,source,review_id)
          REFERENCES feedback_reviews ON DELETE CASCADE
    )""",
)


def connect():
    import psycopg
    from psycopg.rows import dict_row
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def initialize(conn):
    """Local prototype DDL; replace with migrations for production."""
    with conn.transaction():
        for statement in DDL:
            conn.execute(statement)


def discover(conn, tenant: str, *, profile: str = "accurate", source: str | None = None,
             since: str | None = None, max_docs: int = 5000, min_topic_size: int = 10,
             batch_size: int = 32, seed: int = 42, offline: bool = False,
             save_model: str | None = None) -> dict:
    """Discover topics in one bounded snapshot and persist versioned assignments."""
    if not tenant or profile not in EMBEDDERS:
        raise ValueError("Provide tenant and a valid profile")
    if not 30 <= max_docs <= 50000 or not 2 <= min_topic_size <= max_docs:
        raise ValueError("max_docs 30..50000; min_topic_size 2..max_docs")
    if not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be 1..128")
    start_date = date.fromisoformat(since) if since else None
    lock = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s), hashtext(%s)) AS ok",
                        ("signalmira_topics", tenant)).fetchone()["ok"]
    if not lock:
        raise RuntimeError("Topic discovery already running for this tenant")
    try:
        query = """SELECT source,review_id,text FROM feedback_reviews
            WHERE tenant_id=%s AND length(trim(text)) > 0"""
        params: list = [tenant]
        if source is not None:
            query += " AND source=%s"
            params.append(source)
        if start_date is not None:
            query += " AND review_date >= %s"
            params.append(start_date)
        query += " ORDER BY review_date DESC NULLS LAST, source,review_id LIMIT %s"
        params.append(max_docs)
        rows = conn.execute(query, params).fetchall()
        if len(rows) < max(30, 2 * min_topic_size):
            raise ValueError("Not enough reviews for reliable topic discovery")
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, OSError):
            device = "cpu"
        model_id = EMBEDDERS[profile]
        embedder = SentenceTransformer(model_id, device=device, local_files_only=offline)
        texts = [r["text"] for r in rows]
        embeddings = embedder.encode(texts, batch_size=batch_size,
                                     show_progress_bar=False, convert_to_numpy=True,
                                     normalize_embeddings=True)
        # UMAP randomness is fixed; models/versions and input data must also be versioned.
        reducer = UMAP(n_neighbors=min(15, len(texts)-1), n_components=5,
                       min_dist=0.0, metric="cosine", random_state=seed)
        topic_model = BERTopic(embedding_model=embedder, umap_model=reducer,
                              min_topic_size=min_topic_size,
                              calculate_probabilities=False, low_memory=True,
                              verbose=False)
        topic_ids, _ = topic_model.fit_transform(texts, embeddings)
        counts = Counter(int(x) for x in topic_ids)
        run_id = uuid4()
        labels = {}
        for number, count in sorted(counts.items()):
            words = topic_model.get_topic(number) if number >= 0 else None
            label = ", ".join(word for word, _ in (words or [])[:5]) if number >= 0 else "outlier"
            labels[number] = label or f"topic-{number}"

        # A retrained snapshot gets a new run ID: never silently renumber dashboard labels.
        with conn.transaction():
            conn.execute("""INSERT INTO topic_runs
                (tenant_id,run_id,model_name,sample_count,start_date,source_filter,random_seed)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (tenant, run_id, model_id, len(rows), start_date, source, seed))
            with conn.cursor() as cur:
                cur.executemany("""INSERT INTO topic_labels
                    (tenant_id,run_id,topic_id,name,review_count)
                    VALUES (%s,%s,%s,%s,%s)""", [
                    (tenant, run_id, k, name, counts[k]) for k, name in labels.items()])
                # executemany uses psycopg3 pipeline optimization; memory is bounded by max_docs.
                for i in range(0, len(rows), 250):
                    cur.executemany("""INSERT INTO topic_assignments
                        (tenant_id,run_id,source,review_id,topic_id)
                        VALUES (%s,%s,%s,%s,%s)""", [
                        (tenant, run_id, row["source"], row["review_id"], int(topic_ids[j]))
                        for j, row in enumerate(rows[i:i+250], start=i)])
        if save_model:
            # Safe and compact; save_model path is admin-managed, not derived from tenant.
            topic_model.save(save_model, serialization="safetensors", save_ctfidf=True,
                             save_embedding_model=model_id)
        return {"tenant_id": tenant, "run_id": str(run_id), "model": model_id,
                "device": device, "review_count": len(rows), "topics": [
                    {"id": k, "label": labels[k], "reviews": counts[k]}
                    for k in sorted(labels) ]}
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s), hashtext(%s))",
                     ("signalmira_topics", tenant))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--tenant")
    parser.add_argument("--profile", choices=list(EMBEDDERS), default="accurate")
    parser.add_argument("--source", help="Only this review source")
    parser.add_argument("--since", help="Start date, YYYY-MM-DD")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Only cached model weights")
    parser.add_argument("--save-model", help="Optional admin-controlled directory for safetensors")
    args = parser.parse_args()
    if not args.init and not args.tenant:
        parser.error("--tenant required")
    with connect() as conn:
        if args.init:
            initialize(conn)
            print("Topic discovery tables ready")
        else:
            import json
            print(json.dumps(discover(conn, args.tenant, profile=args.profile,
                source=args.source, since=args.since, max_docs=args.max_docs,
                min_topic_size=args.min_topic_size, batch_size=args.batch_size,
                seed=args.seed, offline=args.offline, save_model=args.save_model),
                ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
