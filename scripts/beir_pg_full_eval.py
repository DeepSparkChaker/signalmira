
"""Evaluate SignalMira PostgreSQL hybrid retrieval and rerankers.

Requires a separate database: signalmira_beir.
Uses official BEIR NFCorpus relevance labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from time import perf_counter

DATABASE = "signalmira_beir"
TENANT = "beir_eval"
SOURCE = "beir-nfcorpus"
DEFAULT_DATA = Path("tests/data/beir_eval/nfcorpus")
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


def guard_database(conn) -> None:
    """Never run against the normal SignalMira database."""
    row = conn.execute(
        "SELECT current_database() AS db"
    ).fetchone()
    db = row["db"] if isinstance(row, dict) else row[0]

    if db != DATABASE:
        raise RuntimeError(
            f"REFUSING: connected to {db!r}; expected {DATABASE!r}"
        )


def connect():
    from storage.postgres import connect as pg_connect

    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError(
            "Set DATABASE_URL to the separate signalmira_beir database"
        )

    conn = pg_connect(dsn)
    try:
        guard_database(conn)
    except BaseException:
        conn.close()
        raise

    return conn


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)

    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            result[row["query_id"]][row["doc_id"]] = int(
                row["relevance"]
            )

    return dict(result)


def unique_documents(rows: list[dict], limit: int) -> list[str]:
    """Convert ranked chunks to unique document IDs."""
    seen = set()
    ids = []

    for row in rows:
        doc_id = str(row["review_id"])

        if doc_id in seen:
            continue

        seen.add(doc_id)
        ids.append(doc_id)

        if len(ids) >= limit:
            break

    return ids


def select_unique_chunks(
    chunks: list[dict],
    limit: int = 25,
) -> list[dict]:
    """Keep the first RRF-ranked chunk per distinct document."""
    selected = []
    seen = set()

    for chunk in chunks:
        doc_id = (
            chunk["source"],
            chunk["review_id"],
        )

        if doc_id in seen:
            continue

        seen.add(doc_id)
        selected.append(chunk)

        if len(selected) >= limit:
            break

    return selected


def ingest(conn, data: Path) -> dict:
    """Import benchmark documents into the isolated database."""
    from psycopg.types.json import Jsonb
    from security.tenant_context import tenant_transaction

    docs = read_jsonl(data / "corpus.jsonl")

    if not docs:
        raise ValueError("Empty BEIR corpus")

    if any(
        not str(doc.get("doc_id", "")).strip()
        or not str(doc.get("text", "")).strip()
        for doc in docs
    ):
        raise ValueError("Invalid BEIR documents")

    if len({str(doc["doc_id"]) for doc in docs}) != len(docs):
        raise ValueError("Duplicate BEIR document IDs")

    sql = """
        INSERT INTO public.feedback_reviews
          (tenant_id, source, review_id, text,
           content_hash, attributes)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id, source, review_id)
        DO NOTHING
    """

    inserted = 0

    for start in range(0, len(docs), 100):
        with tenant_transaction(conn, TENANT):
            for doc in docs[start:start + 100]:
                body = str(doc["text"])
                digest = hashlib.sha256(
                    body.encode("utf-8")
                ).hexdigest()

                cursor = conn.execute(
                    sql,
                    (
                        TENANT,
                        SOURCE,
                        str(doc["doc_id"]),
                        body,
                        digest,
                        Jsonb({"benchmark": "BEIR NFCorpus"}),
                    ),
                )
                inserted += cursor.rowcount

    return {
        "corpus_documents": len(docs),
        "new_reviews": inserted,
        "tenant": TENANT,
    }


def create_embedder(revision: str, device: str):
    """Load the existing pinned SignalMira embedder."""
    if not SHA.fullmatch(revision):
        raise ValueError("Provide a pinned embedding model SHA")

    from retrieval.embedders import LocalQwenEmbedder

    return LocalQwenEmbedder(
        revision=revision,
        device=device,
        offline=True,
    )


def index(conn, revision: str, device: str) -> dict:
    """Reuse existing chunking and embedding pipelines."""
    from indexing.chunk_store import sync as chunk_sync
    from retrieval.semantic import sync as embedding_sync

    embedder = create_embedder(revision, device)

    chunks = chunk_sync(
        conn,
        TENANT,
        batch_size=100,
    )

    print("Chunk sync:", json.dumps(chunks), flush=True)

    embedder.warmup()

    embeddings = embedding_sync(
        conn,
        TENANT,
        embedder,
        page_size=64,
        batch_size=8,
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
    }


def check(conn, revision: str | None = None) -> dict:
    """Verify that the benchmark corpus and indexes exist."""
    counts = {}

    for table in (
        "feedback_reviews",
        "rag_chunks",
        "rag_embeddings",
    ):
        counts[table] = conn.execute(
            f"""
            SELECT count(*) AS n
            FROM public.{table}
            WHERE tenant_id=%s
            """,
            (TENANT,),
        ).fetchone()["n"]

    counts["bm25_index"] = str(
        conn.execute(
            """
            SELECT to_regclass(
                'public.rag_chunks_bm25_simple_idx'
            ) AS idx
            """
        ).fetchone()["idx"]
    )

    counts["hnsw_index"] = str(
        conn.execute(
            """
            SELECT to_regclass(
                'public.rag_embeddings_hnsw_cosine_idx'
            ) AS idx
            """
        ).fetchone()["idx"]
    )

    if revision is not None:
        embedder = create_embedder(revision, "cpu")

        counts["matching_document_embeddings"] = conn.execute(
            """
            SELECT count(*) AS n
            FROM rag_embeddings
            WHERE tenant_id=%s AND model_key=%s
            """,
            (TENANT, embedder.spec.model_key),
        ).fetchone()["n"]

    return counts


def evaluate(
    conn,
    data: Path,
    revision: str,
    device: str,
    reranker_model: str,
    reranker_revision: str | None,
    mode: str,
    output: Path,
    limit: int,
    fused_k: int = 25,
    unique_docs: bool = False,
) -> dict:
    """Run full BM25 + dense + RRF + optional reranking."""

    from retrieval.hybrid import search_vector
    from retrieval.reranker import Reranker, RerankerConfig
    from scripts.eval_retrieval_rankings import (
        evaluate as score_rankings,
    )

    if not 8 <= fused_k <= 100:
        raise ValueError("fused_k must be between 8 and 100")

    questions = read_jsonl(
        data / "queries.jsonl"
    )[:limit]

    if not questions:
        raise ValueError("No benchmark questions found")

    labels = load_qrels(data / "qrels.tsv")

    selected_labels = {
        str(question["query_id"]): labels[str(question["query_id"])]
        for question in questions
    }

    embedder = create_embedder(revision, device)

    available = check(conn, revision)
    expected = len(read_jsonl(data / "corpus.jsonl"))

    if (
        available["feedback_reviews"] != expected
        or available["rag_chunks"] < expected
        or available["matching_document_embeddings"]
            != available["rag_chunks"]
        or available["bm25_index"] == "None"
        or (
            mode == "hnsw"
            and available["hnsw_index"] == "None"
        )
    ):
        raise RuntimeError(
            f"Benchmark corpus/indexes incomplete: {available}"
        )

    embedder.warmup()
    print("Embedding model warmed", flush=True)

    reranker = None

    if reranker_model != "off":
        if (
            not reranker_revision
            or not SHA.fullmatch(reranker_revision)
        ):
            raise ValueError(
                "A pinned reranker revision is required"
            )

        reranker = Reranker(
            RerankerConfig(
                model=reranker_model,
                revision=reranker_revision,
                device=device,
                offline=True,
                timeout_s=120,
                batch_size=8,
                max_candidates=25,
            )
        )

        reranker.warmup()
        print(f"{reranker_model} warmed", flush=True)

    before = {}
    after = {}
    candidate_runs = {}
    fusion_runs = {}

    timings = {
        "embed_ms": [],
        "postgres_ms": [],
        "rerank_ms": [],
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with output.open("w", encoding="utf-8") as file:
            for i, question in enumerate(questions, 1):
                query_id = str(question["query_id"])
                text = str(question["query"])

                # STEP 1 — Embed the question
                t0 = perf_counter()
                query_vector = embedder.encode_query(text)
                t1 = perf_counter()

                # STEP 2 — Real PostgreSQL hybrid retrieval
                fused = search_vector(
                    conn,
                    TENANT,
                    text,
                    query_vector,
                    model_key=embedder.spec.model_key,
                    lexical_backend="bm25",
                    allow_shared_idf=True,
                    mode=mode,
                    sparse_k=40,
                    dense_k=40,
                    fused_k=fused_k,
                    rrf_k=60,
                    source=SOURCE,
                )
                t2 = perf_counter()

                # All distinct documents from the RRF pool
                fusion_runs[query_id] = unique_documents(
                    fused, fused_k
                )

                # STEP 3 — Keep at most 25 reranking candidates
                if unique_docs:
                    candidates = select_unique_chunks(
                        fused, limit=25
                    )
                else:
                    candidates = fused[:25]

                candidate_runs[query_id] = unique_documents(
                    candidates, 25
                )

                # Original RRF top 8
                before[query_id] = unique_documents(
                    candidates, 8
                )

                # STEP 4 — Rerank the selected candidates
                if reranker is not None and candidates:
                    reranked = reranker.rank(
                        text,
                        candidates,
                        top_k=len(candidates),
                    )

                    after[query_id] = unique_documents(
                        reranked, 8
                    )
                else:
                    after[query_id] = before[query_id]

                t3 = perf_counter()

                # STEP 5 — Record latency
                timings["embed_ms"].append(
                    (t1 - t0) * 1000
                )
                timings["postgres_ms"].append(
                    (t2 - t1) * 1000
                )
                timings["rerank_ms"].append(
                    (t3 - t2) * 1000
                )

                # STEP 6 — Save document-level predictions
                file.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "baseline_doc_ids": before[query_id],
                            "reranked_doc_ids": after[query_id],
                            "candidate_doc_ids": candidate_runs[query_id],
                            "fusion_doc_ids": fusion_runs[query_id],
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

                if i % 10 == 0:
                    print(
                        f"Evaluated {i}/{len(questions)}",
                        flush=True,
                    )

    finally:
        if reranker is not None:
            reranker.close()

        embedder.close()

    def quantile(numbers: list[float], fraction: float) -> float:
        values = sorted(numbers)
        position = max(
            0,
            min(
                len(values) - 1,
                int((len(values) - 1) * fraction + 0.999999),
            ),
        )
        return round(values[position], 1)

    result = {
        "retriever": (
            "SignalMira PostgreSQL BM25 + "
            "Qwen3 embeddings + pgvector + SQL RRF"
        ),
        "database": DATABASE,
        "mode": mode,
        "reranker": reranker_model,

        # Experiment configuration
        "fused_k": fused_k,
        "unique_docs": unique_docs,
        "rerank_candidate_limit": 25,

        "queries": len(questions),
        "baseline_rrf": score_rankings(
            selected_labels, before, 8
        ),
        "reranked": score_rankings(
            selected_labels, after, 8
        ),

        # Distinct-document recall entering reranker
        "candidate_recall_at_25": score_rankings(
            selected_labels, candidate_runs, 25
        )["recall@25"],

        # Distinct-document recall in entire RRF pool
        "fusion_document_recall": score_rankings(
            selected_labels, fusion_runs, fused_k
        )[f"recall@{fused_k}"],

        "stage_ms": {
            stage: {
                "p50": quantile(values, 0.50),
                "p95": quantile(values, 0.95),
            }
            for stage, values in timings.items()
        },
        "predictions": str(output),
    }

    output.with_suffix(".summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for name in ("ingest", "check", "index", "eval"):
        command = sub.add_parser(name)
        command.add_argument(
            "--data",
            type=Path,
            default=DEFAULT_DATA,
        )

        if name in ("check", "index", "eval"):
            command.add_argument("--revision")

        if name in ("index", "eval"):
            command.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default="cuda",
            )

        if name == "eval":
            command.add_argument(
                "--reranker",
                choices=("off", "qwen", "bge"),
                default="qwen",
            )
            command.add_argument("--reranker-revision")
            command.add_argument(
                "--mode",
                choices=("exact", "hnsw"),
                default="exact",
            )
            command.add_argument(
                "--limit",
                type=int,
                default=100,
            )

            # NEW: Experiment settings
            command.add_argument(
                "--fused-k",
                type=int,
                default=25,
            )
            command.add_argument(
                "--unique-docs",
                action="store_true",
            )

            command.add_argument(
                "--output",
                type=Path,
                default=Path(
                    "tests/data/beir_eval/full_pg_qwen.jsonl"
                ),
            )

    args = parser.parse_args()

    if args.command in ("index", "eval") and not args.revision:
        parser.error("--revision is required")

    if args.command == "eval" and not 1 <= args.limit <= 323:
        parser.error("--limit must be 1..323")

    with connect() as conn:
        if args.command == "ingest":
            result = ingest(conn, args.data)

        elif args.command == "check":
            result = check(conn, args.revision)

        elif args.command == "index":
            result = index(
                conn,
                args.revision,
                args.device,
            )

        else:
            result = evaluate(
                conn,
                args.data,
                args.revision,
                args.device,
                args.reranker,
                args.reranker_revision,
                args.mode,
                args.output,
                args.limit,
                args.fused_k,
                args.unique_docs,
            )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
