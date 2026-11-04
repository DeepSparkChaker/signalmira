"""Evaluate labeled semantic retrieval on actual tenant review evidence.

Input JSONL, one per line:
  {"query": "payment failed", "expected_chunk_ids": ["<uuid>"], "language": "en"}

This is *not* an LLM judge. Human-labeled relevance is required; do not use
vector similarity to create your own 'ground truth'.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from uuid import UUID


def scores(retrieved: list[str], relevant: set[str]) -> tuple[float, float]:
    if not relevant:
        raise ValueError("Reference must include at least one relevant chunk UUID")
    hit = relevant.intersection(retrieved)
    first = next((i + 1 for i, item in enumerate(retrieved) if item in relevant), None)
    return len(hit) / len(relevant), 1.0 / first if first else 0.0


def evaluate(conn, *, tenant: str, embedder, gold: Path, k: int = 10,
             max_queries: int = 1000) -> dict:
    from retrieval.semantic import search
    if not 1 <= k <= 100 or not 1 <= max_queries <= 10000:
        raise ValueError("Invalid evaluation size")
    totals = []
    groups = defaultdict(list)
    with gold.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if len(totals) >= max_queries:
                raise ValueError("Too many evaluation queries; increase --max-queries deliberately")
            record = json.loads(line)
            question = record["query"]
            expected = {str(UUID(v)) for v in record["expected_chunk_ids"]}
            if not expected or not isinstance(question, str) or not question.strip():
                raise ValueError("Each query requires nonblank text and expected_chunk_ids")
            start = time.perf_counter()
            found = search(conn, tenant, question, embedder, k=k, mode="exact")
            elapsed = (time.perf_counter() - start) * 1000
            recall, mrr = scores([str(row["chunk_id"]) for row in found], expected)
            result = (recall, mrr, elapsed)
            totals.append(result)
            groups[record.get("language", "und")].append(result)
    if not totals:
        raise ValueError("Empty benchmark file")
    def report(values):
        return {
            "n": len(values),
            "mean_recall_at_k": round(statistics.mean(row[0] for row in values), 4),
            "mean_reciprocal_rank_at_k": round(statistics.mean(row[1] for row in values), 4),
            "median_end_to_end_ms": round(statistics.median(row[2] for row in values), 1),
        }
    return {"k": k, "document_model_key": embedder.spec.model_key,
            "query_key": embedder.spec.query_key,
            "all": report(totals),
            "by_language": {lang: report(values) for lang, values in sorted(groups.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenant", required=True, help="Authorized local operator tenant")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--revision", required=True, help="Pinned 40-hex model revision")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    from storage.postgres import connect
    from retrieval.embedders import LocalQwenEmbedder
    with connect(os.environ["DATABASE_URL"]) as conn:
        model = LocalQwenEmbedder(revision=args.revision, device=args.device)
        model.warmup()  # exclude first-load time from per-query measurement
        print(json.dumps(evaluate(conn, tenant=args.tenant, embedder=model,
                                  gold=args.gold, k=args.k,
                                  max_queries=args.max_queries), indent=2))


if __name__ == "__main__":
    main()
