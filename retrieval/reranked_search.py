"""Optional SignalMira integration: existing hybrid.search -> selected reranker.

Does not modify retrieval.hybrid, BM25, pgvector, HNSW, or embeddings.
CLI: python -m retrieval.reranked_search --help
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from time import perf_counter
from typing import Any

from retrieval.reranker import Reranker, RerankerConfig, RerankOutcome


async def search_and_rerank(conn, tenant: str, query: str, embedder,
                            reranker: Reranker, **hybrid_options: Any) -> RerankOutcome:
    """Keep existing tenant_transaction in hybrid.search; don't share conn concurrently."""
    from retrieval.hybrid import search  # lazy import; no DB required for unit tests

    candidates = await asyncio.to_thread(
        search, conn, tenant, query, embedder, **hybrid_options
    )
    return await reranker.arank_with_status(query, candidates, top_k=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenant", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--revision", required=True,
                   help="pinned Qwen3 EMBEDDING revision (existing rag_embeddings)")
    p.add_argument("--reranker", choices=("qwen", "bge", "off"), default="qwen")
    p.add_argument("--reranker-revision",
                   help="selected RERANKER's separate pinned commit SHA")
    p.add_argument("--mode", choices=("exact", "hnsw"), default="exact")
    p.add_argument("--lexical-backend", choices=("bm25", "fts"), default="fts")
    p.add_argument("--allow-shared-idf", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--rerank-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--source")
    p.add_argument("--product-id")
    p.add_argument("--rerank-timeout", type=float, default=60.0,
                   help="CLI includes first cold model load; API should warm up")
    args = p.parse_args()

    from storage.postgres import connect
    from retrieval.embedders import LocalQwenEmbedder
    from retrieval.hybrid import search

    embedder = LocalQwenEmbedder(revision=args.revision,
                                 device=args.device, offline=args.offline)
    params = dict(lexical_backend=args.lexical_backend,
                  allow_shared_idf=args.allow_shared_idf, mode=args.mode,
                  source=args.source, product_id=args.product_id)
    started = perf_counter()
    with connect(os.environ["DATABASE_URL"]) as conn:
        if args.reranker == "off":
            hits = search(conn, args.tenant, args.query, embedder, **params)
            retrieval_ms = (perf_counter() - started) * 1000
            result = {"status": "rrf_only", "retrieval_ms": round(retrieval_ms, 1),
                      "results": hits[:8]}
        else:
            if not args.reranker_revision:
                p.error("--reranker-revision is required unless --reranker off")
            settings = RerankerConfig(model=args.reranker,
                                     revision=args.reranker_revision,
                                     device=args.rerank_device,
                                     offline=args.offline,
                                     timeout_s=args.rerank_timeout)
            ranker = Reranker(settings)  # ONLY selected model can be initialized
            try:
                hits = search(conn, args.tenant, args.query, embedder, **params)
                retrieval_ms = (perf_counter() - started) * 1000
                outcome = asyncio.run(ranker.arank_with_status(args.query, hits, top_k=8))
                result = {"status": outcome.status,
                          "reason": outcome.reason,
                          "reranker": outcome.selected_model,
                          "retrieval_ms": round(retrieval_ms, 1),
                          "rerank_ms": round(outcome.latency_ms, 1),
                          "results": outcome.results}
            finally:
                ranker.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
