"""Local SignalMira retrieval lab: warm the selected model and inspect RRF vs rerank.

No Streamlit imports: testable without UI, no state retained across tenants.
Uses the existing retrieval.hybrid.search_vector() tenant transaction and SQL.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
SAMPLE_QUERY = "Customers complaining about rapid battery drain"
SAMPLE_REVIEW_ID = "uci-amazon-0694"
SAMPLE_REVIEW_TEXT = "The biggest complaint I have is, the battery drains superfast."


@dataclass(frozen=True, slots=True)
class LabConfig:
    embedding_revision: str = EMBEDDING_REVISION
    reranker_model: str = "qwen"  # qwen, bge, off
    reranker_revision: str = ""
    device: str = "cuda"
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not _SHA.fullmatch(self.embedding_revision):
            raise ValueError("Set a valid pinned embedding model SHA")
        if self.reranker_model not in ("qwen", "bge", "off"):
            raise ValueError("Reranker must be qwen, bge or off")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda")
        if self.reranker_model != "off" and not _SHA.fullmatch(self.reranker_revision):
            raise ValueError("Set SIGNALMIRA_RERANKER_SHA to the selected model's 40-char SHA")
        if not 0.1 <= self.timeout_s <= 120:
            raise ValueError("Reranker timeout must be 0.1..120 seconds")


def warm_selected(config: LabConfig) -> tuple[Any, Any | None, dict[str, float]]:
    """Called only once by Streamlit resource cache; loads NO unselected reranker."""
    from retrieval.embedders import LocalQwenEmbedder
    from retrieval.reranker import Reranker, RerankerConfig

    embedder = LocalQwenEmbedder(
        revision=config.embedding_revision, device=config.device, offline=True
    )
    ranker = None
    try:
        t0 = perf_counter()
        embedder.warmup()  # document path
        embedder.encode_query(SAMPLE_QUERY)  # query prompt path
        t1 = perf_counter()
        if config.reranker_model != "off":
            ranker = Reranker(RerankerConfig(
                model=config.reranker_model,
                revision=config.reranker_revision,
                device=config.device,
                offline=True,
                timeout_s=config.timeout_s,
            ))
            ranker.warmup()
        t2 = perf_counter()
        return embedder, ranker, {
            "embedding_warmup_ms": round((t1 - t0) * 1000, 1),
            "reranker_warmup_ms": round((t2 - t1) * 1000, 1),
        }
    except Exception:
        if ranker is not None:
            ranker.close()
        embedder.close()
        raise


def _search_vector(*args: Any, **kwargs: Any):
    """Import the existing hybrid SQL only when retrieval is requested."""
    from retrieval.hybrid import search_vector
    return search_vector(*args, **kwargs)


def search_lab(
    conn: Any, tenant: str, query: str, embedder: Any, ranker: Any | None, *,
    lexical_backend: str = "bm25", allow_shared_idf: bool = False,
    mode: str = "exact", source: str | None = None, top_k: int = 8,
) -> dict[str, Any]:
    """Keep one query vector, one SQL retrieval and optional rerank; no DB writes."""
    if not isinstance(query, str) or not query.strip() or len(query) > 4096:
        raise ValueError("Provide a question of 1..4096 characters")
    if top_k != 8:
        raise ValueError("Local lab currently displays exactly top 8")
    if lexical_backend not in ("bm25", "fts") or mode not in ("exact", "hnsw"):
        raise ValueError("Unsupported retrieval setting")

    # Separate timers: first query embedding; then only database hybrid SQL.
    t0 = perf_counter()
    vector = embedder.encode_query(query)
    t1 = perf_counter()
    hits = _search_vector(
        conn, tenant, query, vector, model_key=embedder.spec.model_key,
        lexical_backend=lexical_backend, allow_shared_idf=allow_shared_idf,
        mode=mode, sparse_k=40, dense_k=40, fused_k=25, rrf_k=60,
        source=source,
    )
    t2 = perf_counter()
    numbered = [{**hit, "rrf_rank": i} for i, hit in enumerate(hits, start=1)]
    rrf_eight = numbered[:top_k]
    if ranker is None:
        ranked = rrf_eight
        status, reason, rerank_ms = "rrf_only", None, 0.0
    else:
        # Streamlit executes this synchronous function in a script thread.
        # On timeout/busy, arank_with_status() returns original RRF order.
        outcome = asyncio.run(ranker.arank_with_status(query, numbered, top_k=top_k))
        ranked, status, reason = outcome.results, outcome.status, outcome.reason
        rerank_ms = outcome.latency_ms
    return {
        "query": query,
        "status": status,
        "reason": reason,
        "candidate_count": len(numbered),
        "rrf_top8": rrf_eight,
        "results": ranked,
        "embedding_ms": round((t1 - t0) * 1000, 1),
        "postgres_ms": round((t2 - t1) * 1000, 1),
        "rerank_ms": round(rerank_ms, 1),
        "total_ms": round(((t2 - t0) * 1000) + rerank_ms, 1),
    }
