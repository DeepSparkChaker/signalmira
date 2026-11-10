"""EXAMPLE ONLY. Integrate snippets into existing SignalMira FastAPI app.

Select exactly ONE reranker at service startup, never by untrusted request.
Uses the pinned SHA for the selected model. Not an app replacement.
"""
from contextlib import asynccontextmanager
import asyncio
import os

from retrieval.reranker import Reranker, RerankerConfig


@asynccontextmanager
async def lifespan(app):
    selected = os.getenv("SIGNALMIRA_RERANKER_MODEL", "qwen")
    if selected not in ("qwen", "bge"):
        raise ValueError("SIGNALMIRA_RERANKER_MODEL must be 'qwen' or 'bge'")
    revision_var = ("SIGNALMIRA_QWEN_RERANKER_SHA" if selected == "qwen"
                    else "SIGNALMIRA_BGE_RERANKER_SHA")
    reranker = Reranker(RerankerConfig(
        model=selected, revision=os.environ[revision_var],
        device=os.getenv("SIGNALMIRA_RERANKER_DEVICE", "auto"),
        timeout_s=float(os.getenv("SIGNALMIRA_RERANKER_TIMEOUT_S", "10")),
    ))
    try:
        await asyncio.to_thread(reranker.warmup)  # selected model only; first request fast
        app.state.reranker = reranker
        yield
    finally:
        await asyncio.to_thread(reranker.close)


# Inside an existing tenant-authorized endpoint, after hybrid.search() returns
# its top 25 candidates and RLS has been enforced:
# outcome = await request.app.state.reranker.arank_with_status(query, candidates, top_k=8)
# return {"status": outcome.status, "evidence": outcome.results}
# A timeout/failure falls back to RRF. BGE is NEVER loaded automatically.
