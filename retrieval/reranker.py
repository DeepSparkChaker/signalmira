"""SignalMira: one selected, pinned, lazily loaded multilingual reranker.

Qwen is the default. BGE is a manually selected alternative, NEVER an
automatic fallback. On inference failure/busy/timeout, return original RRF
order. One bounded worker prevents abandoned, timed-out GPU inference from
spawning an unlimited queue. No data or embedding caches.

Requires sentence-transformers >= 5.4 for Qwen3 CrossEncoder support.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("signalmira.reranker")
MODEL_IDS = {
    "qwen": "Qwen/Qwen3-Reranker-0.6B",
    "bge": "BAAI/bge-reranker-v2-m3",
}
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


class RerankerBusy(RuntimeError):
    """The one bounded reranker worker is already running an inference."""


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    revision: str
    model: Literal["qwen", "bge"] = "qwen"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 8
    max_length: int = 512
    max_candidates: int = 25
    timeout_s: float = 10.0
    offline: bool = False
    qwen_instruction: str | None = None  # None: model's saved default instruction

    def __post_init__(self) -> None:
        if self.model not in MODEL_IDS:
            raise ValueError("model must be 'qwen' or 'bge'")
        if not isinstance(self.revision, str) or not _SHA.fullmatch(self.revision):
            raise ValueError("reranker revision must be a pinned 40-hex HF commit SHA")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not 1 <= self.batch_size <= 32:
            raise ValueError("batch_size must be in 1..32")
        if not 128 <= self.max_length <= 2048:
            raise ValueError("max_length must be in 128..2048")
        if not 1 <= self.max_candidates <= 25:
            raise ValueError("max_candidates must be in 1..25")
        if not 0.01 <= self.timeout_s <= 120.0:
            raise ValueError("timeout_s must be in 0.01..120")
        if self.qwen_instruction is not None:
            if self.model != "qwen":
                raise ValueError("qwen_instruction is only valid for the Qwen model")
            if not self.qwen_instruction.strip() or len(self.qwen_instruction) > 1000:
                raise ValueError("qwen_instruction must be non-empty and <= 1000 chars")

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.model]


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    results: list[dict[str, Any]]
    status: Literal["reranked", "rrf_fallback", "empty"]
    selected_model: str
    latency_ms: float
    reason: str | None = None


def _default_model_loader(settings: RerankerConfig) -> Any:
    """Import and load ONLY the model explicitly selected in settings."""
    import torch  # no torch/CUDA context at module import time
    from sentence_transformers import CrossEncoder

    use_cuda = settings.device != "cpu" and torch.cuda.is_available()
    if settings.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested, but PyTorch has no CUDA device")
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float32
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    kwargs: dict[str, Any] = {
        "device": device,
        "revision": settings.revision,
        "local_files_only": settings.offline,
        "max_length": settings.max_length,
        "model_kwargs": {"torch_dtype": dtype},
        # Explicit raw logits for both models; never compare scores across models.
        "activation_fn": torch.nn.Identity(),
    }
    if settings.qwen_instruction is not None:
        kwargs.update(
            prompts={"signalmira": settings.qwen_instruction},
            default_prompt_name="signalmira",
        )
    # Qwen's saved default rerank instruction is used when no override is given.
    model = CrossEncoder(settings.model_id, **kwargs)
    model.eval()
    LOG.info("reranker model loaded: %s %s on %s", settings.model_id,
             settings.revision[:12], device)
    return model


def _prepare(query: str, candidates: Sequence[Mapping[str, Any]],
             *, top_k: int, max_candidates: int) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip() or len(query) > 4096:
        raise ValueError("nonempty query <= 4096 characters required")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 25:
        raise ValueError("top_k must be an integer in 1..25")
    if len(candidates) > max_candidates:
        raise ValueError(f"at most {max_candidates} RRF candidates allowed")

    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidates must be mappings")
        chunk_id = candidate.get("chunk_id")
        chunk_text = candidate.get("chunk_text")
        if chunk_id is None or not str(chunk_id).strip():
            raise ValueError("each candidate requires a nonblank chunk_id")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError("each candidate requires nonblank chunk_text")
        if len(chunk_text) > 20000:
            raise ValueError("chunk_text exceeds 20000 characters; rechunk first")
        identity = str(chunk_id)
        if identity in seen:
            continue  # retain original first appearance and its RRF rank
        seen.add(identity)
        output.append(dict(candidate))
    return output


class Reranker:
    """One instance = one selected model and at most one in-flight inference.

    The model isn't loaded until warmup() or rank()/arank() is called.
    Use one instance per dedicated inference worker (not per HTTP request).
    """

    def __init__(self, settings: RerankerConfig, *,
                 model_loader: Callable[[RerankerConfig], Any] | None = None):
        self.settings = settings
        self._loader = model_loader or _default_model_loader
        self._model: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rerank")
        self._guard = threading.Lock()
        self._inflight: Future | None = None
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _score(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if self._model is None:
            self._model = self._loader(self.settings)
        pairs = [(query, doc["chunk_text"]) for doc in docs]
        scores = np.asarray(self._model.predict(
            pairs,
            batch_size=self.settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ), dtype=np.float64).reshape(-1)
        if len(scores) != len(docs) or not np.isfinite(scores).all():
            raise RuntimeError("reranker produced invalid/mismatched scores")
        # Stable ties preserve original RRF order.
        return sorted(
            ({**doc, "rerank_score": float(score),
              "rerank_model": self.settings.model_id}
             for doc, score in zip(docs, scores)),
            key=lambda doc: -doc["rerank_score"],
        )[:top_k]

    def _submit(self, query: str, docs: list[dict[str, Any]], top_k: int) -> Future:
        with self._guard:
            if self._closed:
                raise RuntimeError("reranker is closed")
            if self._inflight is not None and not self._inflight.done():
                raise RerankerBusy("reranker already processing another batch")
            future = self._executor.submit(self._score, query, docs, top_k)
            self._inflight = future
            return future

    def warmup(self) -> None:
        """Explicitly load and run inference once; call in FastAPI lifespan."""
        self.rank("battery complaints", [
            {"chunk_id": "warmup", "chunk_text": "Battery drains quickly."}
        ], top_k=1)

    def rank(self, query: str, candidates: Sequence[Mapping[str, Any]], *,
             top_k: int = 8) -> list[dict[str, Any]]:
        """Synchronous scoring, raises on bad input/busy/inference error."""
        docs = _prepare(query, candidates, top_k=top_k,
                        max_candidates=self.settings.max_candidates)
        if not docs:
            return []
        return self._submit(query, docs, top_k).result()

    async def arank_with_status(self, query: str,
                                candidates: Sequence[Mapping[str, Any]], *,
                                top_k: int = 8) -> RerankOutcome:
        """Fast fallback on failure/busy/timeout. Timed-out inference remains bounded.

        A timed-out worker is NOT interrupted by asyncio cancellation: the
        actual thread continues, and subsequent requests fall back to RRF
        rather than queue another model call. Do not use a zero timeout to
        terminate GPU work; use a separate process for hard kill semantics.
        """
        started = perf_counter()
        docs = _prepare(query, candidates, top_k=top_k,
                        max_candidates=self.settings.max_candidates)
        if not docs:
            return RerankOutcome([], "empty", self.settings.model_id, 0.0)
        try:
            future = self._submit(query, docs, top_k)
            wrapped = asyncio.wrap_future(future)
            # Consume a possible late exception after a timeout to avoid a
            # "Future exception was never retrieved" loop warning.
            wrapped.add_done_callback(
                lambda f: None if f.cancelled() else f.exception()
            )
            result = await asyncio.wait_for(
                asyncio.shield(wrapped), timeout=self.settings.timeout_s
            )
            return RerankOutcome(result, "reranked", self.settings.model_id,
                                 (perf_counter() - started) * 1000)
        except RerankerBusy:
            reason = "busy"
        except TimeoutError:
            reason = "timeout"
        except asyncio.CancelledError:
            raise  # request cancelled by caller; don't swallow cancellation
        except Exception:
            reason = "model_error"
            LOG.exception("reranker inference failed; serving RRF candidates")
        if reason in ("busy", "timeout"):
            LOG.warning("reranker %s; serving RRF candidates", reason)
        return RerankOutcome(docs[:top_k], "rrf_fallback", self.settings.model_id,
                             (perf_counter() - started) * 1000, reason)

    async def arank(self, query: str,
                    candidates: Sequence[Mapping[str, Any]], *,
                    top_k: int = 8) -> list[dict[str, Any]]:
        """Simple FastAPI-friendly API returning rows only."""
        return (await self.arank_with_status(query, candidates, top_k=top_k)).results

    def close(self) -> None:
        """Call at worker shutdown, after stopping intake of new requests."""
        with self._guard:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._model = None


def _cli() -> None:
    p = argparse.ArgumentParser(description="Rerank a saved hybrid retrieval JSON list")
    p.add_argument("--candidates", required=True, type=Path,
                   help="JSON file containing the hybrid top-25 list")
    p.add_argument("--query", required=True)
    p.add_argument("--model", choices=tuple(MODEL_IDS), default="qwen")
    p.add_argument("--revision", required=True, help="40-character reranker model SHA")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()
    with args.candidates.open(encoding="utf-8") as handle:
        candidates = json.load(handle)
    if not isinstance(candidates, list):
        p.error("--candidates must contain a JSON list")
    settings = RerankerConfig(
        revision=args.revision, model=args.model, device=args.device,
        offline=args.offline, batch_size=args.batch_size,
        max_length=args.max_length, timeout_s=args.timeout,
    )
    reranker = Reranker(settings)
    try:
        # Only the selected model gets loaded.
        outcome = asyncio.run(reranker.arank_with_status(
            args.query, candidates, top_k=args.top_k
        ))
        print(json.dumps({
            "status": outcome.status,
            "model": outcome.selected_model,
            "reason": outcome.reason,
            "rerank_ms": round(outcome.latency_ms, 1),
            "results": outcome.results,
        }, indent=2, ensure_ascii=False, default=str))
    finally:
        reranker.close()


if __name__ == "__main__":
    _cli()
