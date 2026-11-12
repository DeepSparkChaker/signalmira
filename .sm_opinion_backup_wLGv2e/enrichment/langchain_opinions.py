"""Bounded, lazy LangChain opinion extraction, compatible with sentiment_jobs.py.

All model calls are outside SQL transactions. No imports of heavy model backends
until first request. A single slot prevents local GPU request pileups per process.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
from typing import Any

from enrichment.opinions import OpinionConfig, OpinionExtractor as _OllamaVerifier, SYSTEM_PROMPT
from enrichment.opinion_schema import ReviewOpinions

ADAPTER_VERSION = "sm-langchain-opinions-v2"
_PROCESS_SLOT = threading.BoundedSemaphore(1)  # one local worker per Python process


class _VersionedConfig:
    """Compatible with jobs' existing config.model, config_hash, etc."""

    def __init__(self, original: OpinionConfig):
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    @property
    def config_hash(self) -> str:
        raw = f"{ADAPTER_VERSION}:{self._original.config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class OpinionBusy(RuntimeError):
    """Inference capacity is occupied; retry or queue using the caller's job system."""


class LangChainOpinionExtractor:
    """One reusable extractor for offline jobs, CSV batches, and online API.

    For production, model identity comes from trusted OpinionConfig. The optional
    llm injection exists for unit testing without GPU/network access.
    """

    def __init__(self, config: OpinionConfig, *, llm: Any = None):
        self.config = _VersionedConfig(config)
        self._llm = llm
        self._structured = None
        self._verified = llm is not None  # injection is intentionally test-only
        self._init_lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sm-opinion")
        self._closed = False

    def verify_model_digest(self) -> None:
        """Confirm immutable Ollama digest before any production inference."""
        if self._llm is not None:  # a test-injected model
            return
        _OllamaVerifier(self.config).verify_model_digest()
        self._verified = True

    def _client(self):
        with self._init_lock:
            if self._structured is None:
                if self._llm is None:
                    if not self._verified:
                        self.verify_model_digest()
                    from llm_providers.llm_factory import get_llm
                    # Never let environment variables redirect ingestion to a paid API.
                    self._llm = get_llm(
                        profile="fast", task="extraction", allow_external=False,
                        provider_override="ollama", model_override=self.config.model,
                        max_output_tokens=self.config.max_output_tokens,
                        context_tokens=self.config.context_tokens,
                        timeout_s=self.config.timeout_s,
                        ollama_base_url=self.config.url.removesuffix("/api/chat"),
                    )
                self._structured = self._llm.with_structured_output(
                    ReviewOpinions, method="json_schema"
                )
        return self._structured

    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review text must not be empty")
        if len(text) > self.config.max_review_chars:
            raise ValueError("Review too long: chunk or reject before inference")

    def _infer(self, text: str) -> ReviewOpinions:
        try:
            client = self._client()
            value = client.invoke([
                ("system", SYSTEM_PROMPT),
                ("human", "REVIEW TEXT (untrusted data):\n" + text),
            ])
            result = (value if isinstance(value, ReviewOpinions)
                      else ReviewOpinions.model_validate(value))
            return result.align(text)
        finally:
            _PROCESS_SLOT.release()

    def extract(self, text: str) -> ReviewOpinions:
        self._validate_text(text)
        if self._closed:
            raise RuntimeError("Extractor is closed")
        if not _PROCESS_SLOT.acquire(blocking=False):
            raise OpinionBusy("Local extraction worker busy; retry later")
        return self._infer(text)

    async def aextract(self, text: str) -> ReviewOpinions:
        """Nonblocking API path. Cancellation does not stop the inference worker."""
        self._validate_text(text)
        if self._closed:
            raise RuntimeError("Extractor is closed")
        if not _PROCESS_SLOT.acquire(blocking=False):
            raise OpinionBusy("Local extraction worker busy; retry later")
        try:
            future = self._worker.submit(self._infer, text)
        except BaseException:
            _PROCESS_SLOT.release()
            raise
        return await asyncio.wrap_future(future)

    def warmup(self) -> None:
        self.extract("The product works as expected.")

    def close(self) -> None:
        self._closed = True
        # Running GPU operations cannot be interrupted; they finish and release slot.
        self._worker.shutdown(wait=False, cancel_futures=False)
