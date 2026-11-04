"""SignalMira Qwen3-Embedding-0.6B: one reproducible 1024-d document contract.

Explicit checkpoint revisions are mandatory for deployed services; the local CLI
may resolve 'main' once for experimentation. Share one instance per worker.
No text, queries, or vectors are retained in a global cache.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlsplit

import numpy as np

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024
TASK = "Given a customer feedback question, retrieve review passages that provide relevant evidence"
MAX_DOCUMENT_TOKENS = 2048
MAX_QUERY_TOKENS = 512
MAX_INPUT_CHARS = 8192
DEFAULT_BATCH = 8  # conservative on a 12GB GPU; benchmark 8, 16, 32
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
DOC_POLICY = f"document:raw:no-prompt:normalized-l2:tokens-{MAX_DOCUMENT_TOKENS}:v2"
QUERY_POLICY = f"query:instruct-v2:tokens-{MAX_QUERY_TOKENS}:task={TASK}"


class EmbeddingValidationError(ValueError):
    """Bad text, output vector, or tokenizer length."""


class EmbeddingBackendError(RuntimeError):
    """Embedding service or model returned an invalid response."""


def _canonical_hash(items) -> str:
    return hashlib.sha256(json.dumps(items, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    revision: str
    backend: str = "sentence-transformers"
    name: str = MODEL_NAME
    dimensions: int = DIMENSIONS

    def __post_init__(self):
        if not isinstance(self.revision, str) or not _SHA.fullmatch(self.revision):
            raise ValueError("Supply a pinned 40-character Hugging Face commit SHA")
        if self.backend not in ("sentence-transformers", "vllm") or self.dimensions != DIMENSIONS:
            raise ValueError("Unsupported embedder backend or vector dimension")
        if self.name != MODEL_NAME:
            raise ValueError("Unsupported model")

    @property
    def model_key(self) -> str:
        """Index contract: only settings that affect DOCUMENT embeddings."""
        return _canonical_hash((self.name, self.revision.lower(), self.backend,
                                self.dimensions, DOC_POLICY))

    @property
    def query_key(self) -> str:
        """Query instruction can change without invalidating document vectors."""
        return _canonical_hash((self.model_key, QUERY_POLICY))


class Embedder(Protocol):
    @property
    def spec(self) -> EmbeddingSpec: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _checked_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_INPUT_CHARS:
        raise EmbeddingValidationError(f"Expected nonblank text <= {MAX_INPUT_CHARS} characters")
    return value  # preserve exact original evidence


def query_text(query: str) -> str:
    return f"Instruct: {TASK}\nQuery: {_checked_text(query)}"


def validate_matrix(values, expected_rows: int | None = None) -> np.ndarray:
    """Single vectorized finite/shape check + L2 normalization; returns float32."""
    a = np.asarray(values, dtype=np.float32)
    if (a.ndim != 2 or a.shape[1] != DIMENSIONS or
            (expected_rows is not None and a.shape[0] != expected_rows) or
            not np.isfinite(a).all()):
        raise EmbeddingValidationError("Expected finite float32 matrix (N, 1024)")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-8):
        raise EmbeddingValidationError("Embedding contains a zero/invalid vector")
    return np.ascontiguousarray(a / norms, dtype=np.float32)


def checked_vector(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float32)
    if a.shape != (DIMENSIONS,):
        raise EmbeddingValidationError("Expected one 1024-dimensional vector")
    return validate_matrix(a[None, :], 1)[0]


def _validate_lengths(tokenizer, texts: Sequence[str], maximum: int) -> None:
    """Never silently truncate; use small bounded *batched* tokenizer calls.

    SentenceTransformer.encode() will tokenize again. Reuse would require a
    custom model forward; correctness is preferable to duplicating that stack.
    """
    if not texts:
        return
    encoded = tokenizer(list(texts), add_special_tokens=True,
                        truncation=False, padding=False)
    for i, ids in enumerate(encoded["input_ids"]):
        if len(ids) > maximum:
            raise EmbeddingValidationError(
                f"Input at batch position {i}: {len(ids)} tokens > {maximum}; rechunk"
            )


class LocalQwenEmbedder:
    """Lazy checkpoint, thread-safe single load/encode on one device (bounded VRAM)."""

    def __init__(self, *, revision: str | None = None, device: str = "auto",
                 offline: bool = False, batch_size: int = DEFAULT_BATCH,
                 attention: str = "sdpa"):
        if revision is not None and not _SHA.fullmatch(revision):
            raise ValueError("--revision must be a 40-character immutable commit SHA")
        if revision is None and os.getenv("SIGNALMIRA_REQUIRE_PINNED") == "1":
            raise ValueError("Set --revision to a pinned SHA in production")
        if device not in ("auto", "cpu", "cuda") or attention not in ("sdpa", "flash_attention_2"):
            raise ValueError("Unsupported device or attention backend")
        if not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be in 1..64")
        self._requested_revision = revision
        self._device = device
        self._offline = offline
        self._batch_size = batch_size
        self._attention = attention
        self._lock = threading.RLock()
        self._spec: EmbeddingSpec | None = None
        self._path: str | None = None
        self._model = None

    @property
    def spec(self) -> EmbeddingSpec:
        if self._spec is None:
            with self._lock:
                if self._spec is None:
                    from huggingface_hub import snapshot_download
                    directory = Path(snapshot_download(
                        repo_id=MODEL_NAME, revision=self._requested_revision,
                        local_files_only=self._offline,
                    ))
                    if not _SHA.fullmatch(directory.name):
                        raise EmbeddingBackendError("Unverifiable model snapshot revision")
                    self._path = str(directory)
                    self._spec = EmbeddingSpec(revision=directory.name)
        return self._spec

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    use_cuda = self._device != "cpu" and torch.cuda.is_available()
                    if self._device == "cuda" and not use_cuda:
                        raise EmbeddingBackendError("CUDA was explicitly required but is unavailable")
                    device = "cuda" if use_cuda else "cpu"
                    dtype = torch.float16 if use_cuda else torch.float32
                    if use_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    _ = self.spec  # checkpoint resolved inside the reentrant lock
                    kwargs = {"torch_dtype": dtype, "attn_implementation": self._attention}
                    model = SentenceTransformer(
                        self._path, device=device, local_files_only=True,
                        model_kwargs=kwargs,
                        tokenizer_kwargs={"padding_side": "left"},
                    )
                    model.max_seq_length = MAX_DOCUMENT_TOKENS
                    model.eval()
                    if model.get_sentence_embedding_dimension() != DIMENSIONS:
                        raise EmbeddingBackendError("Qwen model returned the wrong vector dimension")
                    self._model = model
        return self._model

    def _encode(self, texts: Sequence[str], *, maximum: int) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        output = []
        # Serialize calls per model. Concurrent FastAPI requests cannot OOM the
        # same GPU by running forward passes on the same instance simultaneously.
        with self._lock:
            model = self._load()
            import torch
            for group in batched(texts, self._batch_size):
                cleaned = [_checked_text(t) for t in group]
                _validate_lengths(model.tokenizer, cleaned, maximum)
                with torch.inference_mode():
                    encoded = model.encode(
                        cleaned, batch_size=min(self._batch_size, len(cleaned)),
                        normalize_embeddings=False, convert_to_numpy=True,
                        show_progress_bar=False, prompt="",
                    )
                output.append(validate_matrix(encoded, len(cleaned)))
        return np.concatenate(output, axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, maximum=MAX_DOCUMENT_TOKENS)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([query_text(text)], maximum=MAX_QUERY_TOKENS)[0]

    def warmup(self) -> None:
        self.encode_documents(["The payment failed after checkout."])

    def close(self) -> None:
        with self._lock:
            self._model = None
            # Let framework reclaim memory; do not clear a shared CUDA cache per batch.

    async def encode_query_async(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.encode_query, text)


class VllmQwenEmbedder:
    """Optional trusted vLLM server; separate backend model_key until parity proven.

    The client cannot attest server model weights; pin and verify vLLM server
    configuration operationally. Loads only the tokenizer locally for validation.
    """

    def __init__(self, *, revision: str, base_url: str = "http://127.0.0.1:8000/v1",
                 batch_size: int = 16, timeout: float = 30.0,
                 served_model_name: str = MODEL_NAME):
        u = urlsplit(base_url)
        try:
            _ = u.port  # reject fake ports like :8000.evil.com
        except ValueError as exc:
            raise ValueError("Invalid embedding endpoint port") from exc
        if (u.scheme not in ("http", "https") or u.username or u.password or not u.hostname or
                u.fragment or u.query or
                (u.scheme == "http" and u.hostname not in ("localhost", "127.0.0.1", "::1"))):
            raise ValueError("Use a trusted HTTPS or loopback HTTP embedding endpoint")
        if not 1 <= batch_size <= 128 or timeout <= 0:
            raise ValueError("Invalid vLLM batch size/timeout")
        if not served_model_name:
            raise ValueError("Served model name required")
        self._spec = EmbeddingSpec(revision=revision, backend="vllm")
        self.base_url = base_url
        self.served_model_name = served_model_name
        self.batch_size = batch_size
        self.timeout = timeout
        self._lock = threading.RLock()
        self._client = None
        self._tokenizer = None

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def _load_tokenizer(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        MODEL_NAME, revision=self.spec.revision,
                        local_files_only=os.getenv("SIGNALMIRA_OFFLINE") == "1",
                        padding_side="left",
                    )
        return self._tokenizer

    def _client_instance(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key=os.environ.get("VLLM_API_KEY", "local-only"),
                        timeout=self.timeout, max_retries=2,
                    )
        return self._client

    def _encode(self, texts: Sequence[str], *, maximum: int) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        matrices = []
        tokenizer = self._load_tokenizer()
        client = self._client_instance()
        for group in batched(texts, self.batch_size):
            cleaned = [_checked_text(t) for t in group]
            _validate_lengths(tokenizer, cleaned, maximum)
            resp = client.embeddings.create(model=self.served_model_name, input=cleaned)
            data = sorted(resp.data, key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(cleaned))):
                raise EmbeddingBackendError("vLLM embedding response indices missing/duplicated")
            matrices.append(validate_matrix([item.embedding for item in data], len(cleaned)))
        return np.concatenate(matrices, axis=0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, maximum=MAX_DOCUMENT_TOKENS)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([query_text(text)], maximum=MAX_QUERY_TOKENS)[0]

    async def encode_query_async(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self.encode_query, text)

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def local_embedder(*, revision: str | None = None, device: str = "auto",
                   offline: bool = False, batch_size: int = DEFAULT_BATCH) -> LocalQwenEmbedder:
    """Create one worker-owned instance; retain it in FastAPI lifespan/CLI job."""
    return LocalQwenEmbedder(revision=revision, device=device,
                             offline=offline, batch_size=batch_size)
