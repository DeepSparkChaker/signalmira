"""Shared business service for FastAPI, CSV streams, and offline enrichment jobs.

Does NOT read DB, choose tenant, authorize paid API, or buffer uploaded files.
Those duties belong to the authenticated caller and the durable job queue.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib

from enrichment.langchain_opinions import LangChainOpinionExtractor
from enrichment.opinion_schema import ReviewOpinions
from security.tenant_context import validate_tenant


@dataclass(frozen=True, slots=True)
class OpinionResult:
    tenant_id: str
    review_id: str
    input_hash: str
    model_id: str
    model_revision: str
    config_hash: str
    data: ReviewOpinions
    provider: str = "ollama"

    def as_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id, "review_id": self.review_id,
            "input_hash": self.input_hash, "provider": self.provider,
            "model_id": self.model_id, "model_revision": self.model_revision,
            "config_hash": self.config_hash, "data": self.data.model_dump(mode="json"),
        }


def _prepare(tenant: str, review_id: str, text: str) -> tuple[str, str]:
    tenant = validate_tenant(tenant)  # tenant MUST originate in trusted auth/job context
    if not isinstance(review_id, str) or not 1 <= len(review_id) <= 256:
        raise ValueError("review_id length must be 1..256")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return tenant, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wrap(extractor: LangChainOpinionExtractor, tenant: str, review_id: str,
          input_hash: str, result: ReviewOpinions) -> OpinionResult:
    return OpinionResult(tenant, review_id, input_hash, extractor.config.model,
                         extractor.config.model_revision, extractor.config.config_hash,
                         result, extractor.config.provider)


def analyze_review(extractor: LangChainOpinionExtractor, *, tenant: str,
                   review_id: str, text: str) -> OpinionResult:
    tenant, content_hash = _prepare(tenant, review_id, text)
    return _wrap(extractor, tenant, review_id, content_hash, extractor.extract(text))


async def aanalyze_review(extractor: LangChainOpinionExtractor, *, tenant: str,
                          review_id: str, text: str) -> OpinionResult:
    tenant, content_hash = _prepare(tenant, review_id, text)
    return _wrap(extractor, tenant, review_id, content_hash, await extractor.aextract(text))


def analyze_many(extractor: LangChainOpinionExtractor, *, tenant: str,
                 rows: Iterable[Mapping[str, str]], max_rows: int = 1000) -> Iterator[OpinionResult]:
    """Stream CSV rows, one review at a time; caller handles retry/commit."""
    if not isinstance(max_rows, int) or not 1 <= max_rows <= 100_000:
        raise ValueError("max_rows must be between 1 and 100000")
    tenant = validate_tenant(tenant)
    for i, row in enumerate(rows):
        if i >= max_rows:
            break
        yield analyze_review(extractor, tenant=tenant,
                             review_id=row["review_id"], text=row["text"])
