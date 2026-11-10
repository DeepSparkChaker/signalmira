"""Stateless reusable extraction API; no database access or CSV buffering.

A trusted authenticated API/job supplies the tenant identity and handles persistence.
Use the existing sentiment_jobs.py for durable incremental PostgreSQL sync.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Iterable, Iterator

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

    def as_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "review_id": self.review_id,
            "input_hash": self.input_hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "config_hash": self.config_hash,
            "data": self.data.model_dump(mode="json"),
        }


def analyze_review(extractor: LangChainOpinionExtractor, *,
                   tenant: str, review_id: str, text: str) -> OpinionResult:
    """Same entry point for preview/API and background work; no writes."""
    tenant = validate_tenant(tenant)
    if not isinstance(review_id, str) or not 1 <= len(review_id) <= 256:
        raise ValueError("Invalid review_id")
    output = extractor.extract(text)
    return OpinionResult(
        tenant_id=tenant, review_id=review_id,
        input_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        model_id=extractor.config.model,
        model_revision=extractor.config.model_revision,
        config_hash=extractor.config.config_hash,
        data=output,
    )


def analyze_many(extractor: LangChainOpinionExtractor, *, tenant: str,
                 rows: Iterable[dict], max_rows: int = 1000) -> Iterator[OpinionResult]:
    """Streaming iterator: caller may pass csv.DictReader; constant application RAM."""
    if not 1 <= max_rows <= 100_000:
        raise ValueError("Invalid max_rows")
    tenant = validate_tenant(tenant)
    for i, row in enumerate(rows):
        if i >= max_rows:
            break
        yield analyze_review(extractor, tenant=tenant,
                             review_id=row["review_id"], text=row["text"])
