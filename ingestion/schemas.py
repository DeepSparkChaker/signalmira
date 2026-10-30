"""Validated feedback contract shared by CSV, scraper, and future GCS readers."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackRecord(BaseModel):
    """One original review; analysis and embeddings belong to later stages."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1, max_length=80)
    review_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(min_length=1, max_length=120)
    company: str | None = None
    product_id: str | None = None
    review_date: date | None = None
    url: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id", "review_id", "source")
    @classmethod
    def no_blank_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Identifier must not be blank")
        return value

    @field_validator("text")
    @classmethod
    def no_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review text must not be blank")
        return value  # Keep exact evidence, including original whitespace.


ALIASES = {
    "review_id": ("review_id", "id", "reviewId", "comment_id"),
    "text": ("text", "review", "review_text", "reviewText", "content", "body", "comment"),
    "company": ("company", "brand", "business", "business_name"),
    "product_id": ("product_id", "asin", "sku"),
    "review_date": ("review_date", "date", "created_at", "reviewDate"),
    "url": ("url", "review_url", "source_url"),
}
_KNOWN = {alias for group in ALIASES.values() for alias in group}


def _pick(row: Mapping[str, Any], field: str) -> Any:
    for name in ALIASES[field]:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def normalize_feedback(
    row: Mapping[str, Any], *, tenant_id: str, source: str
) -> FeedbackRecord:
    """Normalize one external record; identity/tenant always come from caller."""
    if not isinstance(row, Mapping):
        raise ValueError("Expected one JSON object / CSV row")

    text = _pick(row, "text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing or empty text field")
    review_id = _pick(row, "review_id")
    url = _pick(row, "url")
    if review_id is None:
        # Last-resort identity: stable for unchanged content; source IDs are preferred.
        fingerprint_input = f"{source}\0{url or ''}\0{text}".encode("utf-8")
        review_id = "auto-" + hashlib.sha256(fingerprint_input).hexdigest()[:24]

    raw_date = _pick(row, "review_date")
    if isinstance(raw_date, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", raw_date):
        raw_date = raw_date[:10]  # Preserve date portion of ISO timestamp.

    return FeedbackRecord(
        tenant_id=tenant_id,
        review_id=str(review_id),
        text=text,
        source=source,
        company=str(_pick(row, "company")) if _pick(row, "company") is not None else None,
        product_id=str(_pick(row, "product_id")) if _pick(row, "product_id") is not None else None,
        review_date=raw_date,
        url=str(url) if url is not None else None,
        attributes={k: v for k, v in row.items() if k not in _KNOWN},
    )
