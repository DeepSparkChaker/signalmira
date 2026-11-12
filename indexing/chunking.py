"""Exact-offset recursive review chunking; no database or model dependencies.

Up to 512 characters: one unchanged review. Longer reviews: bounded chunks
with overlap and natural-boundary preference. Offsets refer to original text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

# Prefer paragraphs, sentences, punctuation, words, then exact character cuts.
# Keep Unicode punctuation for multilingual customer feedback.
SEPARATORS = (
    "\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", "؟", "؛",
    "; ", ", ", "،", "，", "、", " ",
)
ALGORITHM = "recursive-char-v1"


class FeedbackLike(Protocol):
    """Compatible with ingestion.schemas.FeedbackRecord."""

    tenant_id: str
    source: str
    review_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    """Character lengths, not model tokens; changing settings changes version."""

    max_chars: int = 512
    overlap_chars: int = 64
    min_cut_fraction: float = 0.7

    def __post_init__(self) -> None:
        if self.max_chars < 2 or not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("Require max_chars >= 2 and 0 <= overlap_chars < max_chars")
        if not 0 < self.min_cut_fraction <= 1:
            raise ValueError("min_cut_fraction must be in (0, 1]")

    @property
    def version(self) -> str:
        """Stable fingerprint of the exact chunking policy."""
        policy = json.dumps(
            [ALGORITHM, self.max_chars, self.overlap_chars, self.min_cut_fraction, SEPARATORS],
            ensure_ascii=False, separators=(",", ":"),
        )
        return f"{ALGORITHM}:{hashlib.sha256(policy.encode()).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class RagChunk:
    """One search chunk, with original-text offsets and source version."""

    chunk_id: str
    tenant_id: str
    source: str
    review_id: str
    chunk_index: int
    text: str
    start_index: int
    end_index: int  # Exclusive
    input_hash: str  # SHA-256 of full original review text
    chunker_version: str


def _cut(text: str, floor: int, ceiling: int, separators: tuple[str, ...]) -> int:
    """Recursively fall back from strong boundaries to a character boundary."""
    if not separators:
        return ceiling
    separator = separators[0]
    index = text.rfind(separator, floor, ceiling)
    if index >= 0:
        return index + len(separator)
    return _cut(text, floor, ceiling, separators[1:])


def _bounds(text: str, settings: ChunkSettings) -> Iterator[tuple[int, int]]:
    """Yield contiguous, overlapping original offsets; never search for text."""
    start, length = 0, len(text)
    while start < length:
        limit = min(start + settings.max_chars, length)
        if limit == length:
            yield start, limit
            return
        floor = min(limit, start + max(1, int(settings.max_chars * settings.min_cut_fraction)))
        stop = _cut(text, floor, limit, SEPARATORS)
        yield start, stop
        start = max(start + 1, stop - settings.overlap_chars)


def chunk_review(
    review: FeedbackLike | Mapping[str, object], *, settings: ChunkSettings | None = None
) -> Iterator[RagChunk]:
    """Generate deterministic chunks from FeedbackRecord or psycopg dict_row."""
    settings = settings or ChunkSettings()
    def get(name: str) -> object:
        return review.get(name) if isinstance(review, Mapping) else getattr(review, name, None)

    ids: dict[str, str] = {}
    for field in ("tenant_id", "source", "review_id"):
        value = get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {field}")
        ids[field] = value
    text = get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text must not be empty")

    version = settings.version
    input_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for index, (start, end) in enumerate(_bounds(text, settings)):
        identity = json.dumps(
            [ids["tenant_id"], ids["source"], ids["review_id"], version, index, start, end],
            ensure_ascii=False, separators=(",", ":"),
        )
        yield RagChunk(
            chunk_id=str(uuid5(NAMESPACE_URL, identity)),
            tenant_id=ids["tenant_id"],
            source=ids["source"],
            review_id=ids["review_id"],
            chunk_index=index,
            text=text[start:end],
            start_index=start,
            end_index=end,
            input_hash=input_hash,
            chunker_version=version,
        )


def chunk_reviews(
    reviews: Iterable[FeedbackLike | Mapping[str, object]], *, settings: ChunkSettings | None = None
) -> Iterator[RagChunk]:
    """Stream an arbitrary number of reviews without collecting them in RAM."""
    settings = settings or ChunkSettings()
    for review in reviews:
        yield from chunk_review(review, settings=settings)
