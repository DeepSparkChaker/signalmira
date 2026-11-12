"""Strict, evidence-grounded, Unicode-safe opinion extraction contract.

Offsets are Python character offsets into the *original* raw review, end-exclusive.
The classifier's overall sentiment is stored separately; LLM output is diagnostic.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Polarity = Literal['positive', 'negative', 'neutral', 'mixed']
SCHEMA_VERSION = 'sm-opinions-v1'


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    quote: str = Field(min_length=1, max_length=2000)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode='after')
    def check_bounds(self):
        if self.end <= self.start:
            raise ValueError('end must exceed start')
        return self

    def verify(self, source: str) -> None:
        if self.end > len(source) or source[self.start:self.end] != self.quote:
            raise ValueError(f'Unverifiable evidence span at {self.start}:{self.end}')

    def align(self, source: str) -> 'EvidenceSpan':
        """Repair wrong LLM offsets only when the exact quote occurs exactly once."""
        if self.end <= len(source) and source[self.start:self.end] == self.quote:
            return self
        start = source.find(self.quote)
        if start < 0 or source.find(self.quote, start + 1) >= 0:
            raise ValueError('Unverifiable or ambiguous evidence quote')
        return self.model_copy(update={'start': start, 'end': start + len(self.quote)})


class Opinion(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    aspect: str = Field(min_length=1, max_length=100)
    target: str | None = Field(default=None, max_length=160)
    sentiment: Polarity
    evidence: EvidenceSpan
    reported_problem: str | None = Field(default=None, max_length=400)
    customer_suggestion: EvidenceSpan | None = None


class ReviewOpinions(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    overall_sentiment: Polarity
    opinions: list[Opinion] = Field(default_factory=list, max_length=16)

    def align(self, source: str) -> 'ReviewOpinions':
        """Deterministically align unique verbatim quotes before final verification."""
        updated = []
        for opinion in self.opinions:
            evidence = opinion.evidence.align(source)
            suggestion = (opinion.customer_suggestion.align(source)
                          if opinion.customer_suggestion is not None else None)
            updated.append(opinion.model_copy(update={
                'evidence': evidence, 'customer_suggestion': suggestion}))
        return self.model_copy(update={'opinions': updated}).verify(source)

    def verify(self, source: str) -> 'ReviewOpinions':
        if not source.strip():
            raise ValueError('Review text cannot be empty')
        seen: set[tuple[str, int, int]] = set()
        for opinion in self.opinions:
            opinion.evidence.verify(source)
            if opinion.customer_suggestion is not None:
                opinion.customer_suggestion.verify(source)
            key = (opinion.aspect.casefold(), opinion.evidence.start, opinion.evidence.end)
            if key in seen:
                raise ValueError('Duplicate opinion evidence')
            seen.add(key)
        return self
