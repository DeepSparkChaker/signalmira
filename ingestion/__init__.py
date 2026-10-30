"""Normalize feedback from external sources into one trusted schema."""

from .schemas import FeedbackRecord, normalize_feedback

__all__ = ["FeedbackRecord", "normalize_feedback"]
