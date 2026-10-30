"""Read authorized scraper exports (.jsonl preferred; small .json arrays)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schemas import FeedbackRecord, normalize_feedback

MAX_JSON_BYTES = 10_000_000  # JSON arrays are bounded; prefer JSONL at scale.
MAX_JSONL_LINE_CHARS = 2_000_000  # Reject unexpectedly massive JSONL records.


def read_scraped(
    path: str | Path, *, tenant_id: str, source: str
) -> Iterator[FeedbackRecord]:
    """Stream JSONL or read a bounded JSON array; validate each item."""
    path = Path(path)
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8-sig") as stream:
            line_number = 0
            while line := stream.readline(MAX_JSONL_LINE_CHARS + 1):
                line_number += 1
                if len(line) > MAX_JSONL_LINE_CHARS:
                    raise ValueError(f"{path}:{line_number}: oversized JSONL record")
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    yield normalize_feedback(item, tenant_id=tenant_id, source=source)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("JSON file too large; export as streaming JSONL instead")
        with path.open("r", encoding="utf-8-sig") as stream:
            payload: Any = json.load(stream)
        if isinstance(payload, dict):
            payload = payload.get("items", payload.get("reviews"))
        if not isinstance(payload, list):
            raise ValueError("JSON must be an array or an object with 'items'/'reviews'")
        for i, item in enumerate(payload, 1):
            try:
                yield normalize_feedback(item, tenant_id=tenant_id, source=source)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:item {i}: {exc}") from exc
    else:
        raise ValueError("Scraped export must be .jsonl, .ndjson, or .json")
