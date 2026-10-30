"""Memory-efficient CSV reader: one normalized review at a time."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .schemas import FeedbackRecord, normalize_feedback


def read_csv(path: str | Path, *, tenant_id: str, source: str) -> Iterator[FeedbackRecord]:
    """Stream CSV; reject malformed rows with their file line number."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header is missing")
        for row in reader:
            if None in row:
                raise ValueError(f"{path}:{reader.line_num}: more fields than CSV header")
            if not row or all(value is None or not str(value).strip() for value in row.values()):
                continue
            try:
                yield normalize_feedback(row, tenant_id=tenant_id, source=source)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
