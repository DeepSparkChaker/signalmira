"""Compatibility helper: no language statistics influence model selection."""
from __future__ import annotations


def discovery_profile(conn, tenant: str, explicit: str | None = None) -> str:
    """Default to BGE-M3 for NEW taxonomies; allow trusted explicit override."""
    if explicit is not None and explicit not in {
        "fast", "accurate", "multilingual", "multilingual_m3"
    }:
        raise ValueError("Unknown topic profile")
    return explicit or "multilingual_m3"
