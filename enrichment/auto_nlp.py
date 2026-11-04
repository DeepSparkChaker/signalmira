"""One-review real-time language-aware enrichment, reusable from API/frontend.

This module does not write reviews or infer business issues from language ID.
Call from trusted API after authentication. For full ABSA/problems/topics, pass
existing analysis and topic-match callables. No models load at import time.

    analyze_one('El pago falla constantemente', tenant='demo',
                ner_enabled=True, conn=conn)

The raw feedback remains unchanged. NED resolves only against THIS tenant's
catalog. Model failures propagate; API may choose a partial-response policy.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Any, Callable

from enrichment.language import LanguageResult, detect_text
from security.tenant_context import tenant_transaction, validate_tenant


def ner_profile(language: str, *, override: str | None = None) -> str:
    """Language is metadata, not a routing rule. Use multilingual by default.

    A trusted operator may explicitly choose English-only 'accurate' after
    evaluating that tenant/source. Never derive this decision from language ID.
    """
    if override is not None and override not in {"multilingual", "accurate"}:
        raise ValueError("NER override must be 'multilingual' or 'accurate'")
    return override or "multilingual"


@lru_cache(maxsize=4)
def _ner_engine(profile: str, device: str, offline: bool):
    from enrichment.ner import Config, NER
    return NER(Config(profile=profile, device=device, offline=offline))


@lru_cache(maxsize=1)
def _qwen_extractor():
    from agent import build_extractor
    return build_extractor()


def _extract_opinions(text: str) -> dict:
    """Existing agent.py structured ABSA, lazy and evidence-span validated."""
    from agent import analyze
    return analyze(_qwen_extractor(), text).model_dump(mode="json")


def analyze_one(
    text: str,
    *,
    tenant: str,
    conn: Any | None = None,
    ner_enabled: bool = True,
    ner_engine: Any = None,
    detector: Callable[[str], LanguageResult] = detect_text,
    opinion_extractor: Callable[[str], Any] | None = None,
    topic_predictor: Callable[[str], Any] | None = None,
    device: str = "auto",
    offline: bool = True,
    with_llm: bool = False,
    ner_profile_override: str | None = None,
) -> dict:
    """Real-time preview without new data writes.

    `opinion_extractor` can be your existing structured Qwen ABSA function;
    `topic_predictor` must use a tenant-authorized activated taxonomy.
    With with_llm=True, reuse the existing local Ollama/Qwen `agent.py`
    extractor for issues/sentiment. It is lazy and not persisted in this API.
    Otherwise no LLM is loaded by default. Language detection is informational;
    NER always uses the multilingual checkpoint unless the trusted caller
    explicitly requests 'accurate' for a verified English-only workflow.
    """
    tenant = validate_tenant(tenant)
    language = detector(text)
    profile = ner_profile(language.language, override=ner_profile_override)
    output: dict[str, Any] = {
        "tenant_id": tenant,
        "language": asdict(language),
        "ner_profile": profile,
        "entities": [],
        "opinions": None,
        "topic": None,
    }
    if ner_enabled:
        worker = ner_engine if ner_engine is not None else _ner_engine(profile, device, offline)
        output["entities"] = worker.extract(text)
        if conn is not None and output["entities"]:
            # NED does not need separate language-specific model weights.
            from enrichment.ned import resolve
            with tenant_transaction(conn, tenant):
                for mention in output["entities"]:
                    mention["resolution"] = resolve(
                        conn, tenant=tenant, mention=mention["text"],
                        label=mention["label"]
                    )
    if with_llm and opinion_extractor is None:
        opinion_extractor = _extract_opinions
    if opinion_extractor is not None:
        output["opinions"] = opinion_extractor(text)
    if topic_predictor is not None:
        output["topic"] = topic_predictor(text)
    return output
