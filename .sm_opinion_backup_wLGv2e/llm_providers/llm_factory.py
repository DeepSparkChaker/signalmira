"""SignalMira LLM client factory: lazy, no implicit paid fallback.

Trusted backend code alone may supply overrides. No model is loaded on import.
This module does not authorize access to a tenant or verify model weights.
"""
from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlsplit

Profile = Literal["fast", "accurate", "premium"]
Task = Literal["extraction", "analyst"]

DEFAULTS = {
    "fast": ("ollama", "qwen3.5:4b"),
    "accurate": ("ollama", "qwen3.5:9b"),
    "premium": ("openai", "gpt-5.5"),
}
EXTERNAL = frozenset({"openai", "anthropic"})


def _private_vllm_url(value: str) -> str:
    """Reject credentials, query params, and insecure non-loopback endpoints."""
    p = urlsplit(value)
    if (p.scheme not in {"http", "https"} or not p.hostname
            or p.username or p.password or p.query or p.fragment
            or (p.scheme == "http" and p.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("vLLM must use HTTPS or local loopback HTTP; no URL credentials")
    return value.rstrip("/")


def get_llm(
    profile: Profile = "fast",
    task: Task = "extraction",
    *,
    allow_external: bool = False,
    provider_override: str | None = None,
    model_override: str | None = None,
    max_output_tokens: int | None = None,
    context_tokens: int | None = None,
    timeout_s: int = 60,
    ollama_base_url: str | None = None,
):
    """Construct only the chosen client; never silently substitute a provider.

    Overrides are trusted configuration, NOT arguments derived from review text.
    """
    if profile not in DEFAULTS or task not in {"extraction", "analyst"}:
        raise ValueError("Unknown model profile or task")
    if not isinstance(timeout_s, int) or not 1 <= timeout_s <= 600:
        raise ValueError("timeout_s must be 1..600 seconds")

    default_provider, default_model = DEFAULTS[profile]
    prefix = f"SM_LLM_{profile.upper()}"
    provider = (provider_override or os.getenv(f"{prefix}_PROVIDER")
                or (os.getenv("LLM_BACKEND") if profile != "premium" else None)
                or default_provider).lower()
    model = (model_override or os.getenv(f"{prefix}_MODEL") or default_model).strip()
    if not model or len(model) > 256:
        raise ValueError("Invalid model ID")
    if provider not in {"ollama", "vllm", "openai", "anthropic"}:
        raise ValueError(f"Unknown provider: {provider}")
    if provider in EXTERNAL and not allow_external:
        raise PermissionError("Paid/external inference requires trusted tenant authorization")

    tokens = (max_output_tokens if max_output_tokens is not None
              else (1536 if task == "extraction" else 3072))
    context = (context_tokens if context_tokens is not None
               else (4096 if task == "extraction" else 8192))
    if not isinstance(tokens, int) or not 128 <= tokens <= 8192:
        raise ValueError("Invalid max_output_tokens")
    if not isinstance(context, int) or not 2048 <= context <= 65536:
        raise ValueError("Invalid context_tokens")
    if provider != "ollama" and ollama_base_url is not None:
        raise ValueError("ollama_base_url may be supplied only for Ollama")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=model, temperature=0, reasoning=False if task == "extraction" else None,
                      num_ctx=context, num_predict=tokens, keep_alive="5m",
                      client_kwargs={"timeout": timeout_s})
        if ollama_base_url is not None:
            # Already validated by OpinionConfig for the extraction worker.
            p = urlsplit(ollama_base_url)
            if (p.scheme != "http" or p.hostname not in {"localhost", "127.0.0.1"}
                    or p.port != 11434 or p.path not in {"", "/"}
                    or p.username or p.password or p.query or p.fragment):
                raise ValueError("Ollama only supports the local loopback server")
            kwargs["base_url"] = ollama_base_url
        return ChatOllama(**kwargs)

    if provider == "vllm":
        base = _private_vllm_url(os.environ["VLLM_BASE_URL"])
        key = os.environ["VLLM_API_KEY"]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url=base, api_key=key,
                          max_tokens=tokens, timeout=timeout_s, max_retries=0)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=tokens, timeout=timeout_s,
                          max_retries=0)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    from langchain_anthropic import ChatAnthropic
    # Some current Claude models reject non-default temperature; omit it.
    return ChatAnthropic(model=model, max_tokens=tokens, timeout=timeout_s,
                         max_retries=0)
