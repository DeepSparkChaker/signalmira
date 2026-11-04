"""Prefetch and verify models BEFORE launching Streamlit. No inference / GPU needed.

Example: uv run --extra ner --extra topics python -m frontend.prepare_models \\
           --download --ner balanced --tenant demo
"""
from __future__ import annotations

import argparse
import os


def ner_ready(profile: str) -> tuple[bool, str]:
    """Check exact local artifacts; never download on a page render."""
    from enrichment.ner import MODELS
    try:
        if profile == "fast":
            import spacy
            spacy.load(MODELS[profile], disable=["parser", "lemmatizer", "tagger"])
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=MODELS[profile], local_files_only=True)
        return True, "cached"
    except Exception as exc:
        return False, str(exc).splitlines()[0]


def prepare(profile: str, tenant: str, *, download: bool) -> None:
    """Fetch selected NER weights and the tenant's active/pending topic checkpoint."""
    from enrichment.ner import MODELS
    if profile == "fast":
        if download:
            from spacy.cli import download as spacy_download
            spacy_download(MODELS[profile])
    elif download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODELS[profile], local_files_only=False)
    ok, reason = ner_ready(profile)
    if not ok:
        raise RuntimeError(f"NER {profile} is not ready: {reason}")
    print(f"NER ready: {MODELS[profile]}")

    # Topic checkpoint is tenant-dependent; no taxonomy = no topic model to prefetch.
    from frontend.service import connect, guard, _tables, _taxonomy
    guard(tenant)
    with connect() as conn:
        choice = _taxonomy(conn, tenant, _tables(conn))
        if choice is None:
            print("Topic: no tenant taxonomy yet (preview will show unavailable)")
            return
        rev = conn.execute("""SELECT model_revision FROM sm_topic_taxonomies
            WHERE tenant_id=%s AND taxonomy_id=%s""",
            (tenant, choice["taxonomy_id"])).fetchone()["model_revision"]
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=choice["model_name"], revision=rev,
                      local_files_only=not download)
    print(f'Topic ready: {choice["model_name"]} @ {rev}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ner", choices=["fast", "balanced", "accurate", "multilingual"],
                   default="balanced")
    p.add_argument("--tenant", default="demo")
    p.add_argument("--download", action="store_true", help="Permit initial model download")
    args = p.parse_args()
    if args.download and os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}:
        p.error("Unset HF_HUB_OFFLINE before downloading models")
    prepare(args.ner,args.tenant,download=args.download)
    print("Ready for Streamlit; models will be loaded into RAM on first use.")


if __name__ == "__main__":
    main()
