"""Read-only extraction preview: one review or bounded CSV stream; no DB writes.

uv run python -m scripts.opinion_preview --model qwen3.5:4b \
    --revision "$LLM_REVISION" --text 'Camera great, battery poor.'
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor
from enrichment.opinion_service import analyze_review, analyze_many


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("ollama", "vllm"), default="ollama")
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True,
                   help="Ollama SHA256 digest (64 hex) or vLLM HF SHA (40 hex)")
    p.add_argument("--tenant", default="demo")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--csv", type=Path, help="CSV with review_id,text headers")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    if not 1 <= args.limit <= 100_000:
        p.error("--limit must be 1..100000")
    config = OpinionConfig(model=args.model, model_revision=args.revision,
                           provider=args.provider)
    extractor = LangChainOpinionExtractor(config)
    try:
        extractor.verify_model_digest()
        if args.text is not None:
            result = analyze_review(extractor, tenant=args.tenant,
                                    review_id="preview-1", text=args.text)
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        else:
            csv.field_size_limit(100_000)  # fail before consuming huge CSV cells
            with args.csv.open(encoding="utf-8-sig", newline="") as f:
                rows = csv.DictReader(f)
                if not {"review_id", "text"} <= set(rows.fieldnames or ()):
                    p.error("CSV requires review_id,text columns")
                for result in analyze_many(extractor, tenant=args.tenant,
                                           rows=rows, max_rows=args.limit):
                    print(json.dumps(result.as_dict(), ensure_ascii=False))
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
