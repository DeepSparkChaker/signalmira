"""Read-only preview for one review or a bounded CSV stream.

Run: uv run python -m scripts.opinion_preview --model qwen3.5:4b \\
       --revision "$LLM_REVISION" --text 'Camera great but battery bad.'
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from enrichment.opinions import OpinionConfig
from enrichment.langchain_opinions import LangChainOpinionExtractor
from enrichment.opinion_service import analyze_review, analyze_many


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True, help='Full installed Ollama SHA-256 digest')
    p.add_argument('--tenant', default='demo')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--text')
    g.add_argument('--csv', type=Path, help='CSV with review_id,text columns')
    p.add_argument('--limit', type=int, default=10)
    args = p.parse_args()
    config = OpinionConfig(model=args.model, model_revision=args.revision)
    extractor = LangChainOpinionExtractor(config)
    try:
        extractor.verify_model_digest()
        if args.text is not None:
            result = analyze_review(extractor, tenant=args.tenant,
                                    review_id='preview-1', text=args.text)
            print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        else:
            with args.csv.open(encoding='utf-8-sig', newline='') as file:
                rows = csv.DictReader(file)
                if not {'review_id', 'text'} <= set(rows.fieldnames or ()):
                    p.error('CSV requires review_id and text columns')
                for result in analyze_many(extractor, tenant=args.tenant,
                                           rows=rows, max_rows=args.limit):
                    print(json.dumps(result.as_dict(), ensure_ascii=False))
    finally:
        extractor.close()


if __name__ == '__main__':
    main()
