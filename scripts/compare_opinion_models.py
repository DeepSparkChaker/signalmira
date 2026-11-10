"""Compare Qwen3.5-4B and 9B on exactly the SAME human-labeled reviews.

  uv run python -m scripts.compare_opinion_models \\
    --gold tests/data/opinion_eval/real_reviews_150.jsonl --split dev \\
    --a tests/data/opinion_eval/qwen4b_dev.jsonl \\
    --b tests/data/opinion_eval/qwen9b_dev.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.eval_opinion_gold import load_gold, load_predictions, calculate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    args = p.parse_args()
    rows = load_gold(args.gold, args.split)
    expected = {row["id"] for row in rows}
    a, b = load_predictions(args.a), load_predictions(args.b)
    if not expected <= a.keys() or not expected <= b.keys():
        raise ValueError("Both model runs must cover every labeled review in the same split")
    first, second = calculate(rows, a), calculate(rows, b)
    items = [
        ("Valid output rate", "structured_output_success_rate"),
        ("Overall macro F1", "overall_sentiment", "macro_f1"),
        ("Aspect F1", "aspect_extraction", "f1"),
        ("Aspect+polarity F1", "aspect_plus_polarity", "f1"),
        ("Problem recall (presence)", "reported_problem_presence", "recall"),
        ("Suggestion precision (presence)", "customer_suggestion_presence", "precision"),
        ("Suggestion recall (presence)", "customer_suggestion_presence", "recall"),
        ("Latency p50 ms", "latency_ms", "p50"),
        ("Latency p95 ms", "latency_ms", "p95"),
    ]
    print(f"Same {len(rows)} gold reviews | split={args.split} | A={args.a.name} | B={args.b.name}")
    print(f"{'Metric':32} {'A':>10} {'B':>10} {'B-A':>10}")
    print("-" * 66)
    for label, *path in items:
        def get(record):
            for key in path:
                record = record[key]
            return record
        va, vb = get(first), get(second)
        if va is None or vb is None:
            print(f"{label:32} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
        else:
            print(f"{label:32} {va:>10.4f} {vb:>10.4f} {vb - va:>+10.4f}")
    print("Presence metrics are NOT semantic correctness of problem text or suggestion wording.")


if __name__ == "__main__":
    main()
