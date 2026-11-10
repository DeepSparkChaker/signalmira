"""Human annotation CLI for sampled opinion reviews. No model is used.

  uv run python -m scripts.label_opinion_gold \\
      --file tests/data/opinion_eval/real_reviews_150.jsonl --split dev

Each finished review is atomically saved; Ctrl-C resumes safely on next run.
Labels must come from a human reading the ORIGINAL review.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

LABELS = ("positive", "negative", "neutral", "mixed")


def _ask(prompt: str, choices: tuple[str, ...] | None = None) -> str:
    while True:
        value = input(prompt).strip()
        if value.lower() in {"quit", "exit"}:
            raise KeyboardInterrupt
        if choices is None or value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def _bool(label: str) -> bool:
    return _ask(label + " [y/n]: ", ("y", "n")) == "y"


def label_one(item: dict) -> dict:
    text = item["text"]
    print("\n" + "=" * 65)
    print(f"{item['id']} | {item.get('language', 'und')} | {item.get('split', '?')}")
    print("REVIEW: " + text)
    print("Only label EXPLICIT opinions. Do not infer causes or suggestions.")
    overall = _ask("Overall sentiment [positive/negative/neutral/mixed]: ", LABELS)
    opinions: list[dict] = []
    print("Enter aspects in English snake_case; empty aspect finishes this review.")
    while True:
        aspect = _ask(f"Aspect #{len(opinions) + 1} (Enter to finish): ").lower().replace(" ", "_")
        if not aspect:
            break
        if len(opinions) == 16:
            raise ValueError("Maximum 16 opinions per review")
        sentiment = _ask("  Sentiment [positive/negative/neutral/mixed]: ", LABELS)
        problem = _bool("  Does customer explicitly report a problem?")
        suggestion = _bool("  Does customer explicitly suggest/request a change?")
        quote = None
        if suggestion:
            while True:
                quote = _ask("  Exact ORIGINAL-language suggestion quote: ")
                if quote and quote in text:
                    break
                print("Quote must be a nonempty, exact substring of this review.")
        opinion = {"aspect": aspect, "sentiment": sentiment,
                   "reported_problem": problem, "customer_suggestion": suggestion}
        if quote is not None:
            opinion["suggestion_quote"] = quote
        opinions.append(opinion)
    result = {"overall_sentiment": overall, "opinions": opinions}
    print("Gold labels:", json.dumps(result, ensure_ascii=False, indent=2))
    if not _bool("Save this human annotation?"):
        raise KeyboardInterrupt("Not saved; rerun to try this review")
    return result


def atomic_save(path: Path, items: list[dict]) -> None:
    # Write same-directory temporary file then atomically replace; no partial JSONL.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".opinion_labels_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--split", choices=("all", "dev", "test"), default="dev")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = sum(row.get("gold") is not None for row in rows)
    print(f"Loaded {len(rows)} real reviews, {completed} labeled. No LLM will be called.")
    try:
        for row in rows:
            if row.get("gold") is not None:
                continue
            if args.split != "all" and row.get("split") != args.split:
                continue
            row["gold"] = label_one(row)
            atomic_save(args.file, rows)
            completed += 1
            print(f"Saved safely. Total labeled: {completed}/{len(rows)}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Completed annotations were already saved; rerun to resume.")


if __name__ == "__main__":
    main()
