"""Reproducible multilingual opinion evaluation against HUMAN-ANNOTATED reviews.

Run selected model (one at a time; do NOT load 4B and 9B together):
  uv run python -m scripts.eval_opinion_gold --gold tests/data/opinion_eval/real_reviews_150.jsonl \\
      --split dev --model "$LLM_MODEL" --revision "$LLM_REVISION" \\
      --out tests/data/opinion_eval/qwen4b_dev.jsonl

Re-score saved predictions with no model/GPU:
  uv run python -m scripts.eval_opinion_gold --gold tests/data/opinion_eval/real_reviews_150.jsonl \\
      --split dev --predictions tests/data/opinion_eval/qwen4b_dev.jsonl

No database writes. Tests only human-labeled rows. Failures count against recall.
The problem/suggestion metrics measure PRESENCE, not semantic correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("negative", "neutral", "positive", "mixed")
ASPECT_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")


def load_gold(path: Path, split: str = "dev") -> list[dict]:
    selected: list[dict] = []
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["id"]
        if key in ids:
            raise ValueError(f"Duplicate review id: {key}")
        ids.add(key)
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["input_hash"]:
            raise ValueError(f"Review changed after sampling: {key}")
        if split != "all" and row.get("split") != split:
            continue
        gold = row.get("gold")
        if gold is None:
            continue  # NEVER treat unlabeled as negative or empty gold.
        if not isinstance(gold, dict) or gold.get("overall_sentiment") not in LABELS:
            raise ValueError(f"Invalid overall sentiment gold: {key}")
        opinions = gold.get("opinions")
        if not isinstance(opinions, list) or len(opinions) > 16:
            raise ValueError(f"Invalid gold opinion list: {key}")
        for opinion in opinions:
            if not isinstance(opinion, dict) or not isinstance(opinion.get("aspect"), str) or not ASPECT_RE.fullmatch(opinion["aspect"]):
                raise ValueError(f"Gold aspects must be English snake_case: {key}")
            if opinion.get("sentiment") not in LABELS:
                raise ValueError(f"Invalid gold polarity: {key}")
            for field in ("reported_problem", "customer_suggestion"):
                if type(opinion.get(field)) is not bool:
                    raise ValueError(f"Gold {field} must be explicitly true/false: {key}")
            quote = opinion.get("suggestion_quote")
            if quote is not None and (not opinion["customer_suggestion"] or not isinstance(quote, str) or quote not in row["text"]):
                raise ValueError(f"Human suggestion quote not found in review: {key}")
        selected.append(row)
    if not selected:
        raise ValueError(f"No human-annotated rows in split={split!r}. Run scripts.label_opinion_gold first.")
    return selected


def load_predictions(path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["id"]
        if key in results:
            raise ValueError(f"Duplicate prediction {key}")
        results[key] = row
    return results


def _aspect(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def match_opinions(gold: list[dict], pred: list[dict]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Pair by exact aspect+sentiment first, then aspect. One-to-one matching.

    Duplicate-aspect opinions with distinct meanings require human audit.
    """
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[int, int]] = []
    for require_polarity in (True, False):
        for gi, g in enumerate(gold):
            if gi in used_g:
                continue
            eligible = [pi for pi, p in enumerate(pred)
                        if pi not in used_p and _aspect(str(p.get("aspect", ""))) == g["aspect"]
                        and (not require_polarity or p.get("sentiment") == g["sentiment"])]
            if not eligible:
                continue
            # Favor matching the correct suggestion/problem presence when both
            # gold/pred contain several opinions for the same aspect and polarity.
            pi = min(eligible, key=lambda j: (
                (bool(pred[j].get("reported_problem")) != g.get("reported_problem", False))
                + (bool(pred[j].get("customer_suggestion")) != g.get("customer_suggestion", False)), j))
            used_g.add(gi)
            used_p.add(pi)
            matches.append((gi, pi))
    return matches, used_g, used_p


def _prf(counter: Counter) -> dict[str, Any]:
    tp, fp, fn = (int(counter[k]) for k in ("tp", "fp", "fn"))
    if tp + fp + fn == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": None,
                "recall": None, "f1": None}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if recall is not None and precision + recall else
          (0.0 if recall is not None else None))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None}


def _quote(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().strip(" .,!?。！？؛،؛:;\t\n")


def _percentile_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None}
    return {"p50": round(statistics.median(values), 2),
            "p95": round(sorted(values)[math.ceil(0.95 * len(values)) - 1], 2)}


def calculate(rows: list[dict], predictions: dict[str, dict]) -> dict:
    """Score every human-labeled row, including inference and validation failures."""
    aspect = Counter()
    joint = Counter()
    problem = Counter()
    suggestion = Counter()
    polarity_class = {label: Counter() for label in LABELS}
    overall_class = {label: Counter() for label in LABELS}
    matched_aspects = 0
    matched_polarity = 0
    exact_suggestion_match = exact_suggestion_total = 0
    failures = 0
    latencies: list[float] = []
    for row in rows:
        gold = row["gold"]
        candidate = predictions.get(row["id"])
        if candidate is not None and candidate.get("input_hash") != row["input_hash"]:
            raise ValueError(f"Prediction/input hash mismatch: {row['id']}")
        success = bool(candidate is not None and candidate.get("status") == "ok"
                       and isinstance(candidate.get("data"), dict))
        if success:
            data = candidate["data"]
            pred = data.get("opinions", [])
            if not isinstance(pred, list) or not isinstance(data.get("overall_sentiment"), str):
                success = False
            elif any(not isinstance(x, dict) for x in pred):
                success = False
        if not success:
            failures += 1
            pred = []
            predicted_overall = "__error__"
        else:
            predicted_overall = data["overall_sentiment"]
        if success and isinstance(candidate.get("latency_ms"), (int, float)):
            latencies.append(float(candidate["latency_ms"]))
        golden = gold["opinions"]
        pairs, used_g, used_p = match_opinions(golden, pred)
        aspect["tp"] += len(pairs)
        aspect["fp"] += len(pred) - len(used_p)
        aspect["fn"] += len(golden) - len(used_g)
        matched_aspects += len(pairs)
        for gi, pi in pairs:
            g, p = golden[gi], pred[pi]
            if p.get("sentiment") == g["sentiment"]:
                joint["tp"] += 1
                matched_polarity += 1
            else:
                joint["fp"] += 1
                joint["fn"] += 1
            # Conditional label F1: evaluates sentiment only where aspect matches.
            for label in LABELS:
                if g["sentiment"] == label and p.get("sentiment") == label:
                    polarity_class[label]["tp"] += 1
                elif g["sentiment"] == label:
                    polarity_class[label]["fn"] += 1
                elif p.get("sentiment") == label:
                    polarity_class[label]["fp"] += 1
            for field, metric in (("reported_problem", problem), ("customer_suggestion", suggestion)):
                actual = bool(p.get(field))
                expected = g[field]
                if expected and actual:
                    metric["tp"] += 1
                elif actual:
                    metric["fp"] += 1
                elif expected:
                    metric["fn"] += 1
            if g.get("customer_suggestion", False) and g.get("suggestion_quote"):
                exact_suggestion_total += 1
                value = p.get("customer_suggestion")
                quoted = value.get("quote") if isinstance(value, dict) else ""
                if quoted and _quote(quoted) == _quote(g["suggestion_quote"]):
                    exact_suggestion_match += 1
        for gi, g in enumerate(golden):
            if gi not in used_g:
                joint["fn"] += 1
                if g.get("reported_problem", False):
                    problem["fn"] += 1
                if g.get("customer_suggestion", False):
                    suggestion["fn"] += 1
                if g.get("customer_suggestion", False) and g.get("suggestion_quote"):
                    exact_suggestion_total += 1
        for pi, p in enumerate(pred):
            if pi not in used_p:
                joint["fp"] += 1
                if p.get("reported_problem"):
                    problem["fp"] += 1
                if p.get("customer_suggestion"):
                    suggestion["fp"] += 1
        for label in LABELS:
            if gold["overall_sentiment"] == label and predicted_overall == label:
                overall_class[label]["tp"] += 1
            elif gold["overall_sentiment"] == label:
                overall_class[label]["fn"] += 1
            elif predicted_overall == label:
                overall_class[label]["fp"] += 1
    # Macro F1 is over classes represented in gold, not absent labels.
    supported_overall = {r["gold"]["overall_sentiment"] for r in rows}
    supported_polarity = {o["sentiment"] for r in rows for o in r["gold"]["opinions"]}
    overall_macro = statistics.mean(_prf(overall_class[k])["f1"] for k in supported_overall)
    pol_macro = (statistics.mean(_prf(polarity_class[k])["f1"] for k in supported_polarity)
                 if supported_polarity and matched_aspects else None)
    return {
        "reviews": len(rows), "errors": failures,
        "structured_output_success_rate": round((len(rows) - failures) / len(rows), 4),
        "overall_sentiment": {
            "accuracy": round(sum(1 for r in rows if predictions.get(r["id"], {}).get("status") == "ok"
                                  and predictions[r["id"]].get("data", {}).get("overall_sentiment")
                                  == r["gold"]["overall_sentiment"]) / len(rows), 4),
            "macro_f1": round(overall_macro, 4),
            "classes_in_gold": sorted(supported_overall),
            "per_class": {k: _prf(overall_class[k]) for k in LABELS},
        },
        "aspect_extraction": _prf(aspect),
        "aspect_plus_polarity": _prf(joint),
        "aspect_polarity_on_matched_aspects": {
            "matches": matched_aspects,
            "accuracy": round(matched_polarity / matched_aspects, 4) if matched_aspects else None,
            "macro_f1": round(pol_macro, 4) if pol_macro is not None else None,
            "classes_in_gold": sorted(supported_polarity),
        },
        "reported_problem_presence": _prf(problem),
        "customer_suggestion_presence": _prf(suggestion),
        "suggestion_quote_exact_recall": (
            round(exact_suggestion_match / exact_suggestion_total, 4)
            if exact_suggestion_total else None),
        "suggestion_quote_gold_count": exact_suggestion_total,
        "latency_ms": _percentile_ms(latencies),
        "limitations": [
            "Aspect matching uses EXACT English normalized aspect labels; configure a canonical taxonomy before final comparisons.",
            "Problem/suggestion PRF checks presence on matched aspects, not whether free-text descriptions are factually correct.",
            "Duplicate same-aspect opinions require human review; pair matching uses aspect/polarity.",
            "Only human-labeled reviews are scored; inference failures count as missed gold labels.",
            "Macro F1 covers labels present in gold, which can be unstable for small per-language slices.",
        ],
    }


def score(rows: list[dict], predictions: dict[str, dict]) -> dict:
    if not rows:
        raise ValueError("No labeled rows")
    output = {"all": calculate(rows, predictions), "by_language": {}, "by_split": {}}
    for lang in sorted({r.get("language", "und") for r in rows}):
        output["by_language"][lang] = calculate(
            [r for r in rows if r.get("language", "und") == lang], predictions)
    for split in sorted({r.get("split", "unknown") for r in rows}):
        output["by_split"][split] = calculate(
            [r for r in rows if r.get("split", "unknown") == split], predictions)
    return output


def run_model(rows: list[dict], *, model: str, revision: str, provider: str,
              out: Path, resume: bool = False) -> dict:
    """Real opinion-service calls, keeping only one selected model warm."""
    from enrichment.opinions import OpinionConfig
    from enrichment.langchain_opinions import LangChainOpinionExtractor
    from enrichment.opinion_service import analyze_review

    if out.exists() and not resume:
        raise FileExistsError(f"{out} exists. Use --resume or choose a new --out.")
    config = OpinionConfig(model=model, model_revision=revision, provider=provider)
    extractor = LangChainOpinionExtractor(config)
    previous = load_predictions(out) if resume and out.exists() else {}
    for item in previous.values():
        if (item.get("model_id") != model or item.get("model_revision") != revision
                or item.get("config_hash") != extractor.config.config_hash
                or item.get("provider") != provider):
            extractor.close()
            raise ValueError("Cannot resume across differing models/prompts/configurations")
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for r in rows if r["id"] in previous)
    try:
        warm_start = time.perf_counter()
        extractor.verify_model_digest()
        extractor.warmup()
        warmup_ms = round((time.perf_counter() - warm_start) * 1000, 1)
        print(f"Warmup: {warmup_ms} ms; {completed} already evaluated", flush=True)
        with out.open("a" if resume else "x", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                if row["id"] in previous:
                    continue
                started = time.perf_counter()
                record: dict[str, Any] = {
                    "id": row["id"], "input_hash": row["input_hash"],
                    "language": row.get("language", "und"), "split": row.get("split"),
                    "provider": provider, "model_id": model, "model_revision": revision,
                    "config_hash": extractor.config.config_hash,
                }
                try:
                    result = analyze_review(extractor, tenant=row.get("tenant_id", "demo"),
                                            review_id=row["review_id"], text=row["text"])
                    record.update(status="ok", data=result.data.model_dump(mode="json"))
                except Exception as exc:
                    # No untrusted customer text or API secrets in logs.
                    record.update(status="error", error_type=type(exc).__name__)
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                f.flush()  # crash-safe progress; no whole-dataset buffering
                print(f"[{i}/{len(rows)}] {row['id']}: {record['status']} ({record['latency_ms']} ms)", flush=True)
    finally:
        extractor.close()
    return {"warmup_ms": warmup_ms, "model": model, "revision": revision,
            "config_hash": extractor.config.config_hash, "output": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="Exact Ollama model tag, e.g. qwen3.5:4b")
    mode.add_argument("--predictions", type=Path, help="Score saved predictions, no model loads")
    p.add_argument("--revision", help="Exact Ollama 64-character digest, required with --model")
    p.add_argument("--provider", choices=("ollama", "vllm"), default="ollama")
    p.add_argument("--out", type=Path, help="Saved predictions JSONL, required with --model")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    rows = load_gold(args.gold, args.split)
    print(f"Evaluating {len(rows)} HUMAN-labeled reviews from split={args.split}", flush=True)
    extra = {}
    if args.model:
        if not args.revision or not args.out:
            p.error("With --model, provide --revision and --out")
        extra = run_model(rows, model=args.model, revision=args.revision,
                          provider=args.provider, out=args.out, resume=args.resume)
        path = args.out
    else:
        path = args.predictions
    predictions = load_predictions(path)
    report = {"predictions": str(path), "gold": str(args.gold), "split": args.split,
              "run": extra, **score(rows, predictions)}
    summary = Path(str(path) + f".{args.split}.summary.json")
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Compact stdout; full report (including language breakdown) lives in JSON.
    all_scores = report["all"]
    print(json.dumps({"summary": str(summary), "labeled_reviews": all_scores["reviews"],
                      "structured_output_success_rate": all_scores["structured_output_success_rate"],
                      "overall_sentiment_macro_f1": all_scores["overall_sentiment"]["macro_f1"],
                      "aspect_micro_f1": all_scores["aspect_extraction"]["f1"],
                      "aspect_polarity_micro_f1": all_scores["aspect_plus_polarity"]["f1"],
                      "problem_presence_recall": all_scores["reported_problem_presence"]["recall"],
                      "suggestion_presence_precision": all_scores["customer_suggestion_presence"]["precision"],
                      "suggestion_presence_recall": all_scores["customer_suggestion_presence"]["recall"],
                      "p50_p95_ms": all_scores["latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
