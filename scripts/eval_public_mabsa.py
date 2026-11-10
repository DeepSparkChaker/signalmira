"""Evaluate SignalMira's actual Qwen opinion extractor on public M-ABSA phone test data.

No PostgreSQL, no manual labels, no writes to SignalMira tables.
Public dataset: https://huggingface.co/datasets/Multilingual-NLP/M-ABSA
Scores COARSE aspect family and family+polarity only (not suggestion/problem/overall).

Prepare 150 public samples (30 per language):
  uv run python -m scripts.eval_public_mabsa --prepare-only --per-language 30

Score the existing 4B model:
  uv run python -m scripts.eval_public_mabsa --model "$LLM_MODEL" \
      --revision "$LLM_REVISION" --predictions tests/data/opinion_eval/mabsa_qwen4b.jsonl
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

REPO = "Multilingual-NLP/M-ABSA"
LANGUAGES = ("en", "fr", "es", "ar", "zh")
POLARITIES = frozenset({"positive", "negative", "neutral"})

# Intentional coarse categories. The source's hierarchical aspect categories are
# DIFFERENT from SignalMira's free-text English aspects; this is a proxy benchmark.
CATEGORY_FAMILY = {
    "After-sales Service": "after_sales", "Appearance Design": "appearance",
    "Audio/Sound": "audio", "Battery/Longevity": "battery",
    "Branding/Marketing": "promotion", "Buyer Attitude": "purchase_intent",
    "Camera": "camera", "Ease of Use": "usability",
    "Key Design": "hardware_design", "Logistics": "delivery",
    "Overall": "overall", "Performance": "performance", "Price": "price",
    "Product Accessories": "accessories", "Product Configuration": "memory_storage",
    "Product Packaging": "packaging", "Product Quality": "product_quality",
    "Screen": "screen", "Seller Service": "customer_support",
    "Shooting Functions": "camera", "Signal": "connectivity",
    "Smart Connect": "connectivity", "System": "software",
    "Application": "software", "Network": "connectivity",
}

# Rule-based, transparent alias mapping, not an LLM judge. Review this mapping
# before trusting any cross-system comparison; unrecognized predictions = FP.
ALIASES = {
    "battery": "battery", "battery_life": "battery", "charging": "battery",
    "battery_charging": "battery", "power": "battery",
    "camera": "camera", "photo": "camera", "photography": "camera",
    "photo_quality": "camera", "shooting_functions": "camera",
    "screen": "screen", "display": "screen", "screen_quality": "screen",
    "delivery": "delivery", "shipping": "delivery", "logistics": "delivery",
    "service_delivery": "delivery",
    "bluetooth": "connectivity", "bluetooth_connection": "connectivity",
    "connectivity": "connectivity", "wifi": "connectivity",
    "wi_fi": "connectivity", "network": "connectivity", "signal": "connectivity",
    "speaker": "audio", "headphone_sound": "audio", "sound": "audio",
    "sound_quality": "audio", "audio": "audio", "volume": "audio",
    "price": "price", "pricing": "price", "cost": "price",
    "value_for_money": "price", "value": "price",
    "product": "overall", "phone": "overall", "device": "overall",
    "overall": "overall", "overall_experience": "overall",
    "app": "software", "application": "software", "software": "software",
    "operating_system": "software", "system": "software",
    "android": "software", "firmware": "software",
    "support": "customer_support", "customer_support": "customer_support",
    "customer_service": "customer_support", "seller": "customer_support",
    "seller_service": "customer_support",
    "ram": "memory_storage", "memory": "memory_storage",
    "storage": "memory_storage", "product_configuration": "memory_storage",
    "performance": "performance", "running_speed": "performance",
    "overheating": "performance", "speed": "performance",
    "processor": "performance", "lag": "performance",
    "design": "appearance", "appearance": "appearance",
    "color": "appearance", "colour": "appearance",
    "weight": "appearance", "aesthetics": "appearance",
    "product_quality": "product_quality", "durability": "product_quality",
    "quality": "product_quality", "build_quality": "product_quality",
    "accessories": "accessories", "charger": "accessories",
    "case": "accessories", "headphones": "accessories",
    "screen_protector": "accessories",
    "packaging": "packaging", "box": "packaging",
    "after_sales": "after_sales", "warranty": "after_sales",
    "return": "after_sales", "refund": "after_sales",
    "promotion": "promotion", "gift": "promotion", "free_gift": "promotion",
    "repurchase": "purchase_intent", "recommendation": "purchase_intent",
    "purchase_intent": "purchase_intent", "recommend": "purchase_intent",
    "usability": "usability", "ease_of_use": "usability",
    "fingerprint": "hardware_design", "buttons": "hardware_design",
    "hardware_design": "hardware_design",
}


def aspect_family(value: str) -> str:
    """Normalize a model aspect to our explicit coarse categories, else count as FP."""
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if key in CATEGORY_FAMILY.values():
        return key
    return ALIASES.get(key, "UNMAPPED:" + key)


def parse_line(line: str) -> tuple[str, list[list[str]]] | None:
    """Read one published M-ABSA sentence + triplets line, no eval/execution."""
    if "####" not in line:
        return None
    text, raw = line.rsplit("####", 1)
    text = text.strip()
    if not text:
        return None
    try:
        triples = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(triples, list):
        return None
    if not all(isinstance(t, (tuple, list)) and len(t) == 3 and
               all(isinstance(x, str) for x in t) for t in triples):
        return None
    return text, triples


def gold_pairs(triples: list[list[str]]) -> set[tuple[str, str]] | None:
    pairs: set[tuple[str, str]] = set()
    for _, category, polarity in triples:
        family = CATEGORY_FAMILY.get(category.split("#", 1)[0])
        if family is None or polarity.lower() not in POLARITIES:
            return None  # never mark an unknown human label as absent
        pairs.add((family, polarity.lower()))
    return pairs


def prepare_sample(per_language: int, seed: int, out: Path,
                   dataset_revision: str | None = None) -> tuple[list[dict], dict]:
    from huggingface_hub import HfApi, hf_hub_download
    if not (1 <= per_language <= 1000):
        raise ValueError("per_language must be 1..1000")
    revision = dataset_revision or HfApi().repo_info(repo_id=REPO, repo_type="dataset").sha
    rng = random.Random(seed)
    selected: list[dict] = []
    stats = {}
    for lang in LANGUAGES:
        path = hf_hub_download(repo_id=REPO, repo_type="dataset",
                               filename=f"phone/{lang}/test.txt", revision=revision)
        candidates = []
        rejected = Counter()
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                parsed = parse_line(line)
                if parsed is None:
                    rejected["invalid_line"] += 1
                    continue
                review, triples = parsed
                if not 10 <= len(review) <= 1800:
                    rejected["length"] += 1
                    continue
                if not (1 <= len(triples) <= 16):
                    rejected["triplets_count"] += 1
                    continue
                family_pairs = gold_pairs(triples)
                if not family_pairs or len(family_pairs) > 16:
                    rejected["unmapped_or_empty_gold"] += 1
                    continue
                candidates.append({
                    "id": f"mabsa-phone-{lang}-{lineno}", "language": lang,
                    "text": review,
                    "input_hash": hashlib.sha256(review.encode("utf-8")).hexdigest(),
                    "gold": [{"aspect": a, "sentiment": p}
                             for a, p in sorted(family_pairs)],
                    "gold_categories": [t[1] for t in triples],
                })
        if len(candidates) < per_language:
            raise RuntimeError(f"Only {len(candidates)} valid {lang} examples for requested {per_language}")
        chosen = rng.sample(candidates, per_language)
        selected.extend(chosen)
        stats[lang] = {"eligible": len(candidates), "selected": len(chosen),
                       "excluded": dict(rejected)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"source": REPO, "revision": revision, "domain": "phone",
            "split": "test", "count": len(selected), "languages": stats,
            "metric_scope": "coarse_category_family_and_sentiment_ONLY",
            "not_evaluated": ["overall_sentiment", "reported_problem", "customer_suggestion"]}
    out.with_suffix(".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected, meta


def load_gold(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty public gold sample")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate gold IDs")
    for r in rows:
        if hashlib.sha256(r["text"].encode("utf-8")).hexdigest() != r["input_hash"]:
            raise ValueError("Input text changed: " + r["id"])
    return rows


def to_pred_pairs(opinions: list[dict]) -> set[tuple[str, str]]:
    return {(aspect_family(str(p.get("aspect", ""))), str(p.get("sentiment", "")))
            for p in opinions if isinstance(p, dict)}


def _metrics(count: Counter) -> dict:
    tp, fp, fn = (count[x] for x in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4)
            if precision + recall else 0.0}


def score(rows: list[dict], predictions: dict[str, dict]) -> dict:
    buckets: dict[str, dict[str, Counter]] = {"all": {"aspect": Counter(), "joint": Counter()}}
    ok, latencies = 0, []
    for row in rows:
        lang = row["language"]
        buckets.setdefault(lang, {"aspect": Counter(), "joint": Counter()})
        pred = predictions.get(row["id"], {})
        if pred.get("input_hash") and pred["input_hash"] != row["input_hash"]:
            raise ValueError("Input mismatch for " + row["id"])
        success = pred.get("status") == "ok"
        pairs = to_pred_pairs(pred.get("opinions", [])) if success else set()
        if success:
            ok += 1
            if isinstance(pred.get("latency_ms"), (float, int)):
                latencies.append(pred["latency_ms"])
        gold = {(a["aspect"], a["sentiment"]) for a in row["gold"]}
        ga, pa = {a for a, _ in gold}, {a for a, _ in pairs}
        for key in ("all", lang):
            c = buckets[key]["aspect"]
            c.update(tp=len(ga & pa), fp=len(pa - ga), fn=len(ga - pa))
            c = buckets[key]["joint"]
            c.update(tp=len(gold & pairs), fp=len(pairs - gold), fn=len(gold - pairs))
    lats = sorted(latencies)
    return {"source": REPO, "reviews": len(rows), "inference_success": ok,
            "inference_failures": len(rows) - ok,
            "successful_reviews_fraction": round(ok / len(rows), 4),
            "aspect_family_coarse": {k: _metrics(v["aspect"]) for k, v in buckets.items()},
            "aspect_plus_polarity_coarse": {k: _metrics(v["joint"]) for k, v in buckets.items()},
            "latency_ms": {"p50": round(statistics.median(lats), 1) if lats else None,
                           "p95": round(lats[math.ceil(0.95 * len(lats)) - 1], 1) if lats else None},
            "limitations": [
                "A COARSE family proxy: M-ABSA labels are hierarchical, SignalMira labels are free text.",
                "Duplicates within one category/polarity are collapsed to one; not exact aspect-triplet F1.",
                "Input text may be machine-translated and human-reviewed; not all are naturally authored reviews.",
                "Does not score customer suggestions, problems, or overall sentiment: no gold labels.",
            ]}


def evaluate(rows: list[dict], model: str, revision: str, output: Path) -> dict:
    from enrichment.opinions import OpinionConfig
    from enrichment.langchain_opinions import LangChainOpinionExtractor
    config = OpinionConfig(model=model, model_revision=revision, provider="ollama")
    extractor = LangChainOpinionExtractor(config)
    predictions: dict[str, dict] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        extractor.verify_model_digest()
        extractor.warmup()
        with output.open("w", encoding="utf-8") as fh:
            for index, row in enumerate(rows, 1):
                started = time.perf_counter()
                pred = {"id": row["id"], "language": row["language"],
                        "input_hash": row["input_hash"],
                        "model": model, "revision": revision,
                        "config_hash": config.config_hash}
                try:
                    obj = extractor.extract(row["text"])
                    pred.update(status="ok", opinions=[o.model_dump(mode="json") for o in obj.opinions],
                                overall_sentiment=obj.overall_sentiment)
                except Exception as exc:
                    pred.update(status="error", error_type=type(exc).__name__, error=str(exc)[:240])
                pred["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fh.flush()
                predictions[row["id"]] = pred
                if index % 10 == 0:
                    print(f"Evaluated {index}/{len(rows)}", flush=True)
    finally:
        extractor.close()
    return score(rows, predictions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true", help="Download and sample published gold data")
    p.add_argument("--per-language", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-revision", help="Optional immutable Hugging Face dataset SHA")
    p.add_argument("--gold", type=Path, default=Path("tests/data/opinion_eval/mabsa_phone_150.jsonl"))
    p.add_argument("--model", help="Installed Ollama model (for scoring phase)")
    p.add_argument("--revision", help="Exact 64-hex Ollama model digest")
    p.add_argument("--predictions", type=Path, default=Path("tests/data/opinion_eval/mabsa_qwen4b.jsonl"))
    p.add_argument("--rescore", action="store_true", help="Use existing predictions; no inference")
    args = p.parse_args()
    if args.prepare_only:
        _, meta = prepare_sample(args.per_language, args.seed, args.gold, args.dataset_revision)
        print(json.dumps({"gold_file": str(args.gold), **meta}, ensure_ascii=False, indent=2))
        return
    rows = load_gold(args.gold)
    if args.rescore:
        predictions = {}
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pred = json.loads(line)
                if pred["id"] in predictions:
                    raise ValueError("Duplicate prediction ID")
                predictions[pred["id"]] = pred
        metrics = score(rows, predictions)
    else:
        if not args.model or not args.revision:
            p.error("Specify --model and --revision, or use --prepare-only / --rescore")
        metrics = evaluate(rows, args.model, args.revision, args.predictions)
    metrics["gold_file"] = str(args.gold)
    metrics["predictions_file"] = str(args.predictions)
    summary = args.predictions.with_suffix(".summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
