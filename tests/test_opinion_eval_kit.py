"""Run: uv run python -m pytest -q tests/test_opinion_eval_kit.py"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_opinion_gold import sample
from scripts.eval_opinion_gold import calculate, load_gold, match_opinions, score


def _row(overall="mixed", opinions=None, lang="en", text="Battery bad, camera good.", name="x:1"):
    return {
        "id": name, "source": "x", "review_id": name.split(":")[-1],
        "language": lang, "split": "dev", "text": text,
        "input_hash": hashlib.sha256(text.encode()).hexdigest(),
        "gold": {"overall_sentiment": overall, "opinions": opinions if opinions is not None else [
            {"aspect": "battery", "sentiment": "negative", "reported_problem": True,
             "customer_suggestion": False},
            {"aspect": "camera", "sentiment": "positive", "reported_problem": False,
             "customer_suggestion": False},
        ]},
    }


def _prediction(row, opinions=None, overall="mixed", status="ok"):
    return {"id": row["id"], "input_hash": row["input_hash"], "status": status,
            "data": {"overall_sentiment": overall, "opinions": opinions if opinions is not None else [
                {"aspect": "battery", "sentiment": "negative", "reported_problem": "battery bad",
                 "customer_suggestion": None},
                {"aspect": "camera", "sentiment": "positive", "reported_problem": None,
                 "customer_suggestion": None},
            ]}, "latency_ms": 123.0}


def test_perfect_predictions():
    row = _row()
    results = calculate([row], {row["id"]: _prediction(row)})
    assert results["aspect_extraction"]["f1"] == 1
    assert results["aspect_plus_polarity"]["f1"] == 1
    assert results["reported_problem_presence"]["recall"] == 1
    assert results["overall_sentiment"]["macro_f1"] == 1
    assert results["customer_suggestion_presence"]["recall"] is None


def test_false_invented_suggestion_counts_as_false_positive():
    row = _row()
    pred = _prediction(row)
    pred["data"]["opinions"][0]["customer_suggestion"] = {"quote": "battery bad"}
    result = calculate([row], {row["id"]: pred})
    assert result["customer_suggestion_presence"]["fp"] == 1
    assert result["customer_suggestion_presence"]["precision"] == 0


def test_failed_inference_counts_as_false_negatives():
    row = _row()
    result = calculate([row], {})
    assert result["errors"] == 1
    assert result["aspect_extraction"]["fn"] == 2
    assert result["aspect_plus_polarity"]["recall"] == 0
    assert result["overall_sentiment"]["accuracy"] == 0


def test_wrong_polarity_is_not_correct_joint():
    row = _row()
    pred = _prediction(row)
    pred["data"]["opinions"][0]["sentiment"] = "positive"
    result = calculate([row], {row["id"]: pred})
    assert result["aspect_extraction"]["recall"] == 1
    assert result["aspect_plus_polarity"]["tp"] == 1
    assert result["aspect_plus_polarity"]["fn"] == 1
    assert result["aspect_polarity_on_matched_aspects"]["accuracy"] == .5


def test_multilingual_sampling_strata_and_split():
    languages = ["en", "fr", "es", "ar", "zh"]
    rows = ({"source": "demo", "review_id": f"{lang}-{i}", "language": lang,
             "text": f"Real review #{i} in {lang}"}
            for lang in languages for i in range(100))
    selected, info = sample(rows, count=150, seed=10)
    assert len(selected) == 150
    assert info["scanned"] == 500
    assert all(info["sample_languages"][lang] == 30 for lang in languages)
    assert sum(r["split"] == "dev" for r in selected) == 30
    assert len({r["id"] for r in selected}) == 150
    assert all(r["gold"] is None for r in selected)


def test_load_gold_skips_unlabeled_and_rejects_edits(tmp_path: Path):
    gold = _row()
    unannotated = dict(_row(name="x:2"), gold=None)
    filename = tmp_path / "gold.jsonl"
    filename.write_text("\n".join(json.dumps(x) for x in [gold, unannotated]) + "\n")
    assert len(load_gold(filename)) == 1
    gold["text"] = "Changed!"
    filename.write_text(json.dumps(gold) + "\n")
    with pytest.raises(ValueError, match="Review changed"):
        load_gold(filename)


def test_split_scoring_isolated():
    first = _row(name="x:1", lang="en")
    second = _row(name="x:2", lang="fr")
    out = score([first, second], {first["id"]: _prediction(first)})
    assert out["by_language"]["en"]["aspect_extraction"]["f1"] == 1
    assert out["by_language"]["fr"]["aspect_extraction"]["recall"] == 0


def test_duplicate_aspects_are_one_to_one():
    gold = [{"aspect": "battery", "sentiment": "positive"},
            {"aspect": "battery", "sentiment": "negative"}]
    pred = [{"aspect": "battery", "sentiment": "negative"}]
    pairs, used_g, used_p = match_opinions(gold, pred)
    assert pairs == [(1, 0)]
    assert used_g == {1} and used_p == {0}
