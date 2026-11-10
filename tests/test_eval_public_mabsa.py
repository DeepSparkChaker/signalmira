"""Offline tests; no HF, PostgreSQL, Ollama or LangChain network required."""
from scripts.eval_public_mabsa import parse_line, gold_pairs, aspect_family, score


def test_parse_public_format():
    row = parse_line("Good camera and battery####[['camera','Camera#General','positive'], ['battery','Battery/Longevity#Battery Life','positive']]")
    assert row and row[0] == "Good camera and battery"
    assert gold_pairs(row[1]) == {("camera", "positive"), ("battery", "positive")}


def test_reject_unknown_gold():
    assert gold_pairs([["x", "Unrecognized#A", "positive"]]) is None


def test_no_arbitrary_code_execution():
    assert parse_line("abc####__import__('os').system('echo surprise')") is None


def test_aliases():
    assert aspect_family("battery_life") == "battery"
    assert aspect_family("Camera") == "camera"
    assert aspect_family("Very Unknown Topic") == "UNMAPPED:very_unknown_topic"


def test_metrics_and_failed_inference():
    rows = [
        {"id": "a", "language": "es", "input_hash": "x", "gold": [
            {"aspect": "battery", "sentiment": "negative"},
            {"aspect": "camera", "sentiment": "positive"}]},
        {"id": "b", "language": "ar", "input_hash": "y", "gold": [
            {"aspect": "delivery", "sentiment": "positive"}]},
    ]
    preds = {"a": {"status": "ok", "input_hash": "x", "latency_ms": 10,
                   "opinions": [{"aspect": "battery_life", "sentiment": "negative"},
                                {"aspect": "camera", "sentiment": "negative"}]},
             "b": {"status": "error", "input_hash": "y"}}
    result = score(rows, preds)
    assert result["aspect_family_coarse"]["all"]["tp"] == 2
    assert result["aspect_family_coarse"]["all"]["fn"] == 1
    assert result["aspect_plus_polarity_coarse"]["all"]["tp"] == 1
    assert result["aspect_plus_polarity_coarse"]["all"]["fp"] == 1
    assert result["aspect_plus_polarity_coarse"]["all"]["fn"] == 2
    assert result["inference_failures"] == 1
