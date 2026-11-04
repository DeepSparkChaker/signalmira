"""Quality-metric correctness, independent of models and databases."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_semantic.py"
spec = importlib.util.spec_from_file_location("sm_quality", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_recall_mrr_multi_relevance():
    assert mod.scores(["c", "b", "a"], {"a", "b"}) == (1.0, 0.5)
    assert mod.scores(["a"], {"a", "b"}) == (0.5, 1.0)
    assert mod.scores(["c"], {"a", "b"}) == (0.0, 0.0)
    with pytest.raises(ValueError):
        mod.scores([], set())
