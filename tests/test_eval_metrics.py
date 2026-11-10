from scripts.eval_retrieval_rankings import metrics, evaluate

def test_metrics_simple():
    labels = {"a": 2, "b": 1, "c": 0}
    result = metrics(["a", "x", "b"], labels, 3)
    assert result["recall"] == 1.0
    assert result["hit"] == 1
    assert result["mrr"] == 1.0
    assert 0 < result["ndcg"] < 1

def test_ndcg_perfect():
    x = metrics(["a", "b"], {"a": 2, "b": 1}, 2)
    assert x["ndcg"] == 1.0
    assert x["recall"] == 1.0

def test_missing_queries_refused():
    try:
        evaluate({"q": {"d": 1}}, {})
    except ValueError:
        return
    raise AssertionError("must reject missing queries")


def test_small_local_baseline(tmp_path):
    import json
    from scripts.run_public_rerank_eval import run
    folder = tmp_path / "fixture"
    folder.mkdir()
    with (folder / "corpus.jsonl").open("w") as f:
        for d, text in (("1", "battery drains quickly"), ("2", "screen is great"), ("3", "slow battery charging")):
            f.write(json.dumps({"doc_id":d,"text":text}) + "\n")
    (folder / "queries.jsonl").write_text(json.dumps({"query_id":"q1","query":"battery drains"})+"\n")
    (folder / "qrels.tsv").write_text("query_id\tdoc_id\trelevance\nq1\t1\t1\nq1\t2\t0\n")
    report = run(folder, "off", None, "cpu", 3, 2, 8, tmp_path / "out.jsonl")
    assert report["baseline"]["hit@2"] == 1.0
    assert report["baseline"]["recall@2"] == 1.0
