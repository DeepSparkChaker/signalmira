"""Evaluate ranked doc IDs against official BEIR judgments.

Input JSONL, one row per query:
{"query_id":"123","doc_ids":["d1","d2",...]}

Usage: python -m scripts.eval_retrieval_rankings --qrels tests/data/beir_eval/nfcorpus/qrels.tsv --runs output.jsonl
Unjudged documents are treated as irrelevant; BEIR qrels can be incomplete.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            result.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return result


def metrics(ranked: list[str], judged: dict[str, int], k: int = 8) -> dict[str, float]:
    found = set()
    unique = []
    for docid in ranked:
        if docid not in found:
            found.add(docid)
            unique.append(docid)
    ids = unique[:k]
    relevant = {docid for docid, rel in judged.items() if rel > 0}
    hits = sum(docid in relevant for docid in ids)
    dcg = sum((2**judged.get(docid, 0) - 1) / math.log2(rank + 2)
              for rank, docid in enumerate(ids))
    ideal = sorted((v for v in judged.values() if v > 0), reverse=True)[:k]
    idcg = sum((2**v - 1) / math.log2(rank + 2) for rank, v in enumerate(ideal))
    rr = next((1 / (rank + 1) for rank, docid in enumerate(ids) if docid in relevant), 0)
    return {"recall": hits / len(relevant) if relevant else 0.0,
            "ndcg": dcg / idcg if idcg else 0.0,
            "hit": float(hits > 0), "mrr": rr, "precision": hits / k}


def evaluate(qrels: dict[str, dict[str, int]], runs: dict[str, list[str]], k: int = 8) -> dict:
    missing = sorted(set(qrels) - set(runs))
    if missing:
        raise ValueError(f"Missing {len(missing)} queries in run; first: {missing[:5]}")
    rows = [metrics(runs[qid], labels, k) for qid, labels in qrels.items()]
    return {"queries": len(rows), "k": k, **{
        f"{metric}@{k}": round(sum(row[metric] for row in rows) / len(rows), 4)
        for metric in ("recall", "ndcg", "hit", "mrr", "precision")}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()
    if args.k < 1 or args.k > 100:
        p.error("--k must be 1..100")
    rankings = {}
    with args.runs.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qid, ids = str(item["query_id"]), item["doc_ids"]
                if qid in rankings or not isinstance(ids, list):
                    raise ValueError("Repeated query or invalid doc_ids list")
                rankings[qid] = [str(x) for x in ids]
    print(json.dumps(evaluate(load_qrels(args.qrels), rankings, args.k), indent=2))

if __name__ == "__main__":
    main()
