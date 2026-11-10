"""Download real BEIR relevance judgments without installing third-party packages.

Usage: python -m scripts.prepare_beir_eval --dataset nfcorpus --limit 100
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {"nfcorpus", "scifact"}
BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"


def prepare(dataset: str, target: Path, limit: int, seed: int = 42) -> dict:
    if dataset not in DATASETS or limit < 1:
        raise ValueError("dataset must be nfcorpus/scifact and limit >= 1")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{dataset}.zip"
    if not archive.exists():
        print(f"Downloading official BEIR {dataset} dataset...", flush=True)
        urllib.request.urlretrieve(f"{BASE_URL}{dataset}.zip", archive)
    with zipfile.ZipFile(archive) as z:
        def read_member(suffix: str) -> str:
            matches = [name for name in z.namelist() if name.endswith("/" + suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one BEIR file named {suffix}, got {matches}")
            return z.read(matches[0]).decode("utf-8")
        corpus_lines = read_member("corpus.jsonl").splitlines()
        query_lines = read_member("queries.jsonl").splitlines()
        qrel_lines = read_member("qrels/test.tsv").splitlines()

    docs = {str(d["_id"]): (d.get("title", "") + "\n" + d.get("text", "")).strip()
            for line in corpus_lines if (d := json.loads(line))}
    queries = {str(q["_id"]): q["text"] for line in query_lines if (q := json.loads(line))}
    qrels: dict[str, dict[str, int]] = {}
    for row in csv.DictReader(qrel_lines, delimiter="\t"):
        qid = row["query-id"]
        docid = row["corpus-id"]
        relevance = int(row["score"])
        if qid in queries and docid in docs:
            qrels.setdefault(qid, {})[docid] = relevance
    eligible = sorted(qid for qid, judgments in qrels.items() if any(v > 0 for v in judgments.values()))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = sorted(eligible[:limit])
    if not selected:
        raise RuntimeError("No usable test questions with relevance judgments")

    folder = target / dataset
    folder.mkdir(exist_ok=True)
    with (folder / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for docid, text in docs.items():
            f.write(json.dumps({"doc_id": docid, "text": text}, ensure_ascii=False) + "\n")
    with (folder / "queries.jsonl").open("w", encoding="utf-8") as f:
        for qid in selected:
            f.write(json.dumps({"query_id": qid, "query": queries[qid]}, ensure_ascii=False) + "\n")
    with (folder / "qrels.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for qid in selected:
            for docid, relevance in sorted(qrels[qid].items()):
                writer.writerow([qid, docid, relevance])
    info = {"dataset": dataset, "queries": len(selected), "documents": len(docs),
            "positive_judgments": sum(v > 0 for qid in selected for v in qrels[qid].values()),
            "folder": str(folder)}
    (folder / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="nfcorpus")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("tests/data/beir_eval"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.out, args.limit), indent=2))

if __name__ == "__main__":
    main()
