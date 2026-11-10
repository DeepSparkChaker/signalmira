"""Evaluate Qwen/BGE reranking on REAL BEIR qrels using a TF-IDF candidate baseline.

This tests reranking, NOT SignalMira PostgreSQL BM25 + pgvector retrieval.
The same candidate pool is used for baseline and reranked metrics.
Usage:
 python -m scripts.run_public_rerank_eval --folder tests/data/beir_eval/nfcorpus \
   --model qwen --revision YOUR_PINNED_QWEN_RERANKER_SHA --device cuda
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
from statistics import median
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.eval_retrieval_rankings import evaluate, load_qrels


def load_jsonl(path: Path, id_key: str, text_key: str):
    return {row[id_key]: row[text_key]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and (row := json.loads(line))}


def run(folder: Path, model_name: str, revision: str | None, device: str, top_n: int,
        final_k: int, batch_size: int, out: Path) -> dict:
    corpus = load_jsonl(folder / "corpus.jsonl", "doc_id", "text")
    queries = load_jsonl(folder / "queries.jsonl", "query_id", "query")
    judgments = load_qrels(folder / "qrels.tsv")
    doc_ids = list(corpus)
    print(f"Indexing {len(doc_ids)} benchmark documents with TF-IDF (NOT pg_textsearch BM25)...", flush=True)
    tfidf = TfidfVectorizer(stop_words="english", max_features=100000, dtype=np.float32)
    matrix = tfidf.fit_transform(corpus.values())
    q_matrix = tfidf.transform(queries.values())
    baseline, reranked = {}, {}
    runner = None
    if model_name != "off":
        if not revision:
            raise ValueError("Pinned 40-char --revision required for Qwen or BGE")
        from retrieval.reranker import Reranker, RerankerConfig
        runner = Reranker(RerankerConfig(model=model_name, revision=revision,
                                        device=device, offline=True,
                                        batch_size=batch_size, timeout_s=120.0))
        print(f"Warming {model_name} model...", flush=True)
        runner.warmup()
    latencies = []
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("w", encoding="utf-8") as f:
            for i, (qid, query) in enumerate(queries.items()):
                scores = np.asarray(q_matrix[i].dot(matrix.T).toarray(), dtype=np.float32).ravel()
                # Stable sort ensures deterministic tie breaks, important for lexical misses.
                best_idx = np.argsort(-scores, kind="stable")[:top_n]
                candidates = [{"chunk_id": doc_ids[int(j)], "chunk_text": corpus[doc_ids[int(j)]]}
                              for j in best_idx]
                baseline[qid] = [item["chunk_id"] for item in candidates[:final_k]]
                if runner:
                    t0 = perf_counter()
                    results = runner.rank(query, candidates, top_k=final_k)
                    latencies.append((perf_counter() - t0) * 1000)
                    reranked[qid] = [item["chunk_id"] for item in results]
                else:
                    reranked[qid] = baseline[qid]
                f.write(json.dumps({"query_id": qid, "doc_ids": reranked[qid]}, ensure_ascii=False) + "\n")
                if (i+1) % 10 == 0:
                    print(f"Evaluated {i+1}/{len(queries)}", flush=True)
    finally:
        if runner:
            runner.close()
    return {"candidate_retriever": "scikit-learn TF-IDF (not SignalMira hybrid)",
            "baseline": evaluate(judgments, baseline, final_k),
            "reranked": evaluate(judgments, reranked, final_k),
            "median_rerank_ms": round(median(latencies), 2) if latencies else None,
            "output_file": str(out)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=Path("tests/data/beir_eval/nfcorpus"))
    p.add_argument("--model", choices=("qwen", "bge", "off"), default="qwen")
    p.add_argument("--revision")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--candidate-k", type=int, default=25)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("tests/data/beir_eval/reranked_results.jsonl"))
    args = p.parse_args()
    if not 1 <= args.top_k <= args.candidate_k <= 25:
        p.error("Require 1 <= top-k <= candidate-k <= 25")
    print(json.dumps(run(args.folder, args.model, args.revision, args.device,
                         args.candidate_k, args.top_k, args.batch_size, args.out), indent=2))

if __name__ == "__main__":
    main()
