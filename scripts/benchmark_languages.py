"""Benchmark original SignalMira rule vs margin-only candidate on 100 labeled reviews.

Run from project root AFTER copying the new enrichment/language.py:
  uv run python scripts/benchmark_languages.py --csv tests/data/language_reviews_100.csv

This evaluation set is handcrafted to smoke-test short reviews. Confirm labels
with human reviewers and evaluate a separate real-world holdout before shipping.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from enrichment.language import _engine, _mixed_scripts, detect_text

BASE_SCORE = 0.57
BASE_MARGIN = 0.12


def baseline(text, scores):
    """Exactly mirror the former v1.0 policy."""
    if not text.strip() or sum(ch.isalpha() for ch in text[:12000]) < 6:
        return "und"
    if _mixed_scripts(text[:12000]):
        return "mul"
    if not scores:
        return "und"
    first = float(scores[0].value)
    second = float(scores[1].value) if len(scores) > 1 else 0.0
    if first < BASE_SCORE or first - second < BASE_MARGIN:
        return "und"
    return scores[0].language.iso_code_639_1.name.lower()


def summarize(rows, column, order):
    print(f"\n{column} (unknown counts as incorrect)")
    print(f"{'Language':<11} {'Correct':>7} {'Total':>7} {'Recall':>9} {'Unknown':>9}")
    good = known = 0
    for lang in order:
        group = [row for row in rows if row['language'] == lang]
        correct = sum(row[column] == lang for row in group)
        unknown = sum(row[column] in {'und', 'mul'} for row in group)
        good += correct
        known += len(group) - unknown
        print(f"{lang:<11} {correct:>7} {len(group):>7} {correct/len(group):>8.1%} {unknown:>9}")
    n = len(rows)
    wrong = sum(row[column] not in (row['language'], 'und', 'mul') for row in rows)
    print(f"Overall accuracy / micro recall: {good}/{n} = {good/n:.1%}")
    print(f"Coverage (non-und/non-mul): {known}/{n} = {known/n:.1%}")
    print(f"Accuracy among labeled: {good}/{known} = {good/known:.1%}" if known else 'Accuracy among labeled: n/a')
    print(f"Wrong assigned labels: {wrong}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='tests/data/language_reviews_100.csv')
    parser.add_argument('--output', default='language_predictions_100.csv')
    args = parser.parse_args()
    with Path(args.csv).open(encoding='utf-8', newline='') as f:
        records = list(csv.DictReader(f))
    expected = ['ar', 'en', 'fr', 'zh', 'es']
    counts = Counter(row['language'] for row in records)
    if counts != Counter({l: 20 for l in expected}):
        raise ValueError(f'Expected 20 reviews in each language: {counts}')
    detector = _engine()     # Model initialized once
    results = []
    for row in records:
        text = row['text']
        scores = detector.compute_language_confidence_values(text)
        old = baseline(text, scores)
        new = detect_text(text, engine=detector).language  # THE actual proposed module
        first = float(scores[0].value) if scores else 0.0
        runner = float(scores[1].value) if len(scores)>1 else 0.0
        results.append({**row, 'original':old, 'candidate':new,
                        'top_score':f'{first:.4f}', 'margin':f'{first-runner:.4f}',
                        'candidate_correct': int(new==row['language'])})
    with Path(args.output).open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    summarize(results,'original',expected)
    summarize(results,'candidate',expected)
    print(f'\nDetailed predictions: {args.output}')
    print('Errors / unknowns under candidate policy:')
    for r in results:
        if r['candidate'] != r['language']:
            print(f"  {r['id']} expected={r['language']} got={r['candidate']} text={r['text']}")


if __name__ == '__main__':
    main()
