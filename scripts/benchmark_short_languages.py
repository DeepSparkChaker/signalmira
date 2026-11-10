"""Compare Lingua's original policy with margin thresholds on a labeled CSV.
Run from the project root:
    uv run python -m scripts.benchmark_short_languages --csv tests/data/language_reviews_short_ar_es_fr_100.csv
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path
from enrichment.language import _engine, _mixed_scripts

MARGINS = (0.12, 0.10, 0.08, 0.05)

def classify(text, scores, *, min_score=0.0, margin=0.12):
    s = text[:12000]
    if sum(ch.isalpha() for ch in s) < 6:
        return "und"
    if _mixed_scripts(s):
        return "mul"
    if not scores:
        return "und"
    top = scores[0].value
    second = scores[1].value if len(scores) > 1 else 0.0
    if top < min_score or top - second < margin:
        return "und"
    return scores[0].language.iso_code_639_1.name.lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='tests/data/language_reviews_short_ar_es_fr_100.csv')
    ap.add_argument('--output', default='short_language_predictions.csv')
    args = ap.parse_args()
    with Path(args.csv).open(encoding='utf-8-sig', newline='') as f:
        rows=list(csv.DictReader(f))
    counts=Counter(r['language'] for r in rows)
    if counts != Counter({'ar':34,'es':33,'fr':33}):
        raise ValueError(f'Expected 34 ar, 33 es, 33 fr; found {counts}')
    detector=_engine()
    columns=['original',*[f'margin_{m:.2f}' for m in MARGINS]]
    for r in rows:
        values=detector.compute_language_confidence_values(r['text'])
        r['original']=classify(r['text'],values,min_score=0.57,margin=0.12)
        for m in MARGINS:
            r[f'margin_{m:.2f}']=classify(r['text'],values,margin=m)
        r['top_candidate']=values[0].language.name if values else ''
        r['top_score']=f'{values[0].value:.4f}' if values else ''
        r['margin']=f'{values[0].value - values[1].value:.4f}' if len(values)>1 else ''
    with Path(args.output).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    print(f'{"Policy":<15} {"Overall":>9} {"Unknown":>9} {"Wrong":>7} {"AR":>8} {"ES":>8} {"FR":>8}')
    for col in columns:
        correct=sum(r[col]==r['language'] for r in rows)
        unknown=sum(r[col] in ('und','mul') for r in rows)
        wrong=len(rows)-correct-unknown
        recalls=[sum(r[col]==lang for r in rows if r['language']==lang)/counts[lang] for lang in ('ar','es','fr')]
        print(f'{col:<15} {correct/len(rows):>8.1%} {unknown:>9} {wrong:>7} '+ ' '.join(f'{v:>8.1%}' for v in recalls))
    print('\nErrors or unknown at margin 0.12:')
    for r in rows:
        if r['margin_0.12'] != r['language']:
            print(f"{r['id']}: expected={r['language']} got={r['margin_0.12']}; {r['text']}")
    print(f'\nDetails saved to {args.output}')

if __name__=='__main__':
    main()
