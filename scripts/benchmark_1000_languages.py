"""Benchmark language-detection acceptance thresholds on labeled reviews.

Run from SignalMira root:
  uv run python -m scripts.benchmark_1000_languages --csv tests/data/language_reviews_1000.csv

No database writes. Uses the same Lingua language candidate scoring as the app.
Treat this synthetic set as a diagnostic, not production validation.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

POLICIES = {
    'original_0.57_0.12': (0.57, 0.12),
    'margin_0.12': (0.0, 0.12),
    'margin_0.10': (0.0, 0.10),
    'margin_0.05': (0.0, 0.05),
}
LANGS = ('ar', 'en', 'fr', 'es')
GROUPS = ('very_short', 'short', 'medium', 'long', 'very_long')


def predict(candidates, text, score_min, margin_min):
    if sum(ch.isalpha() for ch in text) < 6 or not candidates:
        return 'und'
    top_score = candidates[0].value
    runner_score = candidates[1].value if len(candidates) > 1 else 0.0
    if top_score < score_min or top_score-runner_score < margin_min:
        return 'und'
    return candidates[0].language.iso_code_639_1.name.lower()


def metrics(rows, key):
    n=len(rows)
    correct=sum(r[key]==r['language'] for r in rows)
    unknown=sum(r[key] in ('und','mul') for r in rows)
    wrong=n-correct-unknown
    coverage=(n-unknown)/n if n else 0
    precision=correct/(n-unknown) if n>unknown else 0
    return correct,n,unknown,wrong,coverage,precision


def print_table(rows, name):
    print(f'\n{name} | n={len(rows)}')
    print(f"{'Policy':<23} {'Acc':>7} {'Unknown':>8} {'Wrong':>7} {'Coverage':>9} {'LabeledPrec':>11}")
    for key in POLICIES:
        c,n,u,w,cov,prec=metrics(rows,key)
        print(f'{key:<23} {100*c/n if n else 0:>6.1f}% {u:>8} {w:>7} {100*cov:>8.1f}% {100*prec:>10.1f}%')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='tests/data/language_reviews_1000.csv')
    p.add_argument('--output', default='language_predictions_1000.csv')
    p.add_argument('--show-errors',type=int,default=12)
    args=p.parse_args()
    from lingua import LanguageDetectorBuilder
    detector=LanguageDetectorBuilder.from_all_spoken_languages().build()
    with Path(args.csv).open(encoding='utf-8-sig', newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty CSV')
    expected_cols={'id','language','length_group','char_length','text'}
    if not expected_cols.issubset(rows[0]):
        raise ValueError(f'CSV requires: {expected_cols}')
    if len(rows)!=1000 or Counter(r['language'] for r in rows)!={x:250 for x in LANGS}:
        raise ValueError('Expected 1000 reviews, 250 per language')
    if len({r['text'] for r in rows})!=len(rows):
        raise ValueError('Reviews must have distinct text')
    for i,r in enumerate(rows,1):
        scores=detector.compute_language_confidence_values(r['text'])
        r['top_language']=scores[0].language.iso_code_639_1.name.lower() if scores else ''
        r['top_score']=f'{scores[0].value:.5f}' if scores else ''
        r['runner_up']=scores[1].language.iso_code_639_1.name.lower() if len(scores)>1 else ''
        r['margin']=f'{scores[0].value - scores[1].value:.5f}' if len(scores)>1 else ''
        for name, (min_score,min_margin) in POLICIES.items():
            r[name]=predict(scores,r['text'],min_score,min_margin)
        if i%250==0:
            print(f'Scored {i}/{len(rows)}...')
    with Path(args.output).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print_table(rows,'OVERALL')
    for lang in LANGS:
        print_table([r for r in rows if r['language']==lang],f'LANGUAGE {lang} (recall = Acc)')
    for group in GROUPS:
        print_table([r for r in rows if r['length_group']==group],f'LENGTH {group}')
    for key in ('margin_0.12','margin_0.10','margin_0.05'):
        mistakes=[r for r in rows if r[key] != r['language']]
        wrong=[r for r in mistakes if r[key] not in ('und','mul')]
        print(f'\n{key}: {len(wrong)} WRONG LABELS, {len(mistakes)-len(wrong)} UNKNOWN')
        for r in wrong[:args.show_errors]:
            print(f"  WRONG {r['id']} expected={r['language']} got={r[key]} top={r['top_score']} margin={r['margin']}: {r['text'][:100]}")
        for r in (r for r in mistakes if r[key] in ('und','mul')):
            if args.show_errors<=0: break
            print(f"  UNKNOWN {r['id']} expected={r['language']} top={r['top_language']} margin={r['margin']}: {r['text'][:100]}")
            args.show_errors -= 1
        args.show_errors=12
    print('\nPredictions saved:', args.output)
    print('Note: accuracy with unknowns counted incorrect; labeled precision excludes und/mul.')

if __name__=='__main__': main()
