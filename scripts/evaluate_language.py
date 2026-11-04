"""Evaluate actual Lingua accuracy, coverage, and speed on labeled JSONL.

Each input line: {"text":"...", "language":"en|es|fr|ar|zh|..."}
Example: uv run python scripts/evaluate_language.py data/language_gold.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from enrichment.language import detect_text


def evaluate(path: Path):
    lines=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        raise ValueError('No labeled rows')
    rows=[]
    # Model preparation time is excluded; report it separately.
    first=time.perf_counter()
    detect_text('The account balance and the application payment failed.')
    startup_ms=(time.perf_counter()-first)*1000
    for row in lines:
        begin=time.perf_counter()
        result=detect_text(row['text'])
        ms=(time.perf_counter()-begin)*1000
        rows.append((row['language'],result.language,ms))
    good=sum(expected==got for expected,got,_ in rows)
    unknown=sum(got in ('und','mul') for _,got,_ in rows)
    durations=[d for _,_,d in rows]
    by_lang={lang:{'n':sum(e==lang for e,_,_ in rows),
                   'correct':sum(e==lang and g==lang for e,g,_ in rows)}
             for lang in sorted({e for e,_,_ in rows})}
    errors=Counter((e,g) for e,g,_ in rows if e!=g)
    print(json.dumps(dict(n=len(rows),accuracy=round(good/len(rows),4),
                          unknown_rate=round(unknown/len(rows),4),
                          startup_ms=round(startup_ms,1),
                          median_ms=round(statistics.median(durations),2),
                          p95_ms=round(sorted(durations)[int((len(durations)-1)*.95)],2),
                          by_language=by_lang,
                          confusions={f'{a}->{b}': n for (a,b),n in errors.items()}),
                     ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python scripts/evaluate_language.py GOLD.jsonl')
    evaluate(Path(sys.argv[1]))
