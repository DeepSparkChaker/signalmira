
import argparse
import statistics
import time

import torch
from enrichment.ner import Config, NER

TEXT = "My Google Pixel 9 battery is excellent."

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
parser.add_argument("--profile", choices=["fast", "balanced", "accurate", "multilingual"], default="accurate")
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--warmup", type=int, default=5)
args = parser.parse_args()

if args.runs < 1:
    parser.error("--runs must be >= 1")

if args.device == "cuda" and not torch.cuda.is_available():
    parser.error("CUDA unavailable")

def synchronize():
    if args.device == "cuda":
        torch.cuda.synchronize()

ner = NER(Config(
    profile=args.profile,
    device=args.device,
    offline=True
))

# 1. Model loading time
start = time.perf_counter()
ner._load()
synchronize()
load_time = time.perf_counter() - start

# 2. Warm up, excluded from measurements
ner.warmup(count=args.warmup)

# 3. Measure one-review latency
times = []

for _ in range(args.runs):
    synchronize()
    start = time.perf_counter()

    entities = ner.extract(TEXT)

    synchronize()
    times.append((time.perf_counter() - start) * 1000)

ordered = sorted(times)

def percentile(p):
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

print(f"\nModel: {ner.name}")
print(f"Revision: {ner.revision}")
print(f"Device: {ner.device}")
print(f"Load time: {load_time:.2f} s")
print(f"Average: {statistics.mean(times):.2f} ms")
print(f"Median (p50): {statistics.median(times):.2f} ms")
print(f"p95: {percentile(95):.2f} ms")
print(f"p99: {percentile(99):.2f} ms")
print(f"Min: {min(times):.2f} ms")
print(f"Max: {max(times):.2f} ms")
print(f"Entities: {len(entities)}")
