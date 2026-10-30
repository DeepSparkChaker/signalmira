
import argparse
import time

import torch

from enrichment.ner import Config, NER, connect

parser = argparse.ArgumentParser()
parser.add_argument("--tenant", default="demo")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--profile", choices=["balanced", "accurate", "multilingual"], default="accurate")
args = parser.parse_args()

# 1. READ ONLY: fetch 32 reviews
with connect() as conn:
    rows = conn.execute(
        """
        SELECT review_id, text
        FROM feedback_reviews
        WHERE tenant_id = %s
        ORDER BY source, review_id
        LIMIT 32
        """,
        (args.tenant,)
    ).fetchall()

texts = [row["text"] for row in rows]

if not texts:
    raise SystemExit("No reviews found for this tenant.")

print(f"Loaded {len(texts)} reviews from PostgreSQL")

# 2. Load the same model only once
ner = NER(Config(
    profile=args.profile,
    device=args.device,
    offline=True,
))

start = time.perf_counter()
ner.warmup(count=5)
print(f"Load + warmup: {time.perf_counter() - start:.2f} s")

# 3. Benchmark different inference batch sizes
for batch_size in (8, 16, 32):
    # Change configuration for this test
    ner.config = Config(
        profile=args.profile,
        device=args.device,
        batch_size=batch_size,
        offline=True,
    )

    # Warm up this batch size
    list(ner.extract_many(texts))

    if args.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    results = list(ner.extract_many(texts))

    if args.device == "cuda":
        torch.cuda.synchronize()

    seconds = time.perf_counter() - start

    print(
        f"Batch {batch_size:2d} | "
        f"{len(texts) / seconds:.1f} reviews/s | "
        f"{seconds * 1000 / len(texts):.1f} ms/review | "
        f"{sum(map(len, results))} entities"
    )
