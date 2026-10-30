"""CLI: init database, import normalized feedback, or read back a review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ingestion.csv_reader import read_csv
from ingestion.scraped_reader import read_scraped
from .postgres import connect, count_reviews, create_tables, get_review, upsert_reviews


def main() -> None:
    """Read DATABASE_URL and run one database operation."""
    parser = argparse.ArgumentParser(description="SignalMira feedback PostgreSQL storage")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the feedback_reviews table")

    import_cmd = sub.add_parser("import", help="Import CSV/JSON/JSONL")
    import_cmd.add_argument("file", type=Path)
    import_cmd.add_argument("--tenant", required=True)
    import_cmd.add_argument("--source-key", required=True)
    import_cmd.add_argument("--batch-size", type=int, default=500, help="100-1000 is a good starting range")

    read_cmd = sub.add_parser("get", help="Read one review with all database fields")
    read_cmd.add_argument("--tenant", required=True)
    read_cmd.add_argument("--source-key", required=True)
    read_cmd.add_argument("--review-id", required=True)

    args = parser.parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL before running the storage CLI")
    with connect(database_url) as conn:
        if args.command == "init":
            create_tables(conn)
            print("Created/verified feedback_reviews table")
        elif args.command == "import":
            suffix = args.file.suffix.lower()
            if suffix == ".csv":
                records = read_csv(args.file, tenant_id=args.tenant, source=args.source_key)
            elif suffix in (".json", ".jsonl", ".ndjson"):
                records = read_scraped(args.file, tenant_id=args.tenant, source=args.source_key)
            else:
                parser.error("Supported input: .csv, .json, .jsonl, .ndjson")
            written, unchanged = upsert_reviews(conn, records, batch_size=args.batch_size)
            total = count_reviews(conn, tenant_id=args.tenant, source=args.source_key)
            print(f"Written: {written}; skipped/unchanged: {unchanged}; source total: {total}")
        elif args.command == "get":
            row = get_review(
                conn, tenant_id=args.tenant,
                source=args.source_key, review_id=args.review_id,
            )
            if row is None:
                parser.error("Review not found for this tenant/source/id")
            print(json.dumps(row, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
