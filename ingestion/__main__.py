"""CLI: python -m ingestion FILE --tenant demo --source-key partner-a."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .csv_reader import read_csv
from .scraped_reader import read_scraped


def main() -> None:
    """Normalize inputs into JSONL without calling models or databases."""
    parser = argparse.ArgumentParser(description="Normalize feedback from CSV or JSON exports")
    parser.add_argument("file", type=Path)
    parser.add_argument("--tenant", required=True, help="Trusted tenant ID")
    parser.add_argument("--source-key", required=True, help="Stable source identifier")
    parser.add_argument("--output", type=Path, help="JSONL output; default stdout")
    args = parser.parse_args()

    suffix = args.file.suffix.lower()
    if suffix == ".csv":
        records = read_csv(args.file, tenant_id=args.tenant, source=args.source_key)
    elif suffix in (".jsonl", ".ndjson", ".json"):
        records = read_scraped(args.file, tenant_id=args.tenant, source=args.source_key)
    else:
        parser.error("Supported input: .csv, .jsonl, .ndjson, .json")

    count = 0
    if args.output is None:
        for record in records:
            print(record.model_dump_json())
            count += 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=args.output.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                for record in records:
                    stream.write(record.model_dump_json() + "\n")
                    count += 1
            os.replace(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Validated {count} feedback records", file=sys.stderr)


if __name__ == "__main__":
    main()
