"""Convert UCI's labelled Amazon sentences ZIP into standard feedback CSV."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


def main() -> None:
    """Write source IDs and preserve published 0/1 labels for later evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="UCI sentiment labelled sentences ZIP")
    parser.add_argument("output", type=Path, help="Destination CSV")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        name = next((n for n in archive.namelist() if n.endswith("amazon_cells_labelled.txt")), None)
        if not name:
            raise ValueError("Amazon labelled sentences not found in UCI ZIP")
        with archive.open(name) as data:
            reader = csv.reader(io.TextIOWrapper(data, encoding="utf-8"), delimiter="\t")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=["review_id", "text", "label"])
                writer.writeheader()
                count = 0
                for count, fields in enumerate(reader, 1):
                    if len(fields) != 2 or fields[1] not in ("0", "1"):
                        raise ValueError(f"Unexpected UCI data at row {count}")
                    writer.writerow({"review_id": f"uci-amazon-{count:04d}", "text": fields[0], "label": fields[1]})
    print(f"Prepared {count} public review sentences in {args.output}")


if __name__ == "__main__":
    main()
