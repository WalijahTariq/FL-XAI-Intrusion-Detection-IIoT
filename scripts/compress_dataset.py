"""Compress data/ML-EdgeIIoT-dataset.csv to data/ML-EdgeIIoT-dataset.csv.gz.

GitHub rejects any single file larger than 100 MB, so the dataset is committed
in gzip form. pandas reads the .gz file directly; nothing else changes.

Usage (from the repository root):
    python scripts/compress_dataset.py
"""
from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ML-EdgeIIoT-dataset.csv"
DST = ROOT / "data" / "ML-EdgeIIoT-dataset.csv.gz"

GITHUB_HARD_LIMIT_MB = 100
GITHUB_WARNING_MB = 50


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. Put ML-EdgeIIoT-dataset.csv in the data folder first.")
        return 1

    with SRC.open("rb") as f_in, gzip.open(DST, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

    src_mb = SRC.stat().st_size / 1024 / 1024
    dst_mb = DST.stat().st_size / 1024 / 1024
    print(f"Original:   {src_mb:8.1f} MB  ({SRC.name})")
    print(f"Compressed: {dst_mb:8.1f} MB  ({DST.name})")

    if dst_mb >= GITHUB_HARD_LIMIT_MB:
        print("ERROR: compressed file is still over GitHub's 100 MB limit; it cannot be pushed as-is.")
        return 1
    if dst_mb >= GITHUB_WARNING_MB:
        print("NOTE: GitHub will show a warning for files over 50 MB, but the push will still work.")
    print("OK: commit data/ML-EdgeIIoT-dataset.csv.gz (the plain .csv stays ignored by git).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
