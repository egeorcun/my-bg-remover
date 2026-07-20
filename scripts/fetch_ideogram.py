"""Fetch the Ideogram reference for all images in the manifest.
Usage: uv run python scripts/fetch_ideogram.py [--limit N]
"""
import argparse
from pathlib import Path

from benchmark.ideogram import fetch_reference
from benchmark.testset import load_manifest

ROOT = Path(__file__).resolve().parent.parent
MAX_PER_RUN = 250  # cost guard (~$2.50 cap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_PER_RUN)
    n = min(ap.parse_args().limit, MAX_PER_RUN)
    rows = load_manifest(str(ROOT / "data/testset/manifest.jsonl"))[:n]
    failed: list[str] = []
    for i, row in enumerate(rows, 1):
        out = ROOT / "results/ideogram" / f"{row['id']}.png"
        try:
            fetch_reference(str(ROOT / row["image"]), str(out))
        except Exception as e:  # noqa: BLE001 - one image's failure must not stop the whole fetch
            failed.append(row["id"])
            print(f"[{i}/{len(rows)}] {row['id']} -> ERROR, skipped: {e}")
            continue
        print(f"[{i}/{len(rows)}] {row['id']}")
    if failed:
        print(f"\nSkipped {len(failed)} images: {failed}")


if __name__ == "__main__":
    main()
