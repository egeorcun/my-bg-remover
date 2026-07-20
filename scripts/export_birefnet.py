"""Exports a training set in `data/train/manifest.jsonl` (or composite manifest)
format to the directory layout BiRefNet's official training code expects:

    OUT/SPLIT/im/<id>.jpg   (RGB, JPEG quality 95)
    OUT/SPLIT/gt/<id>.png   (L mode — single-channel grayscale alpha)

The stem (`<id>`) matches exactly across both directories. `OUT/stats.json` is also
written: total pair count, category distribution, short-side resolution percentiles
(p10/p50/p90) and the per-category "soft-alpha" ratio (the share of pixels in the
0.05 < a < 0.95 range, averaged over the images in that category — shows how well
soft transitions are represented in matting/transparency sets).

Rows with `gt_alpha=None` (no GT) are not included in the export — BiRefNet
training requires GT.

Idempotency: if both `im/<id>.jpg` and `gt/<id>.png` already exist, they are NOT
REWRITTEN (they are only included in the stats computation from the existing file
on disk) — this lets an interrupted export on large sets be resumed safely.

Duplicate stem collision: if the same `id` appears twice in the manifest,
`benchmark.testset.load_manifest` (the shared manifest infrastructure this script
is built on) raises a "duplicate id" ValueError — the export re-raises this error
as-is (there is NO silent overwrite).

Usage:
    uv run python scripts/export_birefnet.py --manifest data/train_composites/manifest.jsonl \
        --out data/birefnet_format --split-name TRAIN
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.testset import load_manifest

JPEG_QUALITY = 95
SOFT_ALPHA_LOW = 0.05
SOFT_ALPHA_HIGH = 0.95


def _soft_alpha_ratio(alpha: np.ndarray) -> float:
    """Ratio of pixels with 0.05 < a < 0.95 in a [0,1]-normalized alpha array."""
    mask = (alpha > SOFT_ALPHA_LOW) & (alpha < SOFT_ALPHA_HIGH)
    return float(mask.mean())


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def export(manifest_path: str | Path, out_dir: str | Path, split_name: str = "TRAIN") -> dict:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    split_dir = out_dir / split_name
    im_dir = split_dir / "im"
    gt_dir = split_dir / "gt"
    im_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # load_manifest already rejects duplicate ids (see module docstring) —
    # so no separate "duplicate stem" check is needed here.
    rows = [r for r in load_manifest(str(manifest_path)) if r.get("gt_alpha")]

    category_counts: dict[str, int] = {}
    short_sides: list[int] = []
    soft_alpha_by_category: dict[str, list[float]] = {}

    for row in rows:
        stem = row["id"]
        category = row["category"]
        out_img = im_dir / f"{stem}.jpg"
        out_gt = gt_dir / f"{stem}.png"

        if not (out_img.exists() and out_gt.exists()):
            with Image.open(row["image"]) as im:
                im.convert("RGB").save(out_img, format="JPEG", quality=JPEG_QUALITY)
            with Image.open(row["gt_alpha"]) as gt:
                gt.convert("L").save(out_gt, format="PNG")

        category_counts[category] = category_counts.get(category, 0) + 1
        with Image.open(out_img) as im2:
            short_sides.append(min(im2.size))
        with Image.open(out_gt) as gt2:
            alpha = np.asarray(gt2, dtype=np.float32) / 255.0
        soft_alpha_by_category.setdefault(category, []).append(_soft_alpha_ratio(alpha))

    stats = {
        "total": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "resolution_short_side_percentiles": {
            "p10": _percentile(short_sides, 10),
            "p50": _percentile(short_sides, 50),
            "p90": _percentile(short_sides, 90),
        },
        "soft_alpha_ratio_by_category": {
            cat: float(np.mean(vals)) for cat, vals in sorted(soft_alpha_by_category.items())
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="source manifest.jsonl (testset format)")
    parser.add_argument("--out", required=True, help="root directory to write the BiRefNet layout to")
    parser.add_argument("--split-name", default="TRAIN", help="subdirectory name (default: TRAIN)")
    args = parser.parse_args()

    stats = export(args.manifest, args.out, split_name=args.split_name)
    print(f"{stats['total']} pairs exported -> {Path(args.out) / args.split_name}")
    for category, count in stats["category_counts"].items():
        print(f"  {category}: {count}")
    print(f"stats.json written: {Path(args.out) / 'stats.json'}")


if __name__ == "__main__":
    main()
