"""Computes GT-comparison metrics for `results/ideogram/<id>.png` (fal.ai Ideogram
remove-background RGBA outputs, see `benchmark/ideogram.py`) and MERGES them into
`results/baseline/metrics.json` under the model name `"ideogram"`.

Ideogram is NOT a segmenter running through `bgr/registry.py` (it is the
pre-downloaded output of an external API) — so it cannot enter the
`benchmark.run.run_benchmark` flow; this script does the same job for ideogram
only, IN ITS PLACE, using the same metric/merge contract
(`benchmark.run._load_alpha` / `_merge_metrics` — IMPORTED, NOT COPIED, single
source of truth principle). `scripts/compare_v1.py`'s default `--baselines` list
already includes `ideogram` (shown only if actually present in `metrics.json`) —
after this script runs, ideogram automatically appears in the comparison table.

Rows whose `gt_alpha` field in the manifest is empty (no pixel GT) are skipped
(no metric can be computed without GT — same as the existing `benchmark.run`
contract). Rows that have GT in the manifest but whose `results/ideogram/<id>.png`
file is missing (e.g. the fal API call failed) are also skipped — not silently,
but with a WARNING printed to the console.

Usage:
    uv run python scripts/score_ideogram.py
    uv run python scripts/score_ideogram.py --ideogram-dir results/ideogram \
        --manifest data/testset/manifest.jsonl --metrics results/baseline/metrics.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.metrics import all_metrics
from benchmark.run import _load_alpha, _merge_metrics
from benchmark.testset import load_manifest

MODEL_NAME = "ideogram"


def score_ideogram(ideogram_dir: str, manifest_path: str, metrics_path: str) -> dict:
    rows = load_manifest(manifest_path)
    ideogram_dir_p = Path(ideogram_dir)

    per_image: dict[str, dict[str, float]] = {}
    skipped: list[str] = []
    for row in rows:
        if not row["gt_alpha"]:
            continue
        pred_path = ideogram_dir_p / f"{row['id']}.png"
        if not pred_path.exists():
            skipped.append(row["id"])
            continue
        pred = _load_alpha(str(pred_path))
        gt = _load_alpha(row["gt_alpha"])
        per_image[row["id"]] = all_metrics(pred, gt)

    if skipped:
        print(
            f"WARNING: no ideogram output found for {len(skipped)}/{sum(1 for r in rows if r['gt_alpha'])} "
            f"GT-labeled images, skipped: {skipped}"
        )

    categories = {r["id"]: r["category"] for r in rows}
    cat_acc: dict = defaultdict(lambda: defaultdict(list))
    for rid, m in per_image.items():
        for k, v in m.items():
            cat_acc[categories[rid]][k].append(v)
    per_category = {c: {k: float(np.mean(v)) for k, v in ms.items()} for c, ms in cat_acc.items()}
    keys = {k for m in per_image.values() for k in m}
    overall = {k: float(np.mean([m[k] for m in per_image.values()])) for k in keys} if per_image else {}

    new_result = {
        "per_image": {MODEL_NAME: per_image},
        "per_category": {MODEL_NAME: per_category},
        "overall": {MODEL_NAME: overall},
    }
    metrics_path_p = Path(metrics_path)
    metrics_path_p.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_metrics(metrics_path_p, new_result)
    metrics_path_p.write_text(json.dumps(merged, indent=2))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ideogram-dir", default="results/ideogram")
    ap.add_argument("--manifest", default="data/testset/manifest.jsonl")
    ap.add_argument("--metrics", default="results/baseline/metrics.json")
    a = ap.parse_args()
    result = score_ideogram(a.ideogram_dir, a.manifest, a.metrics)
    print(json.dumps(result["overall"].get(MODEL_NAME, {}), indent=2))


if __name__ == "__main__":
    main()
