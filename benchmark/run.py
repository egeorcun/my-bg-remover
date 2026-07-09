"""Benchmark koşucusu: modeller x manifest -> alpha PNG'ler + metrics.json."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.metrics import all_metrics
from benchmark.testset import load_manifest
from bgr.registry import get_segmenter


def _load_alpha(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode == "RGBA":
        return np.asarray(img.split()[-1], dtype=np.float32) / 255.0
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def run_benchmark(models: list[str], manifest_path: str, out_dir: str) -> dict:
    rows = load_manifest(manifest_path)
    out = Path(out_dir)
    per_image: dict = {}
    for name in models:
        seg = get_segmenter(name)
        model_dir = out / name
        model_dir.mkdir(parents=True, exist_ok=True)
        per_image[name] = {}
        for row in rows:
            dst = model_dir / f"{row['id']}.png"
            if not dst.exists():
                alpha = seg.predict_alpha(Image.open(row["image"]))
                Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(dst)
            if row["gt_alpha"]:
                pred = _load_alpha(str(dst))
                gt = _load_alpha(row["gt_alpha"])
                per_image[name][row["id"]] = all_metrics(pred, gt)

    categories = {r["id"]: r["category"] for r in rows}
    per_category: dict = {}
    overall: dict = {}
    for name, images in per_image.items():
        cat_acc: dict = defaultdict(lambda: defaultdict(list))
        for rid, m in images.items():
            for k, v in m.items():
                cat_acc[categories[rid]][k].append(v)
        per_category[name] = {
            c: {k: float(np.mean(v)) for k, v in ms.items()} for c, ms in cat_acc.items()
        }
        keys = {k for m in images.values() for k in m}
        overall[name] = {
            k: float(np.mean([m[k] for m in images.values()])) for k in keys
        }

    result = {"per_image": per_image, "per_category": per_category, "overall": overall}
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    result = _merge_metrics(metrics_path, result)
    metrics_path.write_text(json.dumps(result, indent=2))
    return result


def _merge_metrics(metrics_path: Path, new_result: dict) -> dict:
    """Var olan metrics.json ile birleştir: bu koşuda olmayan modellerin
    per_image/per_category/overall girdileri korunur, bu koşudakiler üzerine yazılır."""
    if not metrics_path.exists():
        return new_result
    try:
        existing = json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return new_result
    merged: dict = {}
    for key in ("per_image", "per_category", "overall"):
        merged[key] = {**existing.get(key, {}), **new_result.get(key, {})}
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = run_benchmark(a.models.split(","), a.manifest, a.out)
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
