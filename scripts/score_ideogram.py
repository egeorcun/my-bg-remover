"""`results/ideogram/<id>.png` (fal.ai Ideogram remove-background RGBA çıktıları,
bkz. `benchmark/ideogram.py`) için GT-karşılaştırmalı metrikleri hesaplar ve
`results/baseline/metrics.json`'a `"ideogram"` model adıyla BİRLEŞTİRİR (merge).

Ideogram, `bgr/registry.py` üzerinden çalışan bir segmenter DEĞİL (harici bir
API'nin önceden indirilmiş çıktıları) — bu yüzden `benchmark.run.run_benchmark`
akışına giremiyor; bu betik onun YERİNE, aynı metrik/birleştirme sözleşmesini
kullanarak (`benchmark.run._load_alpha` / `_merge_metrics` — İTHAL EDİLİR,
KOPYALANMAZ, tek doğruluk kaynağı ilkesi) yalnız ideogram için aynı işi yapar.
`scripts/compare_v1.py`'nin varsayılan `--baselines` listesi zaten `ideogram`'ı
içeriyor (yalnız `metrics.json`'da fiilen bulunuyorsa gösterilir) — bu betik
koştuktan sonra ideogram karşılaştırma tablosunda otomatik belirir.

Manifest'teki `gt_alpha` alanı boş olan (piksel-GT'siz) satırlar atlanır (GT'siz
metrik hesaplanamaz — mevcut `benchmark.run` sözleşmesiyle aynı). Manifest'te
GT'si olup `results/ideogram/<id>.png` dosyası bulunamayan satırlar da (ör. fal
API çağrısı başarısız olmuş) sessizce değil, KONSOLA UYARI yazılarak atlanır.

Kullanım:
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
            f"UYARI: {len(skipped)}/{sum(1 for r in rows if r['gt_alpha'])} GT'li görsel için "
            f"ideogram çıktısı bulunamadı, atlandı: {skipped}"
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
