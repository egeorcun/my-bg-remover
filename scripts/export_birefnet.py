"""`data/train/manifest.jsonl` (veya kompozit manifest) formatındaki bir eğitim
setini BiRefNet'in resmi eğitim koduna beklediği dizin düzenine export eder:

    OUT/SPLIT/im/<id>.jpg   (RGB, JPEG kalite 95)
    OUT/SPLIT/gt/<id>.png   (L modu — tek kanallı gri tonlamalı alpha)

Stem (`<id>`) her iki dizinde birebir eşleşir. Ayrıca `OUT/stats.json` yazılır:
toplam çift sayısı, kategori dağılımı, kısa-kenar çözünürlük yüzdelikleri
(p10/p50/p90) ve kategori bazında "soft-alpha" oranı (0.05 < a < 0.95 aralığındaki
piksellerin payının, o kategorideki görsellerin ortalaması — matting/saydamlık
setlerinde yumuşak geçişlerin ne kadar temsil edildiğini gösterir).

`gt_alpha=None` olan satırlar (GT'siz) export'a dahil edilmez — BiRefNet eğitimi
GT gerektirir.

İdempotentlik: `im/<id>.jpg` ve `gt/<id>.png` ikisi de zaten varsa yeniden
YAZILMAZ (yalnız stats hesaplamasına disk üzerindeki mevcut dosyadan dahil edilir)
— büyük setlerde kesintiye uğramış bir export'un güvenle sürdürülmesini sağlar.

Duplicate stem çakışması: manifest'te aynı `id` iki kez geçerse
`benchmark.testset.load_manifest` (bu script'in üzerine kurulduğu ortak manifest
altyapısı) "tekrarlanan id" ValueError'ı fırlatır — export bu hatayı olduğu gibi
yükseltir (sessizce üzerine yazma YOKTUR).

Kullanım:
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
    """[0,1] normalize alpha dizisinde 0.05 < a < 0.95 olan piksellerin oranı."""
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

    # load_manifest zaten tekrarlanan id'leri reddeder (bkz. modül docstring'i) —
    # bu yüzden burada ayrıca bir "duplicate stem" kontrolüne gerek yok.
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
    parser.add_argument("--manifest", required=True, help="kaynak manifest.jsonl (testset formatı)")
    parser.add_argument("--out", required=True, help="BiRefNet düzeninin yazılacağı kök dizin")
    parser.add_argument("--split-name", default="TRAIN", help="alt dizin adı (varsayılan: TRAIN)")
    args = parser.parse_args()

    stats = export(args.manifest, args.out, split_name=args.split_name)
    print(f"{stats['total']} çift export edildi -> {Path(args.out) / args.split_name}")
    for category, count in stats["category_counts"].items():
        print(f"  {category}: {count}")
    print(f"stats.json yazıldı: {Path(args.out) / 'stats.json'}")


if __name__ == "__main__":
    main()
