"""`data/train/manifest.jsonl` + arka plan havuzundan compositing/augmentasyonlu
eğitim kopyaları üretir (`bgr/compositing.py`: compose + augment).

Kategori bazlı per-image çarpanları (bkz. Faz 2 planı Task 4):
- transparent: ×`per-image`×10 (compose + augment) — efektif karışımda ≥%20 pay için (bkz. docs/reports/2026-07-faz2-veri.md karışım hesabı; eski ×4 %7'de kalıyordu)
  yüksek çarpan.
- camouflage: ×`per-image`×2 ama **compose YOK**, yalnız augment — orijinal arka
  plan korunur (compositing kamuflajı bozar: obje-arka plan doku/renk uyumu
  kamuflajın özü, rastgele bir bg'ye yapıştırmak bu sinyali yok eder).
- diğer tüm kategoriler (hair/complex/thin/general/product/illustration): ×`per-image`×1,
  compose + augment.

Kullanım:
    uv run python scripts/make_composites.py --manifest data/train/manifest.jsonl \
        --backgrounds data/backgrounds --per-image 1 --seed 42 --out data/train_composites/
    uv run python scripts/make_composites.py ... --limit 20   # duman/smoke koşusu

Determinizm: her (kaynak satır id, kopya indeksi) çifti için `np.random.SeedSequence`
ile BAĞIMSIZ bir alt-akış türetilir (global sıralı bir rng yerine) — böylece hem
"aynı seed -> aynı çıktı" hem de kesintiye uğramış/kısmi bir koşunun (zaten üretilmiş
id'ler atlanarak) güvenle sürdürülmesi aynı anda sağlanır: atlanan öğeler, henüz
üretilmemiş öğelerin rastgele akışını etkilemez.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.testset import append_entries, load_manifest
from bgr.compositing import augment, compose

CATEGORY_MULTIPLIER: dict[str, int] = {"transparent": 10, "camouflage": 2}
DEFAULT_MULTIPLIER = 1
NO_COMPOSE_CATEGORIES = {"camouflage"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def multiplier(category: str) -> int:
    return CATEGORY_MULTIPLIER.get(category, DEFAULT_MULTIPLIER)


def _item_rng(seed: int, key: str) -> np.random.Generator:
    """(global seed, öğe anahtarı) çiftinden bağımsız/deterministik rastgele akış.

    İşlem sırasından ve önceden atlanmış (zaten var olan) öğelerden ETKİLENMEZ —
    her öğe kendi id'sinden türetilen sabit bir alt-seed kullanır.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """target_size = (w, h); verilirse ve boyut uyuşmuyorsa alpha yeniden ölçeklenir."""
    im = Image.open(path).convert("L")
    if target_size is not None and im.size != target_size:
        im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _save_pair(rgb: np.ndarray, alpha: np.ndarray, img_path: Path, gt_path: Path) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=92)
    Image.fromarray(np.round(alpha.clip(0, 1) * 255).astype(np.uint8), mode="L").save(gt_path)


def run(
    manifest_path: Path,
    backgrounds_dir: Path,
    per_image: int,
    seed: int,
    out_dir: Path,
    limit: int | None = None,
) -> dict[str, int]:
    manifest_path, backgrounds_dir, out_dir = Path(manifest_path), Path(backgrounds_dir), Path(out_dir)
    out_img_dir = out_dir / "images"
    out_gt_dir = out_dir / "gt"
    out_manifest = out_dir / "manifest.jsonl"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_manifest(str(manifest_path)) if r.get("gt_alpha")]
    if limit is not None and limit < len(rows):
        order = np.random.default_rng(seed).permutation(len(rows))[:limit]
        rows = [rows[i] for i in sorted(order.tolist())]

    bg_paths = sorted(p for p in backgrounds_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not bg_paths:
        raise SystemExit(f"arka plan bulunamadı: {backgrounds_dir}")

    existing_ids: set[str] = set()
    if out_manifest.exists():
        existing_ids = {r["id"] for r in load_manifest(str(out_manifest))}

    counts: dict[str, int] = {}
    new_entries: list[dict] = []
    skipped = 0
    for row in rows:
        category = row["category"]
        n_copies = per_image * multiplier(category)
        out_ids = [f"{row['id']}_v{ci:02d}" for ci in range(n_copies)]
        if all(oid in existing_ids for oid in out_ids):
            skipped += len(out_ids)
            continue

        fg_rgb = _load_rgb(Path(row["image"]))
        alpha = _load_alpha(Path(row["gt_alpha"]), target_size=(fg_rgb.shape[1], fg_rgb.shape[0]))

        for out_id in out_ids:
            if out_id in existing_ids:
                skipped += 1
                continue
            rng = _item_rng(seed, out_id)

            if category in NO_COMPOSE_CATEGORIES:
                out_rgb, out_alpha = fg_rgb, alpha
            else:
                bg_idx = int(rng.integers(0, len(bg_paths)))
                bg_rgb = _load_rgb(bg_paths[bg_idx])
                out_rgb, out_alpha = compose(fg_rgb, alpha, bg_rgb, rng)

            out_rgb, out_alpha = augment(out_rgb, out_alpha, rng)

            img_path = out_img_dir / f"{out_id}.jpg"
            gt_path = out_gt_dir / f"{out_id}.png"
            _save_pair(out_rgb, out_alpha, img_path, gt_path)

            new_entries.append(
                {"id": out_id, "image": str(img_path), "category": category, "gt_alpha": str(gt_path)}
            )
            counts[category] = counts.get(category, 0) + 1

    if new_entries:
        append_entries(str(out_manifest), new_entries)
    print(f"{len(new_entries)} yeni kompozit yazıldı, {skipped} zaten vardı (atlandı)")
    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/train/manifest.jsonl")
    parser.add_argument("--backgrounds", default="data/backgrounds")
    parser.add_argument("--per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/train_composites")
    parser.add_argument("--limit", type=int, default=None, help="yalnız ilk N kaynak satırı (duman koşusu)")
    args = parser.parse_args()
    run(args.manifest, args.backgrounds, args.per_image, args.seed, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
