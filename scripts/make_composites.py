"""Generates composited/augmented training copies from `data/train/manifest.jsonl`
+ a background pool (`bgr/compositing.py`: compose + augment).

Per-category per-image multipliers (see the Phase 2 plan, Task 4):
- transparent: ×`per-image`×10 (compose + augment) — a high multiplier so the
  effective mix reaches a ≥20% share (see the mix calculation in
  the project's internal phase report (removed from the repo); the old ×4 was stuck at 7%).
- camouflage: ×`per-image`×2 but **NO compose**, augment only — the original
  background is preserved (compositing destroys camouflage: object-background
  texture/color harmony is the essence of camouflage, pasting onto a random
  bg erases that signal).
- all other categories (hair/complex/thin/general/product/illustration):
  ×`per-image`×1, compose + augment.

v3 (see the internal review notes (not in the repo)): the reason over-deletion
persisted on the real-photo benchmark was that all categories EXCEPT
camouflage were trained only on SYNTHETIC composited backgrounds (the
original background was never seen — a domain gap). To fix this, every
category outside `NO_COMPOSE_CATEGORIES` gets `ORIGINAL_BG_COPIES`
(default 1) extra copies that preserve the ORIGINAL background (NO compose,
augment only — the EXACT same mechanism as camouflage's path); these copies
are named with the `_o<NN>` suffix instead of `_v<NN>` (they NEVER collide
with existing `_v` outputs — a separate namespace, and idempotent re-runs
generate only the missing `_o00`s). `run()`'s `exclude_source_ids` parameter
excludes source row ids already used in the VAL split (and which therefore
must NOT leak into training) from `_o00` generation (VAL leak guard — see
`training/v3_veri_guncelleme_hucresi.py`, derived from `val_stems.json` on
Drive). With `only_original_bg=True`, ALL `_v<NN>` copies (the physical
compose multipliers) are skipped entirely and only the `_o00` set is
generated — to quickly obtain just the new ~14k `_o00` files on a fresh
Colab VM without regenerating the whole 28k composite set (see the same
file, the "composites_o" stage).

Usage:
    uv run python scripts/make_composites.py --manifest data/train/manifest.jsonl \
        --backgrounds data/backgrounds --per-image 1 --seed 42 --out data/train_composites/
    uv run python scripts/make_composites.py ... --limit 20   # smoke run

Determinism: for each (source row id, copy index) pair an INDEPENDENT
sub-stream is derived via `np.random.SeedSequence` (instead of a global
sequential rng) — this simultaneously guarantees both "same seed -> same
output" and the safe resumption of an interrupted/partial run (skipping
already-generated ids): skipped items do not affect the random streams of
items not yet generated. `_o<NN>` ids use the same `_item_rng` mechanism
(since the id string already differs from `_v<NN>`, they get an independent
sub-stream).
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
ORIGINAL_BG_COPIES = 1
"""Number of extra copies added to every category OUTSIDE `NO_COMPOSE_CATEGORIES`
that preserve the original background (no compose, augment only) — with the
`_o<NN>` suffix (see the "v3" note in the module docstring)."""
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def multiplier(category: str) -> int:
    return CATEGORY_MULTIPLIER.get(category, DEFAULT_MULTIPLIER)


def _item_rng(seed: int, key: str) -> np.random.Generator:
    """Independent/deterministic random stream from the (global seed, item key) pair.

    NOT affected by processing order or by previously skipped (already
    existing) items — each item uses a fixed sub-seed derived from its own id.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """target_size = (w, h); if given and the sizes do not match, the alpha is rescaled."""
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
    exclude_source_ids: set[str] | None = None,
    only_original_bg: bool = False,
) -> dict[str, int]:
    """`exclude_source_ids`: for SOURCE row ids in this set (the `row['id']`
    BEFORE the composite `_v`/`_o` suffix is appended) no `_o<NN>` (original
    background) copy is generated — VAL leak guard (see module docstring).
    Does NOT affect `_v<NN>` copies (it only restricts `_o<NN>` generation).

    `only_original_bg=True`: ALL `_v<NN>` copies (the physical compose
    multipliers) are skipped, only the `_o<NN>` set is generated (see the
    "v3" note in the module docstring)."""
    manifest_path, backgrounds_dir, out_dir = Path(manifest_path), Path(backgrounds_dir), Path(out_dir)
    exclude_source_ids = exclude_source_ids or set()

    # The copy index is formatted 2-digit via `{ci:02d}`; at ci >= 100 it
    # overflows to 3 digits and the `_[vo]\d{2}$` pattern of
    # `training.train_colab_lib.strip_composite_copy_suffix` would no longer
    # MATCH those ids — the VAL leak guard would be silently bypassed for
    # them. The copy count is strictly capped at <= 99 per category (in
    # practice the highest existing value is transparent x10).
    max_copies = per_image * max([DEFAULT_MULTIPLIER, *CATEGORY_MULTIPLIER.values()])
    assert max_copies <= 99 and ORIGINAL_BG_COPIES <= 99, (
        f"copy count cannot exceed 99 per category (per_image={per_image} -> up to "
        f"{max_copies} _v copies; ORIGINAL_BG_COPIES={ORIGINAL_BG_COPIES}): the 2-digit "
        f"`_v<NN>`/`_o<NN>` naming would overflow and the VAL leak guard's suffix "
        f"pattern (_[vo]\\d{{2}}$) would break."
    )

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
        raise SystemExit(f"no backgrounds found: {backgrounds_dir}")

    existing_ids: set[str] = set()
    if out_manifest.exists():
        existing_ids = {r["id"] for r in load_manifest(str(out_manifest))}

    counts: dict[str, int] = {}
    new_entries: list[dict] = []
    skipped = 0
    for row in rows:
        category = row["category"]
        n_v_copies = 0 if only_original_bg else per_image * multiplier(category)
        v_ids = [f"{row['id']}_v{ci:02d}" for ci in range(n_v_copies)]

        o_ids: list[str] = []
        if category not in NO_COMPOSE_CATEGORIES and row["id"] not in exclude_source_ids:
            o_ids = [f"{row['id']}_o{ci:02d}" for ci in range(ORIGINAL_BG_COPIES)]
        o_ids_set = set(o_ids)

        out_ids = v_ids + o_ids
        if not out_ids:
            continue
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

            if category in NO_COMPOSE_CATEGORIES or out_id in o_ids_set:
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
    print(f"{len(new_entries)} new composites written, {skipped} already existed (skipped)")
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
    parser.add_argument("--limit", type=int, default=None, help="only the first N source rows (smoke run)")
    parser.add_argument(
        "--exclude-ids-file",
        default=None,
        help="one source id per line (VAL leak guard) — no _o<NN> is generated for these ids",
    )
    parser.add_argument(
        "--only-original-bg",
        action="store_true",
        help="skip _v<NN> (composed) copies entirely, generate only the _o<NN> (original background) set",
    )
    args = parser.parse_args()
    exclude_source_ids = None
    if args.exclude_ids_file:
        exclude_source_ids = {
            line.strip() for line in Path(args.exclude_ids_file).read_text().splitlines() if line.strip()
        }
    run(
        args.manifest,
        args.backgrounds,
        args.per_image,
        args.seed,
        args.out,
        limit=args.limit,
        exclude_source_ids=exclude_source_ids,
        only_original_bg=args.only_original_bg,
    )


if __name__ == "__main__":
    main()
