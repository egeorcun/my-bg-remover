"""Generator of bokeh-background DERIVATIVE copies from existing TRAIN pairs (v8).

Answers the v8 background-purity defect with DATA (HF discussion #1, the cat
masks): on real photos with furry/hairy subjects the model leaves a faint gray
haze in the background — birefnet-hr outputs a bit-zero background on the same
images while lucida v7 leaves 20x more residue in the `hair` test category
(bg_mae 0.0069 vs 0.0003). Root cause: the synthetic fx/glow/smoke training
data taught "keep faint radiance around a subject semi-transparent", and the
model over-generalizes that to bokeh, shadows and blur around fur in real
photos.

The counter-lesson generated here — **bokeh copies (`{stem}_k00`)**: an
existing im/gt pair is re-rendered with its background defocused. The
background region (GT alpha == 0 side) is gaussian-blurred via NORMALIZED
convolution restricted to background pixels (the subject's colors do not bleed
into the blur — the background is extended under the subject before blurring),
with `ORB_PROB` probability bright soft "bokeh orbs" are added to the
background, and the sharp subject is composited back on top using the ORIGINAL
alpha as the matte. The GT is byte-identical to the source: a blurry, glowing
background around a furry subject is still EXACTLY 0.

SOURCE SELECTION CONTRACTS:
- Only stems in the requested categories (default: `hair`) are sources.
- `_e<NN>`/`_m<NN>`/`_k<NN>` derivative stems are NOT used as sources (no
  derivative of a derivative).
- Eligibility (`is_bokeh_source`): the GT must contain BOTH a solid subject
  (alpha > `SOLID_ALPHA_THRESH` ratio >= `SOLID_MIN_RATIO`) and a real
  background to blur (alpha == 0 ratio >= `BG_MIN_RATIO`).
- Stems are scanned in sorted order and the first `count` eligible ones are
  selected (deterministic). If the output copy already exists on disk,
  eligibility is accepted without loading the GT (file existence is proof —
  the select_mixed_sources resume speed-up from scripts/make_v6_copies.py).

OUTPUT CONTRACTS (SAME as scripts/make_v6_copies.py):
- Output layout: `out_dir/im/{stem}.jpg` (RGB, JPEG q92) +
  `out_dir/gt/{stem}.png` (mode-L 8-bit alpha) — `_save_pair` is identical.
- Manifest: `{"id": new_stem, "category": source_category}` lines APPENDED to
  JSONL (`out_manifest`, default `out_dir/manifest.jsonl`).
- Determinism: `_item_rng(seed, new_stem)` — same seed + same stem ->
  bit-identical output, independent of processing order and resume skips.
- Idempotency: an existing im+gt pair is skipped; file present but manifest
  line missing -> only the line is completed.

Usage:
    uv run python scripts/make_bokeh_copies.py \
        --train-im-dir data/TRAIN/im --train-gt-dir data/TRAIN/gt \
        --categories-manifest train_composites_manifest.jsonl \
        --out-dir data/train_v8 --seed 42 --count 9000
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# The TRAIN pool may contain composites from 100MP+ sources (see the same
# note in scripts/make_textfx.py).
Image.MAX_IMAGE_PIXELS = None

DEFAULT_COUNT = 9000
DEFAULT_CATEGORIES = ("hair",)

SOLID_ALPHA_THRESH = 0.9
SOLID_MIN_RATIO = 0.05   # solid subject ratio threshold (alpha > 0.9)
BG_MIN_RATIO = 0.15      # true-background ratio threshold (alpha == 0)

BLUR_FRAC_LO, BLUR_FRAC_HI = 0.008, 0.03  # blur sigma / min(h, w)
BLUR_SIGMA_MIN, BLUR_SIGMA_MAX = 3.0, 25.0
ORB_PROB = 0.5           # probability of bright bokeh orbs in the background
ORB_COUNT_LO, ORB_COUNT_HI = 3, 12  # inclusive-inclusive
ORB_RADIUS_LO, ORB_RADIUS_HI = 0.015, 0.06  # orb radius / min(h, w)
ORB_STRENGTH_LO, ORB_STRENGTH_HI = 0.25, 0.7

# Derivative suffixes: e/m (make_v6_copies) + k (this script) CANNOT be sources.
_DERIVED_SUFFIX_RE = re.compile(r"_[emk]\d{2}$")


# ==========================================================================
# Shared helpers (source: scripts/make_v6_copies.py — same contracts,
# exact copies)
# ==========================================================================
def _item_rng(seed: int, key: str) -> np.random.Generator:
    """Independent/deterministic random stream from the (global seed, item key)
    pair (source: scripts/make_composites.py::_item_rng, exact copy)."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _save_pair(rgb: np.ndarray, alpha: np.ndarray, img_path: Path, gt_path: Path) -> None:
    """Source: scripts/make_composites.py::_save_pair — same save contract."""
    img_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=92)
    Image.fromarray(np.round(alpha.clip(0, 1) * 255).astype(np.uint8), mode="L").save(gt_path)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """target_size = (w, h); if given and the sizes do not match, the alpha is
    rescaled (source: make_composites.py::_load_alpha)."""
    im = Image.open(path).convert("L")
    if target_size is not None and im.size != target_size:
        im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _load_manifest_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def _append_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _list_pair_stems(im_dir: Path, gt_dir: Path) -> list[str]:
    """Stems in the intersection of `im_dir/*.jpg` and `gt_dir/*.png` (sorted) —
    macOS AppleDouble leftovers (`._*`) are filtered out (the v4 cell pattern)."""
    ims = {p.stem for p in Path(im_dir).iterdir()
           if p.is_file() and p.suffix.lower() == ".jpg" and not p.name.startswith("._")}
    gts = {p.stem for p in Path(gt_dir).iterdir()
           if p.is_file() and p.suffix.lower() == ".png" and not p.name.startswith("._")}
    return sorted(ims & gts)


# ==========================================================================
# Bokeh rendering
# ==========================================================================
def is_bokeh_source(alpha: np.ndarray) -> bool:
    """Does the GT contain both a solid subject (alpha > 0.9 ratio >= 5%) and
    enough true background to blur (alpha == 0 ratio >= 15%)?"""
    solid = float((alpha > SOLID_ALPHA_THRESH).mean())
    bg = float((alpha == 0.0).mean())
    return solid >= SOLID_MIN_RATIO and bg >= BG_MIN_RATIO


def _bokeh_background(
    rng: np.random.Generator, rgb: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Defocused background layer (H, W, 3 float in [0, 255]).

    Normalized convolution restricted to background pixels: the numerator
    blurs `rgb * (1 - alpha)` and is divided by the blurred `(1 - alpha)`
    weight — the background is extended UNDER the subject instead of the
    subject's colors bleeding into the blur ring. Optional bright soft orbs
    (screen-like blend toward a light color) imitate out-of-focus highlights;
    they live only in this background layer, so the GT stays 0 on them."""
    h, w = alpha.shape
    sigma = float(np.clip(
        min(h, w) * float(rng.uniform(BLUR_FRAC_LO, BLUR_FRAC_HI)),
        BLUR_SIGMA_MIN, BLUR_SIGMA_MAX,
    ))
    inv = (1.0 - alpha).astype(np.float32)
    num = ndimage.gaussian_filter(rgb.astype(np.float32) * inv[..., None], (sigma, sigma, 0))
    den = ndimage.gaussian_filter(inv, sigma)
    blurred = num / np.maximum(den, 1e-4)[..., None]

    if rng.uniform() < ORB_PROB:
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        for _ in range(int(rng.integers(ORB_COUNT_LO, ORB_COUNT_HI + 1))):
            cy, cx = 0.0, 0.0
            for _try in range(8):  # prefer centers on background pixels
                cy = float(rng.uniform(0, h))
                cx = float(rng.uniform(0, w))
                if alpha[min(h - 1, int(cy)), min(w - 1, int(cx))] < 0.1:
                    break
            r = min(h, w) * float(rng.uniform(ORB_RADIUS_LO, ORB_RADIUS_HI))
            disc = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 2.0) ** 2))
            strength = float(rng.uniform(ORB_STRENGTH_LO, ORB_STRENGTH_HI))
            color = np.asarray(
                [float(rng.uniform(180, 255)) for _ in range(3)], dtype=np.float32
            )
            blurred = blurred + strength * disc[..., None] * (color - blurred)

    return np.clip(blurred, 0.0, 255.0)


def render_bokeh_copy(
    rng: np.random.Generator, rgb: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(new RGB uint8, alpha UNCHANGED). The sharp subject is composited over
    the defocused background with the original alpha as the matte — fur edges
    keep their exact softness, the background becomes bokeh, the GT does not
    move by a single byte (that is the lesson: bokeh/glow around fur = 0)."""
    bg = _bokeh_background(rng, rgb, alpha)
    out = alpha[..., None] * rgb.astype(np.float32) + (1.0 - alpha[..., None]) * bg
    return out.round().clip(0, 255).astype(np.uint8), alpha


# ==========================================================================
# Selection + generation + orchestration (the make_v6_copies.py pattern)
# ==========================================================================
def select_bokeh_sources(
    stems: list[str],
    category_by_stem: dict[str, str],
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    count: int,
    categories: set[str],
) -> list[str]:
    """Scans the requested-category stems IN ORDER and returns the first
    `count` that pass `is_bokeh_source` (deterministic). If the output copy
    already exists on disk, eligibility is accepted without loading the GT
    (file existence is proof — resume speed-up)."""
    chosen: list[str] = []
    if count <= 0:
        return chosen
    for stem in stems:
        if len(chosen) >= count:
            break
        if category_by_stem.get(stem) not in categories or _DERIVED_SUFFIX_RE.search(stem):
            continue
        if (out_im_dir / f"{stem}_k00.jpg").exists() and (out_gt_dir / f"{stem}_k00.png").exists():
            chosen.append(stem)
            continue
        if is_bokeh_source(_load_alpha(train_gt_dir / f"{stem}.png")):
            chosen.append(stem)
    return chosen


def gen_bokeh(
    sources: list[str],
    train_im_dir: Path,
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    category_by_stem: dict[str, str],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """One `{stem}_k00` copy per source. Returns (rows, generated, skipped)."""
    new_rows: list[dict] = []
    generated = skipped = 0
    for stem in sources:
        new_stem = f"{stem}_k00"
        img_path = out_im_dir / f"{new_stem}.jpg"
        gt_path = out_gt_dir / f"{new_stem}.png"
        row = {"id": new_stem, "category": category_by_stem[stem]}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if new_stem not in existing_ids:
                new_rows.append(row)  # file exists, manifest line missing -> line only
            continue
        rng = _item_rng(seed, new_stem)
        rgb = _load_rgb(train_im_dir / f"{stem}.jpg")
        alpha = _load_alpha(train_gt_dir / f"{stem}.png", (rgb.shape[1], rgb.shape[0]))
        out_rgb, out_alpha = render_bokeh_copy(rng, rgb, alpha)
        _save_pair(out_rgb, out_alpha, img_path, gt_path)
        new_rows.append(row)
        generated += 1
    return new_rows, generated, skipped


def run(
    train_im_dir: Path,
    train_gt_dir: Path,
    category_by_stem: dict[str, str],
    out_dir: Path,
    seed: int = 42,
    count: int = DEFAULT_COUNT,
    categories: set[str] | None = None,
    out_manifest: Path | None = None,
    exclude_stems: set[str] | None = None,
) -> dict[str, int]:
    """Runs the bokeh generator; returns kind -> number of newly generated
    pairs (only entries >0 — the make_textfx.run() pattern).

    `exclude_stems`: stems that must NOT be used as sources (VAL leak guard —
    the caller derives it from val_stems.json)."""
    train_im_dir, train_gt_dir = Path(train_im_dir), Path(train_gt_dir)
    out_dir = Path(out_dir)
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)
    categories = set(categories) if categories else set(DEFAULT_CATEGORIES)

    stems = _list_pair_stems(train_im_dir, train_gt_dir)
    if exclude_stems:
        stems = [s for s in stems if s not in exclude_stems]

    sources = select_bokeh_sources(
        stems, category_by_stem, train_gt_dir, out_im_dir, out_gt_dir, count, categories
    )
    print(f"bokeh sources: {len(sources)} (categories: {sorted(categories)})")
    rows, generated, skipped = gen_bokeh(
        sources, train_im_dir, train_gt_dir, out_im_dir, out_gt_dir,
        category_by_stem, seed, existing_ids,
    )

    # only new ids go to the manifest (in-run safety dedup — make_textfx pattern)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    result: dict[str, int] = {}
    if generated:
        result["bokeh"] = generated
    print(f"{generated} new pairs written, {skipped} already existed (skipped)")
    return result


def _load_categories(path: Path) -> dict[str, str]:
    """Stem -> category map from a JSONL manifest (the
    train_composites_manifest.jsonl schema: at least `id` + `category`)."""
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                result[row["id"]] = row["category"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-im-dir", required=True, help="source TRAIN im/ directory (*.jpg)")
    parser.add_argument("--train-gt-dir", required=True, help="source TRAIN gt/ directory (*.png)")
    parser.add_argument(
        "--categories-manifest", required=True,
        help="stem->category JSONL (train_composites_manifest.jsonl)",
    )
    parser.add_argument("--out-dir", required=True, help="output root (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--categories", default=",".join(DEFAULT_CATEGORIES),
        help="comma-separated source categories (default: hair)",
    )
    parser.add_argument("--out-manifest", default=None, help="default: <out-dir>/manifest.jsonl")
    parser.add_argument(
        "--exclude-stems-file", default=None,
        help="one source stem per line (VAL leak guard) — these are not used as sources",
    )
    args = parser.parse_args()
    exclude_stems = None
    if args.exclude_stems_file:
        exclude_stems = {
            line.strip()
            for line in Path(args.exclude_stems_file).read_text().splitlines()
            if line.strip()
        }
    run(
        Path(args.train_im_dir),
        Path(args.train_gt_dir),
        _load_categories(Path(args.categories_manifest)),
        Path(args.out_dir),
        seed=args.seed,
        count=args.count,
        categories={c.strip() for c in args.categories.split(",") if c.strip()},
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
        exclude_stems=exclude_stems,
    )


if __name__ == "__main__":
    main()
