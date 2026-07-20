"""Generator of two kinds of DERIVATIVE copies from existing TRAIN pairs for v6.

Answers the two defects of GitHub issue #1 with DATA (see the task context):

1. **Frame-crop copies (`{stem}_e00`)**: in the training composites the
   subject always stayed INSIDE the canvas, so the model never saw a
   "subject touching the frame" and ERASES those subjects. Here an existing
   im/gt pair is cropped so the crop CUTS the subject's bbox (gt alpha >
   `SUBJECT_ALPHA_THRESH`) from a random edge by 20-60% of the bbox length
   (`CUT_FRAC_LO..CUT_FRAC_HI`) — the crop window's boundary passes THROUGH
   the subject, so the subject touches the frame in the new image. One of
   the 4 edges is picked at random; with `SECOND_EDGE_PROB` probability a
   second cut is made from an ADJACENT (perpendicular) edge. The window area
   keeps at least `MIN_KEEP_AREA` (50%) of the original (no extreme
   shrinking). The GT is cropped with the SAME window and the alpha values
   are UNCHANGED — the part touching the frame stays SOLID (that is exactly
   the lesson: touching the edge is NOT a reason for transparency).

2. **Mixed-opacity copies (`{stem}_m00`, `_m01`)**: solid parts of
   transparent objects (bottle cap, glasses temple...) turn semi-transparent
   because the training set has few samples containing BOTH solid AND soft
   alpha. For pairs in the `transparent` category whose GT is both solid
   (ratio of pixels with alpha > `SOLID_ALPHA_THRESH` >= `SOLID_MIN_RATIO`)
   and soft (ratio of `SOFT_LO` < alpha < `SOFT_HI` >= `SOFT_MIN_RATIO`),
   `MIXED_COPIES` (2) augmented copies are generated. Augment is called via
   `bgr.compositing.augment` with `flip_prob=0.0` (signature verified from
   the code: flip is the only geometric transform; color jitter / blur /
   JPEG artifacts affect only the RGB) — the geometry does not change, the
   alpha is preserved AS IS.

SOURCE SELECTION CONTRACTS:
- `_e<NN>`/`_m<NN>` derivative stems are NOT used as sources (no derivative
  of a derivative); `_o<NN>` (make_composites copies with the original
  background) CAN be sources and are PREFERRED — crops with real backgrounds
  are the most valuable.
- Edge-crop sources are distributed PROPORTIONALLY per category
  (largest-remainder); the order WITHIN a category is deterministic:
  preferred sources first (stems with the `_o<NN>` suffix and
  `ORIGINAL_BG_CATEGORIES` — non-composited/original-background categories,
  e.g. camouflage), then the rest, each group alphabetical within itself.
  This way as large a share as possible of the edge-crop sources (target: at
  least half) comes from samples with real backgrounds.
- Mixed source selection: `transparent` stems are scanned IN ORDER, and the
  first `mixed_cap / MIXED_COPIES` stems that pass the threshold test are
  selected (deterministic). If one of the output copies already exists on
  disk, the source counts as eligible without reloading the GT (a resume
  speed-up — since only eligible sources can produce output, file existence
  is proof of eligibility).

OUTPUT CONTRACTS (SAME as scripts/make_textfx.py):
- Output layout: `out_dir/im/{stem}.jpg` (RGB, JPEG q92) +
  `out_dir/gt/{stem}.png` (mode-L 8-bit alpha) — `_save_pair` is identical.
- Manifest: for each pair a `{"id": new_stem, "category": source_category}`
  line is APPENDED to JSONL (`out_manifest`, default `out_dir/manifest.jsonl`).
- Determinism: `_item_rng(seed, new_stem)` (an exact copy of the
  make_composites.py pattern) — same seed + same stem -> bit-identical
  output, independent of processing order and skipped items (resume safety).
- Idempotency: if the im+gt pair already exists on disk, generation is
  skipped; if the file exists but the manifest line is missing, only the
  line is completed — the file is NOT regenerated.
- A source without a subject (fully empty gt) or without a cuttable edge is
  SILENTLY skipped (no pair generated) — the skip derives from content, so
  it is deterministic; the target count may then fall slightly short
  ("~9,000").

Usage:
    uv run python scripts/make_v6_copies.py \
        --train-im-dir data/TRAIN/im --train-gt-dir data/TRAIN/gt \
        --categories-manifest train_composites_manifest.jsonl \
        --out-dir data/train_v6 --seed 42 --edge-count 9000 --mixed-cap 4000
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from bgr.compositing import augment

# The TRAIN pool may contain composites from 100MP+ sources; PIL's 179MP
# "decompression bomb" threshold is lifted for trusted sources
# (see the same note in scripts/make_textfx.py).
Image.MAX_IMAGE_PIXELS = None

DEFAULT_EDGE_COUNT = 9000
DEFAULT_MIXED_CAP = 4000
MIXED_COPIES = 2  # {stem}_m00 + {stem}_m01
SUBJECT_ALPHA_THRESH = 0.1  # the subject bbox comes from pixels above this threshold
CUT_FRAC_LO, CUT_FRAC_HI = 0.2, 0.6  # the cut share of the bbox length
MIN_KEEP_AREA = 0.5  # window area >= 50% of the original
SECOND_EDGE_PROB = 0.35  # probability of sometimes cutting from two adjacent edges
SOLID_ALPHA_THRESH = 0.9
SOLID_MIN_RATIO = 0.08  # solid pixel ratio threshold (alpha > 0.9)
SOFT_LO, SOFT_HI = 0.05, 0.95
SOFT_MIN_RATIO = 0.08  # soft pixel ratio threshold (0.05 < alpha < 0.95)

# Derivative suffixes: this script's own outputs (_eNN/_mNN) CANNOT be sources.
_DERIVED_SUFFIX_RE = re.compile(r"_[em]\d{2}$")
# make_composites' original-background copies (_oNN) — preferred sources.
_ORIGINAL_BG_SUFFIX_RE = re.compile(r"_o\d{2}$")
# Categories that were never composited (original background always preserved) —
# see scripts/make_composites.py NO_COMPOSE_CATEGORIES.
ORIGINAL_BG_CATEGORIES = {"camouflage"}

_EDGES = ("left", "right", "top", "bottom")


# ==========================================================================
# Shared helpers (source: scripts/make_composites.py + make_textfx.py —
# same contracts, exact copies)
# ==========================================================================
def _item_rng(seed: int, key: str) -> np.random.Generator:
    """Independent/deterministic random stream from the (global seed, item key) pair.

    NOT affected by processing order or by previously skipped (already
    existing) items — each item uses a fixed sub-seed derived from its own id.
    (Source: scripts/make_composites.py::_item_rng, exact copy.)
    """
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
    """Ids in the output manifest (to avoid duplicating lines on resume)."""
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


def _is_preferred_source(stem: str, category: str) -> bool:
    """Is this a source with a real background? `_o<NN>` copies + categories
    that were never composited (see source selection in the module docstring)."""
    return category in ORIGINAL_BG_CATEGORIES or _ORIGINAL_BG_SUFFIX_RE.search(stem) is not None


# ==========================================================================
# Frame-crop (edge-crop) — window selection + generation
# ==========================================================================
def _cut_bounds(
    b_lo: int, b_hi: int, length: int, min_keep_px: int, side: str
) -> tuple[int, int] | None:
    """Returns the valid cut-pixel range [cut_lo, cut_hi] for one axis.

    `side='lo'`: cut from the START of the axis, new window `[b_lo+cut,
    length)` — the boundary passes through the bbox. `side='hi'`: cut from
    the END of the axis, new window `[0, b_hi-cut)`. Constraints: the cut is
    20-60% of the bbox length, the remaining window length >= `min_keep_px`,
    the boundary is STRICTLY inside the bbox (1 <= cut <= b-1). None if no
    valid range exists."""
    b = b_hi - b_lo
    cut_lo = max(1, math.ceil(CUT_FRAC_LO * b))
    cut_hi = math.floor(CUT_FRAC_HI * b)
    if side == "lo":
        cut_hi = min(cut_hi, length - min_keep_px - b_lo, b - 1)
    else:
        cut_hi = min(cut_hi, b_hi - min_keep_px, b - 1)
    if cut_hi < cut_lo:
        return None
    return cut_lo, cut_hi


def _edge_crop_window(
    rng: np.random.Generator, alpha: np.ndarray, min_keep_area: float = MIN_KEEP_AREA
) -> tuple[int, int, int, int] | None:
    """Returns a crop window `(x0, y0, x1, y1)` that cuts the subject from a
    random edge (sometimes from two adjacent edges); None if no valid cut is
    found.

    The window extends to the image border in the uncut directions — so on
    every cut axis the subject touches the EXACT EDGE of the new image. The
    total window area is always >= `min_keep_area` × the original area (in
    the second cut the remaining share is bounded by `min_keep_area / k1`,
    so the product is preserved)."""
    h, w = alpha.shape
    win = [0, 0, w, h]  # x0, y0, x1, y1

    def _apply(edge: str, keep_frac: float) -> bool:
        x0, y0, x1, y1 = win
        ys, xs = np.nonzero(alpha[y0:y1, x0:x1] > SUBJECT_ALPHA_THRESH)
        if xs.size == 0:
            return False
        if edge in ("left", "right"):
            b_lo, b_hi = x0 + int(xs.min()), x0 + int(xs.max()) + 1
            length, min_keep_px = w, math.ceil(keep_frac * w)
            side = "lo" if edge == "left" else "hi"
        else:
            b_lo, b_hi = y0 + int(ys.min()), y0 + int(ys.max()) + 1
            length, min_keep_px = h, math.ceil(keep_frac * h)
            side = "lo" if edge == "top" else "hi"
        bounds = _cut_bounds(b_lo, b_hi, length, min_keep_px, side)
        if bounds is None:
            return False
        cut = int(rng.integers(bounds[0], bounds[1] + 1))
        if edge == "left":
            win[0] = b_lo + cut
        elif edge == "right":
            win[2] = b_hi - cut
        elif edge == "top":
            win[1] = b_lo + cut
        else:
            win[3] = b_hi - cut
        return True

    first = None
    for idx in rng.permutation(len(_EDGES)):
        edge = _EDGES[int(idx)]
        if _apply(edge, min_keep_area):
            first = edge
            break
    if first is None:
        return None

    # sometimes a second cut from an ADJACENT (perpendicular) edge — for the
    # area guarantee, the remaining share is divided by the first cut's
    # remaining ratio (k1 * k2 >= min_keep_area).
    if rng.uniform() < SECOND_EDGE_PROB:
        if first in ("left", "right"):
            k1 = (win[2] - win[0]) / w
            perp = ["top", "bottom"]
        else:
            k1 = (win[3] - win[1]) / h
            perp = ["left", "right"]
        if rng.uniform() < 0.5:
            perp.reverse()
        for edge in perp:
            if _apply(edge, min_keep_area / k1):
                break

    return tuple(win)


def select_edge_sources(
    stems: list[str], category_by_stem: dict[str, str], count: int
) -> list[tuple[str, str]]:
    """Selects the edge-crop sources: PROPORTIONAL per category (largest
    remainder — the category with the biggest fractional share first,
    alphabetical on ties), within a category in deterministic order with
    PREFERRED sources first (`_o<NN>` / `ORIGINAL_BG_CATEGORIES`), then the
    rest. Returns a `(stem, category)` list. `_e/_m` derivatives and stems
    with unknown categories are filtered out."""
    eligible = [
        s for s in stems if s in category_by_stem and not _DERIVED_SUFFIX_RE.search(s)
    ]
    if count <= 0 or not eligible:
        return []
    by_cat: dict[str, list[str]] = {}
    for s in eligible:
        by_cat.setdefault(category_by_stem[s], []).append(s)
    for c, lst in by_cat.items():
        lst.sort(key=lambda s: (0 if _is_preferred_source(s, c) else 1, s))

    total = len(eligible)
    count = min(count, total)
    cats = sorted(by_cat)
    quotas: dict[str, int] = {}
    fracs: list[tuple[float, str]] = []
    used = 0
    for c in cats:
        exact = count * len(by_cat[c]) / total
        quotas[c] = math.floor(exact)
        used += quotas[c]
        fracs.append((-(exact - quotas[c]), c))  # biggest fractional share first
    for _, c in sorted(fracs):
        if used >= count:
            break
        if quotas[c] < len(by_cat[c]):
            quotas[c] += 1
            used += 1
    while used < count:  # if some categories are full, distribute the rest deterministically
        progressed = False
        for c in cats:
            if used >= count:
                break
            if quotas[c] < len(by_cat[c]):
                quotas[c] += 1
                used += 1
                progressed = True
        if not progressed:
            break
    return [(s, c) for c in cats for s in by_cat[c][: quotas[c]]]


def gen_edge_crops(
    sources: list[tuple[str, str]],
    train_im_dir: Path,
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """Returns (manifest rows, generated, skipped). Sources without a cuttable
    edge (empty gt / a bbox that does not fit the constraints) are silently
    skipped."""
    new_rows: list[dict] = []
    generated = skipped = no_fit = 0
    for stem, category in sources:
        new_stem = f"{stem}_e00"
        img_path = out_im_dir / f"{new_stem}.jpg"
        gt_path = out_gt_dir / f"{new_stem}.png"
        row = {"id": new_stem, "category": category}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if new_stem not in existing_ids:
                new_rows.append(row)  # file exists, manifest line missing -> line only
            continue
        rng = _item_rng(seed, new_stem)
        rgb = _load_rgb(train_im_dir / f"{stem}.jpg")
        alpha = _load_alpha(train_gt_dir / f"{stem}.png", (rgb.shape[1], rgb.shape[0]))
        window = _edge_crop_window(rng, alpha)
        if window is None:
            no_fit += 1
            continue
        x0, y0, x1, y1 = window
        # Alpha values are UNCHANGED — pure slicing (the part touching the frame stays solid).
        _save_pair(rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], img_path, gt_path)
        new_rows.append(row)
        generated += 1
    if no_fit:
        print(f"edge-crop: {no_fit} sources skipped because no cuttable edge was found")
    return new_rows, generated, skipped


# ==========================================================================
# Mixed-opacity (mixed) — threshold-tested selection + augmented copies
# ==========================================================================
def is_mixed_opacity(alpha: np.ndarray) -> bool:
    """Does the GT contain both solid (alpha > 0.9 ratio >= 8%) and soft
    (0.05 < alpha < 0.95 ratio >= 8%) pixels? (The solid-part-of-a-transparent-
    object scenario.)"""
    solid = float((alpha > SOLID_ALPHA_THRESH).mean())
    soft = float(((alpha > SOFT_LO) & (alpha < SOFT_HI)).mean())
    return solid >= SOLID_MIN_RATIO and soft >= SOFT_MIN_RATIO


def select_mixed_sources(
    stems: list[str],
    category_by_stem: dict[str, str],
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    max_sources: int,
) -> list[str]:
    """Scans the `transparent` category stems IN ORDER and returns the first
    `max_sources` stems that pass the `is_mixed_opacity` test (deterministic).
    If one of the output copies already exists on disk, eligibility is
    accepted without loading the GT (since only eligible sources can produce
    output, file existence is proof — to avoid rescanning thousands of PNGs
    on resume)."""
    chosen: list[str] = []
    if max_sources <= 0:
        return chosen
    for stem in stems:
        if len(chosen) >= max_sources:
            break
        if category_by_stem.get(stem) != "transparent" or _DERIVED_SUFFIX_RE.search(stem):
            continue
        if any(
            (out_im_dir / f"{stem}_m{ci:02d}.jpg").exists()
            and (out_gt_dir / f"{stem}_m{ci:02d}.png").exists()
            for ci in range(MIXED_COPIES)
        ):
            chosen.append(stem)
            continue
        alpha = _load_alpha(train_gt_dir / f"{stem}.png")
        if is_mixed_opacity(alpha):
            chosen.append(stem)
    return chosen


def gen_mixed(
    sources: list[str],
    train_im_dir: Path,
    train_gt_dir: Path,
    out_im_dir: Path,
    out_gt_dir: Path,
    category_by_stem: dict[str, str],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """`MIXED_COPIES` augmented copies per source (`_m00`, `_m01`):
    `bgr.compositing.augment(..., flip_prob=0.0)` — color jitter / blur /
    JPEG artifacts affect only the RGB, the geometry does not change, the
    alpha is preserved AS IS (the augment signature was verified from the
    code: flip is the only geometric transform and it is disabled).
    Returns (rows, generated, skipped)."""
    new_rows: list[dict] = []
    generated = skipped = 0
    for stem in sources:
        category = category_by_stem[stem]
        pending: list[str] = []
        for ci in range(MIXED_COPIES):
            new_stem = f"{stem}_m{ci:02d}"
            if (out_im_dir / f"{new_stem}.jpg").exists() and (out_gt_dir / f"{new_stem}.png").exists():
                skipped += 1
                if new_stem not in existing_ids:
                    new_rows.append({"id": new_stem, "category": category})
                continue
            pending.append(new_stem)
        if not pending:
            continue
        rgb = _load_rgb(train_im_dir / f"{stem}.jpg")
        alpha = _load_alpha(train_gt_dir / f"{stem}.png", (rgb.shape[1], rgb.shape[0]))
        for new_stem in pending:
            rng = _item_rng(seed, new_stem)
            out_rgb, out_alpha = augment(rgb, alpha, rng, flip_prob=0.0)
            _save_pair(out_rgb, out_alpha, out_im_dir / f"{new_stem}.jpg", out_gt_dir / f"{new_stem}.png")
            new_rows.append({"id": new_stem, "category": category})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# Orchestration
# ==========================================================================
def run(
    train_im_dir: Path,
    train_gt_dir: Path,
    category_by_stem: dict[str, str],
    out_dir: Path,
    seed: int = 42,
    edge_count: int = DEFAULT_EDGE_COUNT,
    mixed_cap: int = DEFAULT_MIXED_CAP,
    out_manifest: Path | None = None,
    exclude_stems: set[str] | None = None,
) -> dict[str, int]:
    """Runs the two derivative generators; returns kind -> number of newly
    generated pairs (only entries >0 — same pattern as make_textfx.run()).

    `category_by_stem`: source stem -> category (from
    `train_composites_manifest.jsonl` on Drive, see
    train_colab_lib.load_stem_categories). Stems not in the map do not enter
    the source pool.
    `exclude_stems`: stems that must NOT be used as sources (VAL leak guard —
    the caller derives it from val_stems.json).
    `mixed_cap`: TOTAL cap on mixed copies (the source-count cap is
    `mixed_cap / MIXED_COPIES`); mixed total = number of eligible pairs × 2,
    clipped by the cap."""
    train_im_dir, train_gt_dir = Path(train_im_dir), Path(train_gt_dir)
    out_dir = Path(out_dir)
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    stems = _list_pair_stems(train_im_dir, train_gt_dir)
    if exclude_stems:
        stems = [s for s in stems if s not in exclude_stems]

    all_rows: list[dict] = []
    result: dict[str, int] = {}
    total_skipped = 0

    if edge_count > 0:
        sources = select_edge_sources(stems, category_by_stem, edge_count)
        n_pref = sum(1 for s, c in sources if _is_preferred_source(s, c))
        print(f"edge-crop sources: {len(sources)} (preferred/real-background: {n_pref})")
        rows, generated, skipped = gen_edge_crops(
            sources, train_im_dir, train_gt_dir, out_im_dir, out_gt_dir, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["edge"] = generated

    if mixed_cap > 0:
        mixed_sources = select_mixed_sources(
            stems, category_by_stem, train_gt_dir, out_im_dir, out_gt_dir,
            max_sources=mixed_cap // MIXED_COPIES,
        )
        print(f"mixed sources: {len(mixed_sources)} (copy target: {len(mixed_sources) * MIXED_COPIES})")
        rows, generated, skipped = gen_mixed(
            mixed_sources, train_im_dir, train_gt_dir, out_im_dir, out_gt_dir,
            category_by_stem, seed, existing_ids,
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["mixed"] = generated

    # only new ids go to the manifest (including an in-run safety dedup — make_textfx pattern)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in all_rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{sum(result.values())} new pairs written, {total_skipped} already existed (skipped)")
    for kind, n in sorted(result.items()):
        print(f"{kind}: {n}")
    return result


def _load_categories(path: Path) -> dict[str, str]:
    """Stem -> category map from a JSONL manifest (lines are expected to have
    at least `id` + `category` — the train_composites_manifest.jsonl schema)."""
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
    parser.add_argument("--edge-count", type=int, default=DEFAULT_EDGE_COUNT)
    parser.add_argument("--mixed-cap", type=int, default=DEFAULT_MIXED_CAP)
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
        edge_count=args.edge_count,
        mixed_cap=args.mixed_cap,
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
        exclude_stems=exclude_stems,
    )


if __name__ == "__main__":
    main()
