"""Generator for the v7 `design` category — synthetic print-design / sticker /
t-shirt-graphic style samples (GitHub issue #2: on designs with halftone, ink
texture, smoky edges and glows melting into white, the model erases or ghosts
the subject; this category closes that style-domain gap).

Each sample is a "print design" composition:

- **Background**: paper-white/cream flat color (245-255 band) or a light
  paper texture (low-amplitude noise); with `PASTEL_BG_PROB` (15%)
  probability a light pastel flat color. The background is alpha=0 in the
  GT — additionally, a `MARGIN_FRAC`-wide band along the outer canvas edge
  is ZEROED out of every element's alpha (the print "margin"; GT corners are
  always 0 — a test contract).
- **Stylized subject (1-2)**: a cutout taken from the `fg_dirs`
  (trans460/HIM2K, im/+gt/ pairs) and `toonout_dir` pools gets a PRINT-STYLE
  filter — CRITICAL: the filter touches ONLY the RGB, the alpha stays AS IS
  (bit-identical) (`apply_print_filter`). Filter menu: (a) halftone (turns
  luminance into a dot grid — dot radius scales with DARKNESS, the classic
  newspaper screen), (b) posterize (3-5 levels) + saturation boost, (c)
  high-contrast "ink" (thresholding + edge emphasis), (d) no filter (25%).
  ToonOut sources are already illustrations — they mostly get
  no-filter/posterize.
- **Smoky edge / airbrush** (`_smoke_alpha`): cloud/smoke blotches curling
  OUTWARD from the subject's alpha are ADDED in the `SMOKE_LO..SMOKE_HI`
  (0.1-0.5) band — and enter the GT AS IS (the smoke is part of the design;
  this texture is exactly why Reddit Photo 1 got erased). The organic look
  comes from a two-octave value-noise ("Perlin-like") mask.
- **Glow/burst** (`RAY_PROB`=50%): a radial ray burst or glow BEHIND the
  subject — semi-transparent in the GT
  (`RAY_ALPHA_LO..RAY_ALPHA_HI` = 0.15-0.6).
- **Display text (1-2 blocks)**: make_textfx's text machinery is REUSED
  (`_get_font`/`_draw_text_rgba`/`_rand_text` are imported, not copied).
  Extras: CURVED text (letters placed one by one on an arc —
  `_curved_text_rgba`), stacked multi-line blocks (`_stacked_text_rgba`),
  distressing (chipping pieces out of the text alpha with a value-noise
  grunge mask — reflected in the GT as is). Placement: top and/or bottom
  band.
- **Small decorations**: star/lightning-bolt/splatter marks (2-6, solid or
  semi-transparent) — simple vector drawings.
- **GT = the alpha UNION of all elements** (the `1-(1-a)(1-b)` chain); the
  background never contributes.

CONTRACTS (EXACTLY THE SAME as scripts/make_textfx.py):
- Stem pattern `design_{i:05d}_c00`; output `out_dir/im/{stem}.jpg` (JPEG
  q92) + `out_dir/gt/{stem}.png` (mode L) — `_save_pair` imported from
  make_textfx.
- Manifest: `{"id": stem, "category": "design"}` lines APPENDED to JSONL.
- Determinism: `_item_rng(seed, stem)` — same seed + same stem ->
  bit-identical output, independent of processing order (resume safety).
  CAUTION: if the source pool (fg_dirs contents / exclude_fg_stems) changes,
  the output changes too — pool selection indices are resolved against the
  pool list.
- Idempotency: if the im+gt pair exists on disk, generation is skipped; if
  the file exists but the manifest line is missing, only the line is
  completed.

The `bg_dir` parameter is accepted for signature parity with
make_textfx.run() but is NOT used — the background is fully synthetic
(paper/pastel).

Usage:
    uv run python scripts/make_design.py --out-dir data/train_design \
        --fg-dirs data/raw_train/trans460_pairs data/raw_train/him2k_merged \
        --toonout-dir /content/downloads/toonout --font-dir /content/fonts \
        --seed 42 --count 6000
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# In the SAME directory as make_textfx (scripts/) — imported via the script
# directory on the CLI, and via the scripts/ entry added to sys.path on the
# Colab/test side. The text/effect machinery and shared contract helpers are
# NOT copied, they are imported.
from make_textfx import (  # noqa: F401  (re-exported shared helpers)
    _append_manifest,
    _bright_color,
    _draw_text_rgba,
    _get_font,
    _item_rng,
    _load_alpha,
    _load_font_paths,
    _load_manifest_ids,
    _load_rgb_capped,
    _pairs_from_dir,
    _rand_text,
    _save_pair,
    _star_points,
)
from make_textfx import _CHARS

# Source pools may contain 100MP+ images (see the same note in make_textfx).
Image.MAX_IMAGE_PIXELS = None

DEFAULT_COUNT = 6000
DEFAULT_CANVAS = (448, 768)

MARGIN_FRAC = 0.02          # canvas edge band — always 0 in the GT (corner guarantee)
PASTEL_BG_PROB = 0.15       # probability of a light pastel flat background
PAPER_NOISE_PROB = 0.5      # probability of light texture on the paper-white branch

FILTER_NONE_PROB = 0.25     # no-filter share for normal sources (last branch of the menu)
SUBJECT_FRAC_LO, SUBJECT_FRAC_HI = 0.35, 0.7  # subject long side / canvas short side
SECOND_SUBJECT_PROB = 0.35
TOON_SUBJECT_PROB = 0.35    # share of subjects drawn from ToonOut when both pools are non-empty

SMOKE_LO, SMOKE_HI = 0.1, 0.5      # smoke alpha band (enters the GT as is)
RAY_PROB = 0.5
RAY_ALPHA_LO, RAY_ALPHA_HI = 0.15, 0.6

CURVED_TEXT_PROB = 0.4      # curved text share
STACKED_TEXT_PROB = 0.35    # (if curved was not picked) stacked multi-line block share
DISTRESS_PROB = 0.5         # probability of the distress/grunge mask
SECOND_TEXT_PROB = 0.5      # probability of a second text band
DECOR_RANGE = (2, 6)        # number of decorations (inclusive-inclusive)

# Halftone/ink "paint" palette: classic black + single-color print inks.
_INK_COLORS: list[tuple[int, int, int]] = [
    (18, 18, 18), (120, 20, 30), (20, 40, 120), (26, 84, 46), (90, 30, 110),
]
_PAPER_RGB = (250, 249, 245)


# ==========================================================================
# Noise helpers — "Perlin-like" masks for smoke and grunge
# (two-octave value-noise; derived from make_textfx's gaussian tooling pattern).
# ==========================================================================
def _value_noise(rng: np.random.Generator, h: int, w: int, cell_px: int) -> np.ndarray:
    """Bilinear upscaling of a coarse random grid — [0,1] float32 (H, W)."""
    gh = max(2, round(h / max(1, cell_px)))
    gw = max(2, round(w / max(1, cell_px)))
    grid = (rng.uniform(0.0, 1.0, (gh, gw)) * 255).astype(np.uint8)
    up = Image.fromarray(grid, mode="L").resize((w, h), Image.BILINEAR)
    return np.asarray(up, dtype=np.float32) / 255.0


def _perlin_noise(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Two-octave value-noise normalized to [0,1] — cloud/smoke/grunge texture."""
    n = 0.65 * _value_noise(rng, h, w, max(8, min(h, w) // 6)) + 0.35 * _value_noise(
        rng, h, w, max(3, min(h, w) // 18)
    )
    n -= n.min()
    mx = float(n.max())
    return (n / mx).astype(np.float32) if mx > 0 else n.astype(np.float32)


# ==========================================================================
# Background — paper white / cream / light pastel (alpha=0 in the GT)
# ==========================================================================
def _design_bg(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Print background (H, W, 3 uint8): 245-255-band paper white/cream
    (sometimes lightly textured) or, with 15% probability, a light pastel
    flat color."""
    w, h = size
    if rng.uniform() < PASTEL_BG_PROB:
        col = (255 - rng.integers(12, 60, 3)).astype(np.float32)  # light pastel
        arr = np.broadcast_to(col, (h, w, 3)).astype(np.float32).copy()
        return arr.round().clip(0, 255).astype(np.uint8)

    base = float(rng.integers(245, 256))
    col = np.array([base, base, base], dtype=np.float32)
    if rng.uniform() < 0.5:  # cream tone: the blue channel is slightly reduced
        col[1] -= float(rng.uniform(0.0, 4.0))
        col[2] -= float(rng.uniform(2.0, 10.0))
    arr = np.broadcast_to(col, (h, w, 3)).astype(np.float32).copy()
    if rng.uniform() < PAPER_NOISE_PROB:  # low-amplitude paper texture
        amp = float(rng.uniform(1.5, 4.0))
        arr += amp * rng.standard_normal((h, w, 1)).astype(np.float32)
    return arr.round().clip(0, 255).astype(np.uint8)


# ==========================================================================
# Print-style filters — RGB ONLY; the alpha is returned AS IS (bit-identical)
# ==========================================================================
def _luminance(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> [0,1] float32 luminance."""
    f = rgb.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) / 255.0


def _filter_halftone(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Classic newspaper screen: per-cell mean luminance -> dot radius
    (dark region = large ink dot). Fully vectorized — no per-cell loop."""
    h, w = rgb.shape[:2]
    cell = int(rng.integers(4, 11))
    lum = _luminance(rgb)
    ch, cw = math.ceil(h / cell), math.ceil(w / cell)
    pad_lum = np.pad(lum, ((0, ch * cell - h), (0, cw * cell - w)), mode="edge")
    lum_c = pad_lum.reshape(ch, cell, cw, cell).mean(axis=(1, 3))
    radius_c = (cell / 2.0) * np.sqrt(np.clip(1.0 - lum_c, 0.0, 1.0)) * float(
        rng.uniform(1.05, 1.35)
    )
    radius = np.repeat(np.repeat(radius_c, cell, axis=0), cell, axis=1)[:h, :w]
    ly = (np.arange(h, dtype=np.float32) % cell) - (cell - 1) / 2.0
    lx = (np.arange(w, dtype=np.float32) % cell) - (cell - 1) / 2.0
    d2 = ly[:, None] ** 2 + lx[None, :] ** 2
    dot = d2 <= radius**2
    ink = np.asarray(_INK_COLORS[int(rng.integers(0, len(_INK_COLORS)))], dtype=np.uint8)
    paper = np.asarray(_PAPER_RGB, dtype=np.uint8)
    return np.where(dot[..., None], ink[None, None, :], paper[None, None, :])


def _filter_posterize(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Posterize (3-5 levels) + saturation boost."""
    levels = int(rng.integers(3, 6))
    step = 256.0 / levels
    q = (np.floor(rgb.astype(np.float32) / step) * (255.0 / (levels - 1))).clip(0, 255)
    im = Image.fromarray(q.astype(np.uint8), mode="RGB")
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(1.2, 1.8)))
    return np.asarray(im, dtype=np.uint8)


def _filter_ink(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """High-contrast 'ink': luminance thresholding + light edge emphasis."""
    lum = _luminance(rgb)
    thresh = float(rng.uniform(0.35, 0.6))
    gray8 = (lum * 255).astype(np.uint8)
    edges = np.asarray(
        Image.fromarray(gray8, mode="L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    ) / 255.0
    ink_mask = (lum <= thresh) | (edges > 0.3)
    ink = np.asarray(_INK_COLORS[int(rng.integers(0, len(_INK_COLORS)))], dtype=np.uint8)
    paper = np.asarray(_PAPER_RGB, dtype=np.uint8)
    return np.where(ink_mask[..., None], ink[None, None, :], paper[None, None, :])


def apply_print_filter(
    rgb: np.ndarray, alpha: np.ndarray, rng: np.random.Generator, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Applies the print-style filter to the RGB ONLY; the alpha is returned
    AS IS (same array, bit-identical) — the critical contract of this
    category's design: the filter changes the style, not the transparency
    ground truth."""
    if kind == "halftone":
        return _filter_halftone(rgb, rng), alpha
    if kind == "posterize":
        return _filter_posterize(rgb, rng), alpha
    if kind == "ink":
        return _filter_ink(rgb, rng), alpha
    return rgb, alpha  # "none"


def _pick_filter(rng: np.random.Generator, is_toon: bool) -> str:
    """Filter menu: 4 equal branches for normal sources (25% no filter);
    ToonOut is already illustration, so it mostly gets no-filter/posterize."""
    u = float(rng.uniform())
    if is_toon:
        if u < 0.5:
            return "none"
        return "posterize" if u < 0.9 else "ink"
    if u < 0.25:
        return "halftone"
    if u < 0.5:
        return "posterize"
    if u < 1.0 - FILTER_NONE_PROB:
        return "ink"
    return "none"


# ==========================================================================
# Smoky edge / airbrush — semi-transparent smoke blotches curling outward from the alpha
# ==========================================================================
def _smoke_alpha(
    alpha: np.ndarray, rng: np.random.Generator, reach_frac: float | None = None
) -> np.ndarray:
    """Smoke/cloud blotches curling OUTWARD from the object boundary: in the
    [SMOKE_LO, SMOKE_HI] band, always 0 INSIDE the object (alpha > 0.05).
    Envelope = a gaussian outward blur of the alpha; texture = Perlin-like
    value-noise (blotchy cutoff + light blur for an organic edge)."""
    h, w = alpha.shape
    if reach_frac is None:
        reach_frac = float(rng.uniform(0.05, 0.12))
    reach = max(2.0, min(h, w) * reach_frac)
    soft = np.asarray(
        Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(reach)
        ),
        dtype=np.float32,
    ) / 255.0
    envelope = np.clip(soft * float(rng.uniform(1.6, 2.4)), 0.0, 1.0)
    noise = _perlin_noise(rng, h, w)
    smoke = (SMOKE_LO + (SMOKE_HI - SMOKE_LO) * noise) * envelope
    smoke = smoke * (noise > float(rng.uniform(0.25, 0.45)))  # blotchy/fragmented cutoff
    smoke = np.asarray(
        Image.fromarray((smoke * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(1.0)
        ),
        dtype=np.float32,
    ) / 255.0
    smoke[alpha > 0.05] = 0.0  # smoke only OUTWARD — the GT inside the object is unchanged
    return np.clip(smoke, 0.0, SMOKE_HI).astype(np.float32)


# ==========================================================================
# Stylized subject layer — cutout + print filter + smoke (alphas as separate elements)
# ==========================================================================
def _resize_pair(
    rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """size = (w, h); RGB LANCZOS, alpha BILINEAR (the make_composites pattern)."""
    rgb_r = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize(size, Image.LANCZOS), dtype=np.uint8
    )
    a_r = np.asarray(
        Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L").resize(
            size, Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    return rgb_r, a_r


def _subject_layers(
    rng: np.random.Generator,
    pair: tuple[Path, Path],
    is_toon: bool,
    canvas_min: int,
    max_w: int,
    max_h: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int]] | None:
    """Element list for a single subject (smoke + body) and the layer size (lw, lh).

    Elements: [(smoke_rgb, smoke_alpha), (subject_rgb, subject_alpha)] — the
    smoke is composited first, the subject sits on top; the GT union contains
    both. Returns None for a source with an empty alpha (the subject is
    skipped)."""
    im_path, gt_path = pair
    rgb = _load_rgb_capped(im_path)
    alpha = _load_alpha(gt_path, (rgb.shape[1], rgb.shape[0]))
    ys, xs = np.nonzero(alpha > 0.05)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    rgb_c, a_c = rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]

    # Print-style filter — RGB ONLY (the apply_print_filter contract).
    rgb_c, a_c = apply_print_filter(rgb_c, a_c, rng, _pick_filter(rng, is_toon))

    # Scale: the subject's long side is 35-70% of the canvas short side.
    target = canvas_min * float(rng.uniform(SUBJECT_FRAC_LO, SUBJECT_FRAC_HI))
    reach_frac = float(rng.uniform(0.05, 0.12))
    ch, cw = a_c.shape
    scale = target / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    rgb_s, a_s = _resize_pair(rgb_c, a_c, (new_w, new_h))

    pad = max(2, int(round(2.5 * reach_frac * min(new_w, new_h))))
    lw, lh = new_w + 2 * pad, new_h + 2 * pad
    a_p = np.zeros((lh, lw), dtype=np.float32)
    a_p[pad : pad + new_h, pad : pad + new_w] = a_s
    subj_rgb = np.zeros((lh, lw, 3), dtype=np.float32)
    subj_rgb[pad : pad + new_h, pad : pad + new_w] = rgb_s.astype(np.float32)

    smoke = _smoke_alpha(a_p, rng, reach_frac)
    smoke_col = np.clip(
        float(rng.integers(150, 236)) + rng.uniform(-12.0, 12.0, 3), 0, 255
    ).astype(np.float32)
    smoke_rgb = np.broadcast_to(smoke_col, (lh, lw, 3)).astype(np.float32).copy()

    layers = [(smoke_rgb, smoke), (subj_rgb, a_p)]

    # If it does not fit the canvas, the layers are scaled down proportionally (smoke included).
    if lw > max_w or lh > max_h:
        f = min(max_w / lw, max_h / lh)
        nw, nh = max(1, int(lw * f)), max(1, int(lh * f))
        resized = []
        for l_rgb, l_a in layers:
            r2, a2 = _resize_pair(l_rgb.round().clip(0, 255).astype(np.uint8), l_a, (nw, nh))
            resized.append((r2.astype(np.float32), a2))
        layers, (lw, lh) = resized, (nw, nh)
    return layers, (lw, lh)


# ==========================================================================
# Glow/burst — a radial ray burst or glow behind the subject (semi-transparent)
# ==========================================================================
def _ray_layer(
    rng: np.random.Generator, size: tuple[int, int], center: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Canvas-sized (rgb, alpha) element: a radial ray burst (sunburst) or a
    gaussian glow. Alpha is semi-transparent in the [RAY_ALPHA_LO,
    RAY_ALPHA_HI] band — it enters the GT as is (the edge band is zeroed
    separately during compositing)."""
    w, h = size
    cx, cy = center
    val = float(rng.uniform(RAY_ALPHA_LO, RAY_ALPHA_HI))
    color = np.asarray(
        (255, 255, 255) if rng.uniform() < 0.4 else _bright_color(rng), dtype=np.float32
    )
    if rng.uniform() < 0.5:  # glow: soft radiance melting into white
        sigma = min(w, h) * float(rng.uniform(0.08, 0.18))
        yy = (np.arange(h, dtype=np.float32) - cy)[:, None]
        xx = (np.arange(w, dtype=np.float32) - cx)[None, :]
        rr2 = xx**2 + yy**2
        a = val * np.exp(-rr2 / (2 * sigma**2))
        # Compact support (v8): fade to EXACT 0 between 1.8σ and 2.5σ. The
        # untruncated gaussian tail put a faint non-zero alpha across the
        # whole canvas GT, teaching the model that a wide haze around any
        # subject should be kept — which surfaced as gray background smears
        # on real photos (HF discussion #1, the cat masks).
        fade = np.clip((2.5 * sigma - np.sqrt(rr2)) / (0.7 * sigma), 0.0, 1.0)
        a = (a * fade).astype(np.float32)
    else:  # sunburst: evenly spaced ray wedges
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        n = int(rng.integers(8, 17))
        rot0 = float(rng.uniform(0, 2 * math.pi))
        r = 0.5 * min(w, h) * float(rng.uniform(0.7, 1.1))
        half = (math.pi / n) * float(rng.uniform(0.25, 0.45))
        for k in range(n):
            ang = rot0 + k * 2 * math.pi / n
            p1 = (cx + r * math.cos(ang - half), cy + r * math.sin(ang - half))
            p2 = (cx + r * math.cos(ang + half), cy + r * math.sin(ang + half))
            d.polygon([(cx, cy), p1, p2], fill=255)
        a = (np.asarray(mask, dtype=np.float32) / 255.0) * val
    rgb = np.broadcast_to(color, (h, w, 3)).astype(np.float32).copy()
    return rgb, a.astype(np.float32)


# ==========================================================================
# Small decorations — star / lightning bolt / splatter marks (simple vector drawings)
# ==========================================================================
_BOLT_PTS = [(0.45, 0.0), (0.62, 0.0), (0.46, 0.42), (0.68, 0.42),
             (0.30, 1.0), (0.44, 0.55), (0.26, 0.55)]


def _decor_layer(
    rng: np.random.Generator, size: tuple[int, int], margin: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Draws 2-6 decorations onto a single RGBA layer; None if there are none."""
    w, h = size
    n = int(rng.integers(DECOR_RANGE[0], DECOR_RANGE[1] + 1))
    if n <= 0:
        return None
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        s = min(w, h) * float(rng.uniform(0.025, 0.07))
        lo_x, hi_x = margin + s, w - margin - s
        lo_y, hi_y = margin + s, h - margin - s
        cx = float(rng.uniform(lo_x, hi_x)) if hi_x > lo_x else w / 2
        cy = float(rng.uniform(lo_y, hi_y)) if hi_y > lo_y else h / 2
        a = int(float(rng.uniform(0.4, 1.0)) * 255)  # solid or semi-transparent
        col = _bright_color(rng) + (a,)
        kind = int(rng.integers(0, 3))
        if kind == 0:  # star
            d.polygon(_star_points(cx, cy, s, s * 0.45, n=int(rng.integers(4, 7))), fill=col)
        elif kind == 1:  # lightning bolt
            ang = float(rng.uniform(-0.5, 0.5))
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            for px, py in _BOLT_PTS:
                dx, dy = (px - 0.45) * 2 * s, (py - 0.5) * 2 * s
                pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
            d.polygon(pts, fill=col)
        else:  # splatter mark: central drop + satellite droplets
            r0 = s * 0.55
            d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=col)
            for _ in range(int(rng.integers(3, 8))):
                ang = float(rng.uniform(0, 2 * math.pi))
                dist = s * float(rng.uniform(0.7, 1.1))
                rr = r0 * float(rng.uniform(0.15, 0.4))
                px, py = cx + dist * math.cos(ang), cy + dist * math.sin(ang)
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=col)
    arr = np.asarray(layer, dtype=np.float32)
    return arr[..., :3], arr[..., 3] / 255.0


# ==========================================================================
# Display text — make_textfx text machinery + curve / stacking / distress
# ==========================================================================
def _word(rng: np.random.Generator, lo: int = 4, hi: int = 10) -> str:
    n = int(rng.integers(lo, hi))
    return "".join(_CHARS[int(rng.integers(0, len(_CHARS)))] for _ in range(n))


def _curved_text_rgba(
    text: str,
    font,
    fill: tuple[int, int, int, int],
    theta: float,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] | None = None,
) -> Image.Image:
    """CURVED display text: letters are placed one by one on an arch (apex at
    the top) and rotated to the arc's tangent. `theta` is the total arc angle
    (radians). Deterministic (takes no rng) — tests call it directly."""
    text = text.strip() or "A"
    theta = max(0.15, min(float(theta), 2.4))
    pad = max(2, stroke_width + 2)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    space_w = max(2, int(probe.textlength("i", font=font)))
    glyphs: list[tuple[Image.Image | None, float]] = []
    for ch in text:
        if ch.isspace():
            glyphs.append((None, float(space_w)))
            continue
        img = _draw_text_rgba(ch, font, fill, stroke_width, stroke_fill, pad)
        glyphs.append((img, float(img.width - 2 * pad)))
    tracking = 2.0
    total_w = sum(wch for _, wch in glyphs) + tracking * max(0, len(glyphs) - 1)
    radius = total_w / theta
    gh_max = max((img.height for img, _ in glyphs if img is not None), default=8)
    sag = radius * (1 - math.cos(theta / 2))
    cw = int(total_w + 2 * gh_max)
    chh = int(sag + 2 * gh_max)
    canvas = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
    cx = cw / 2.0
    cy = gh_max * 0.5 + radius  # circle center; arc apex at y ~= gh_max*0.5
    cum = 0.0
    for img, wch in glyphs:
        phi = -theta / 2 + (cum + wch / 2) / radius
        cum += wch + tracking
        if img is None:
            continue
        rot = img.rotate(-math.degrees(phi), expand=True, resample=Image.BICUBIC)
        gx = cx + radius * math.sin(phi)
        gy = cy - radius * math.cos(phi)
        canvas.alpha_composite(
            rot, (int(round(gx - rot.width / 2)), int(round(gy - rot.height / 2)))
        )
    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


def _stacked_text_rgba(
    rng: np.random.Generator,
    font,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int] | None,
) -> Image.Image:
    """Stacked multi-line display block: 2-3 lines, centered."""
    pad = max(2, stroke_width + 2)
    lines = []
    for _ in range(int(rng.integers(2, 4))):
        n_words = 1 if rng.uniform() < 0.7 else 2
        text = " ".join(_word(rng, 3, 8) for _ in range(n_words))
        lines.append(_draw_text_rgba(text, font, fill, stroke_width, stroke_fill, pad))
    gap = max(1, int(0.12 * max(im.height for im in lines)))
    bw = max(im.width for im in lines)
    bh = sum(im.height for im in lines) + gap * (len(lines) - 1)
    block = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    y = 0
    for im in lines:
        block.alpha_composite(im, ((bw - im.width) // 2, y))
        y += im.height + gap
    return block


def _distress(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Distressing: chips pieces out of the text alpha with a value-noise
    grunge mask (reflected in the GT as is — the missing piece IS the design)."""
    arr = np.array(img)
    noise = _perlin_noise(rng, arr.shape[0], arr.shape[1])
    keep = noise > float(rng.uniform(0.15, 0.35))
    arr[..., 3] = (arr[..., 3] * keep).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _ink_or_bright(rng: np.random.Generator) -> tuple[int, int, int]:
    """Print text color: 50% dark ink, 50% bright display color."""
    if rng.uniform() < 0.5:
        c = rng.integers(0, 70, 3)
        return (int(c[0]), int(c[1]), int(c[2]))
    return _bright_color(rng)


def _text_block(
    rng: np.random.Generator, canvas_size: tuple[int, int], font_paths: list[Path]
) -> Image.Image:
    """A single display text block: curved / stacked / single line (+distress)."""
    cmin = min(canvas_size)
    font_size = max(10, int(cmin * float(rng.uniform(0.07, 0.16))))
    font = _get_font(font_paths, rng, font_size)
    fill = _ink_or_bright(rng) + (255,)
    stroke_width, stroke_fill = 0, None
    if rng.uniform() < 0.4:
        stroke_width = max(1, font_size // 12)
        stroke_fill = _ink_or_bright(rng)
    u = float(rng.uniform())
    if u < CURVED_TEXT_PROB:
        theta = float(rng.uniform(0.5, 1.6))
        img = _curved_text_rgba(_word(rng).upper(), font, fill, theta, stroke_width, stroke_fill)
    elif u < CURVED_TEXT_PROB + STACKED_TEXT_PROB:
        img = _stacked_text_rgba(rng, font, fill, stroke_width, stroke_fill)
    else:
        img = _draw_text_rgba(
            _rand_text(rng), font, fill, stroke_width, stroke_fill, max(2, stroke_width + 2)
        )
    if rng.uniform() < DISTRESS_PROB:
        img = _distress(img, rng)
    return img


# ==========================================================================
# Sample composition — GT = the alpha union of all elements, background alpha=0
# ==========================================================================
def _paste_element(
    rgb_small: np.ndarray, a_small: np.ndarray, x0: int, y0: int, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Embeds a small layer into a canvas-sized (rgb float, alpha float) element."""
    w, h = size
    sh, sw = a_small.shape
    rgb_full = np.zeros((h, w, 3), dtype=np.float32)
    a_full = np.zeros((h, w), dtype=np.float32)
    rgb_full[y0 : y0 + sh, x0 : x0 + sw] = rgb_small
    a_full[y0 : y0 + sh, x0 : x0 + sw] = a_small
    return rgb_full, a_full


def _rgba_to_element(img: Image.Image, x0: int, y0: int, size: tuple[int, int]):
    arr = np.asarray(img, dtype=np.float32)
    return _paste_element(arr[..., :3], arr[..., 3] / 255.0, x0, y0, size)


def _fit_rgba(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    if img.width <= max_w and img.height <= max_h:
        return img
    f = min(max_w / img.width, max_h / img.height)
    return img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)


def _zero_alpha_in_rects(
    a: np.ndarray, rects: list[tuple[int, int, int, int]], pad: int
) -> None:
    """Zeroes `a` (H, W float alpha, in place) inside each (x0, y0, x1, y1)
    rect grown by `pad` px — used to keep the glow out of the text bands."""
    for x0, y0, x1, y1 in rects:
        a[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = 0.0


def _text_bands(rng: np.random.Generator) -> list[str]:
    """Which band(s) get a text block — identical draw order to the pre-v8
    inline code (the second uniform is only consumed when there is ONE band)."""
    n_text = 1 + (1 if rng.uniform() < SECOND_TEXT_PROB else 0)
    if n_text == 2:
        return ["top", "bottom"]
    return ["top"] if rng.uniform() < 0.5 else ["bottom"]


def _render_design_sample(
    rng: np.random.Generator,
    size: tuple[int, int],
    fg_pairs: list[tuple[Path, Path]],
    toon_pairs: list[tuple[Path, Path]],
    font_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """A single design sample: (composite RGB uint8, alpha float32 [0,1]).

    Element order (bottom -> top): glow -> subject(s; smoke + body) ->
    decorations -> text blocks. The GT is the union of all element alphas;
    the canvas's MARGIN_FRAC edge band is zeroed in every element (background
    corners are always 0 in the GT)."""
    w, h = size
    m = max(2, int(MARGIN_FRAC * min(w, h)))
    bg = _design_bg(rng, size)

    elements: list[tuple[np.ndarray, np.ndarray]] = []
    subject_elements: list[tuple[np.ndarray, np.ndarray]] = []
    centers: list[tuple[float, float]] = []

    # 1) Stylized subject(s) — 1-2 (0 if the pools are empty; e.g. text-only in tests).
    n_sub = (1 + (1 if rng.uniform() < SECOND_SUBJECT_PROB else 0)) if (fg_pairs or toon_pairs) else 0
    for _ in range(n_sub):
        if toon_pairs and fg_pairs:
            use_toon = rng.uniform() < TOON_SUBJECT_PROB
        else:
            use_toon = bool(toon_pairs)
        pool = toon_pairs if use_toon else fg_pairs
        pair = pool[int(rng.integers(0, len(pool)))]
        built = _subject_layers(rng, pair, use_toon, min(w, h), w - 2 * m, h - 2 * m)
        if built is None:
            continue
        layers, (lw, lh) = built
        x0 = int(rng.integers(m, max(m, w - m - lw) + 1))
        y0 = int(rng.integers(m, max(m, h - m - lh) + 1))
        for l_rgb, l_a in layers:
            subject_elements.append(_paste_element(l_rgb, l_a, x0, y0, size))
        centers.append((x0 + lw / 2.0, y0 + lh / 2.0))

    # 2) Glow/burst — BEHIND the subject (50%). Held aside: its alpha is
    # zeroed under the text bands below, BEFORE it joins the element list.
    ray_el: tuple[np.ndarray, np.ndarray] | None = None
    if centers and rng.uniform() < RAY_PROB:
        ray_el = _ray_layer(rng, size, centers[0])
    elements += subject_elements

    # 3) Small decorations.
    decor = _decor_layer(rng, size, m)
    if decor is not None:
        elements.append(decor)

    # 4) Display text blocks — top and/or bottom band.
    text_rects: list[tuple[int, int, int, int]] = []  # (x0, y0, x1, y1)
    for band in _text_bands(rng):
        img = _fit_rgba(_text_block(rng, size, font_paths), max(1, int(0.9 * (w - 2 * m))),
                        max(1, int(0.28 * h)))
        tw, th = img.size
        x0 = int(round((w - tw) / 2 + float(rng.uniform(-0.08, 0.08)) * w))
        x0 = min(max(m, x0), max(m, w - m - tw))
        jitter = int(rng.integers(0, max(1, int(0.08 * h))))
        y0 = m + jitter if band == "top" else max(m, h - m - th - jitter)
        elements.append(_rgba_to_element(img, x0, y0, size))
        text_rects.append((x0, y0, x0 + tw, y0 + th))

    # Glow must not run under display text (v8): glow alpha between letters
    # entered the GT as "keep this semi-transparent", which the model
    # reproduced on real designs as white residue between the letters of
    # dense typography. Zeroing the glow inside the text rects keeps the
    # RGB and the GT consistent (no glow pixels are composited there either).
    if ray_el is not None:
        _zero_alpha_in_rects(ray_el[1], text_rects, pad=max(2, int(0.015 * min(w, h))))
        elements.insert(0, ray_el)

    # Composite + GT union (the edge band is zeroed in every element).
    out_rgb = bg.astype(np.float32)
    total_a = np.zeros((h, w), dtype=np.float32)
    for el_rgb, el_a in elements:
        el_a = el_a.copy()
        el_a[:m, :] = 0.0
        el_a[h - m :, :] = 0.0
        el_a[:, :m] = 0.0
        el_a[:, w - m :] = 0.0
        out_rgb = el_a[..., None] * el_rgb + (1 - el_a[..., None]) * out_rgb
        total_a = 1.0 - (1.0 - total_a) * (1.0 - el_a)
    return out_rgb.round().clip(0, 255).astype(np.uint8), total_a.astype(np.float32)


# ==========================================================================
# Generation loop + orchestration (the make_textfx.gen_text / run pattern)
# ==========================================================================
def gen_design(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    fg_pairs: list[tuple[Path, Path]],
    toon_pairs: list[tuple[Path, Path]],
    font_paths: list[Path],
    seed: int,
    existing_ids: set[str],
    canvas_range: tuple[int, int] = DEFAULT_CANVAS,
) -> tuple[list[dict], int, int]:
    """Returns (manifest rows, number of pairs generated, number of pairs skipped)."""
    new_rows: list[dict] = []
    generated = skipped = 0
    lo, hi = canvas_range
    for i in range(count):
        stem = f"design_{i:05d}_c00"
        img_path = out_im_dir / f"{stem}.jpg"
        gt_path = out_gt_dir / f"{stem}.png"
        row = {"id": stem, "category": "design"}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if stem not in existing_ids:
                new_rows.append(row)  # file exists, manifest line missing -> line only
            continue
        rng = _item_rng(seed, stem)
        w = int(rng.integers(lo, hi + 1))
        h = int(rng.integers(lo, hi + 1))
        rgb, alpha = _render_design_sample(rng, (w, h), fg_pairs, toon_pairs, font_paths)
        _save_pair(rgb, alpha, img_path, gt_path)
        new_rows.append(row)
        generated += 1
    return new_rows, generated, skipped


def run(
    out_dir: Path,
    bg_dir: Path | None = None,  # signature parity (make_textfx pattern) — NOT used
    fg_dirs: list[Path] | None = None,
    toonout_dir: Path | None = None,
    font_dir: Path | None = None,
    seed: int = 42,
    count: int = DEFAULT_COUNT,
    out_manifest: Path | None = None,
    canvas_range: tuple[int, int] = DEFAULT_CANVAS,
    exclude_fg_stems: set[str] | None = None,
) -> dict[str, int]:
    """Runs the design generator; returns {"design": newly generated} (only if
    >0 — the make_textfx.run() pattern). `bg_dir` is unused (synthetic background).

    `exclude_fg_stems`: raw fg stems that must NOT be used as sources (VAL
    leak guard — the caller derives it from val_stems.json; see
    training/v7_veri_guncelleme_hucresi.py). CAUTION: if the pool changes,
    the outputs of the same seed change too — the guard set must be kept
    constant across runs (resume must use the same set)."""
    out_dir = Path(out_dir)
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    fg_pairs: list[tuple[Path, Path]] = []
    for d in fg_dirs or []:
        fg_pairs += _pairs_from_dir(Path(d))
    toon_pairs = _pairs_from_dir(Path(toonout_dir)) if toonout_dir else []
    if exclude_fg_stems:
        fg_pairs = [p for p in fg_pairs if p[0].stem not in exclude_fg_stems]
        toon_pairs = [p for p in toon_pairs if p[0].stem not in exclude_fg_stems]
    if count > 0 and not (fg_pairs or toon_pairs):
        raise SystemExit(
            "no source im/gt pairs found for design (--fg-dirs roots must contain "
            "im/ + gt/ and/or --toonout-dir must be given)"
        )
    font_paths = _load_font_paths(Path(font_dir) if font_dir else None)

    rows, generated, skipped = gen_design(
        count, out_im_dir, out_gt_dir, fg_pairs, toon_pairs, font_paths, seed,
        existing_ids, canvas_range=canvas_range,
    )

    # only new ids go to the manifest (including an in-run safety dedup — make_textfx pattern)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{generated} new pairs written, {skipped} already existed (skipped)")
    return {"design": generated} if generated else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="output root (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--bg-dir", default=None,
                        help="accepted for signature parity — NOT used (synthetic background)")
    parser.add_argument(
        "--fg-dirs", nargs="*", default=[],
        help="subject source roots; each root must contain im/ + gt/ subdirectories (matched by stem)",
    )
    parser.add_argument("--toonout-dir", default=None, help="ToonOut root (im/ + gt/)")
    parser.add_argument("--font-dir", default=None,
                        help=".ttf/.otf/.ttc font pool (PIL default if absent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--out-manifest", default=None, help="default: <out-dir>/manifest.jsonl")
    parser.add_argument(
        "--exclude-stems-file", default=None,
        help="one raw fg stem per line (VAL leak guard) — not used as a source",
    )
    args = parser.parse_args()
    exclude = None
    if args.exclude_stems_file:
        exclude = {
            line.strip()
            for line in Path(args.exclude_stems_file).read_text().splitlines()
            if line.strip()
        }
    run(
        Path(args.out_dir),
        bg_dir=Path(args.bg_dir) if args.bg_dir else None,
        fg_dirs=[Path(d) for d in args.fg_dirs],
        toonout_dir=Path(args.toonout_dir) if args.toonout_dir else None,
        font_dir=Path(args.font_dir) if args.font_dir else None,
        seed=args.seed,
        count=args.count,
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
        exclude_fg_stems=exclude,
    )


if __name__ == "__main__":
    main()
