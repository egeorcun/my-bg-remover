"""Data generator for the 3 new v4 training categories (text / fx / illustration).

Designed to run on Colab (CPU is enough, NO GPU required) — `run()` is
importable (see the scripts/ import pattern in
`training/v3_veri_guncelleme_hucresi.py`) and can also be run via the CLI.

Categories:
- **text (~4,000):** synthetic text/logo renders with PIL. Random font
  (.ttf/.otf/.ttc glob inside `--font-dir`; falls back to the PIL default
  font), 1-3 random "brand-like" words, font size (5-40% of the canvas short
  side), color, position, rotation (±30°) and effects: stroke, drop shadow
  (offset+blur), glow (a SEMI-TRANSPARENT copy of the alpha expanded via
  MaxFilter+GaussianBlur, scaled by 0.35-0.8). Some samples get a simple
  vector badge behind the text (circle / rounded-rect / star) for a logo look.
  GT alpha = the render's OWN alpha (shadow+glow included) — it is NOT
  binarized. MID-ALPHA GUARANTEE: every sample is forced to have at least one
  soft effect (glow or blurred shadow); if neither was randomly picked, we
  fall back to glow — so every gt contains intermediate values outside 0/255
  (anti-aliasing alone does not count as a guarantee). Background: a randomly
  cropped real photo from the `--bg-dir` (BG-20k) pool, or with 20%
  probability (`FLAT_BG_PROB`) a flat/gradient color.
- **fx (~3,500):** procedural VFX glow around existing alpha-matted
  foregrounds under the `--fg-dirs` roots (each root has `im/` + `gt/`
  subdirectories, matched by stem): a glow halo (the object alpha blurred
  OUTWARD via MaxFilter + GaussianBlur — applied to EVERY sample, the fx leg
  of the mid-alpha guarantee), gaussian-kernel particle sparkles (some as
  4-armed stars), and thin lens-flare-like light streaks. Sparkle colors are
  bright (white/gold/cyan `_FX_PALETTE` + jitter), element alphas are
  semi-transparent, and the combined fx alpha is clipped at
  `FX_ALPHA_MAX`=0.9. New alpha = max(fg_alpha, fx_alpha); new RGB = the fg's
  composite onto a real background with the sparkle energy SCREEN-blended on
  top (out = 1-(1-base)(1-E), E = premultiplied color accumulated per element
  via screen) — the model learns that "the object + the sparkles around it
  are foreground together".
- **illustration (~3,600):** uses ToonOut's READY-MADE im/gt pairs
  (`--toonout-dir/im`, `--toonout-dir/gt`) — downloading the dataset is NOT
  this script's responsibility. 3 copies per pair: c00/c01 composited onto
  the bg pool via `bgr.compositing.compose` + `augment` (color jitter, JPEG
  artifacts — the EXACT same pattern as in make_composites.py), c02 = the
  original image (NO compose, augment only — same logic as make_composites'
  `_o00` copies). `ceil(count/3)` pairs suffice for the `count` target; the
  default target of 3600 was chosen to match ~50% of the ToonOut pool (the
  first N pairs are used, in deterministic order).

CONTRACTS (see scripts/make_composites.py):
- Filename stem pattern: `{category}_{index:05d}_c{copy:02d}` — for text the
  copy is always 00 (each index is an independent synthesis), for fx the
  index = source fg index and the copy is that source's copy ordinal, for
  illustration the index = ToonOut pair index with c00/c01 composited and
  c02 original. Manifest id = stem.
- Output layout: `out_dir/im/{stem}.jpg` (RGB, JPEG q92) + `out_dir/gt/
  {stem}.png` (mode-L 8-bit alpha) — `_save_pair` is identical to
  make_composites.
- Manifest: for each pair a `{"id": stem, "category": ...}` line is APPENDED
  to JSONL (`out_manifest`, default `out_dir/manifest.jsonl`). Because these
  categories are NOT in the benchmark.testset.CATEGORIES set (text/fx are
  new), that module's append_entries/load_manifest are DELIBERATELY not used.
- Determinism: `_item_rng(seed, stem)` — an exact copy of the
  np.random.SeedSequence pattern in make_composites.py; same seed + same
  stem -> bit-identical output, independent of processing order and of
  skipped items (resume safety).
- Idempotency: if the im+gt pair already exists on disk, generation is
  skipped; if the file exists but the manifest line is missing (interruption
  between save and append), only the line is completed — the file is NOT
  regenerated.
- Memory: images are processed one at a time; sources above 2048px are
  downscaled with LANCZOS (`PIL.Image.MAX_IMAGE_PIXELS = None` — see the
  same note in training/v3_veri_guncelleme_hucresi.py: 100MP+ academic
  sources).

Usage:
    uv run python scripts/make_textfx.py --out-dir data/train_textfx \
        --bg-dir data/backgrounds --fg-dirs data/raw_train/p3m data/raw_train/camo \
        --toonout-dir data/raw_train/toonout --font-dir data/fonts --seed 42 \
        --counts text=4000,fx=3500,illustration=3600
    # categories missing from --counts default to 0 (only the given ones are generated):
    uv run python scripts/make_textfx.py --out-dir out --bg-dir bgs --counts text=100
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from bgr.compositing import augment, compose

# ToonOut/fg sources may contain 100MP+ images; PIL's 179MP "decompression
# bomb" threshold is lifted for trusted sources (see module docstring).
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FONT_EXTS = {".ttf", ".otf", ".ttc"}
MAX_SIDE = 2048
DEFAULT_COUNTS: dict[str, int] = {"text": 4000, "fx": 3500, "illustration": 3600}
FLAT_BG_PROB = 0.2
BADGE_PROB = 0.35
FX_ALPHA_MAX = 0.9
ILLUSTRATION_COPIES = 3  # c00/c01 compose+augment, c02 original (augment only)
# Bright sparkle palette (0-1 RGB): white / gold / cyan range (+ jitter).
_FX_PALETTE: list[tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (1.0, 0.85, 0.4),
    (1.0, 0.75, 0.2),
    (0.4, 0.95, 1.0),
    (0.7, 1.0, 1.0),
]
_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


# ==========================================================================
# Shared helpers (source: scripts/make_composites.py — same contracts)
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


def _load_rgb_capped(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """Loads RGB; downscales with LANCZOS if the long side exceeds `max_side` (memory)."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS
        )
    return np.asarray(im, dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """target_size = (w, h); the alpha is rescaled if the sizes do not match
    (source: make_composites.py::_load_alpha)."""
    im = Image.open(path).convert("L")
    if im.size != target_size:
        im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _list_images(directory: Path | None) -> list[Path]:
    if not directory:
        return []
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def _pairs_from_dir(root: Path) -> list[tuple[Path, Path]]:
    """Matches files under `root/im` + `root/gt` by stem (sorted)."""
    root = Path(root)
    gts = {p.stem: p for p in _list_images(root / "gt")}
    return [(p, gts[p.stem]) for p in _list_images(root / "im") if p.stem in gts]


def _load_manifest_ids(path: Path) -> set[str]:
    """Ids in the output manifest (to avoid duplicating lines on resume).

    benchmark.testset.load_manifest is NOT used: the text/fx categories are
    not in that module's CATEGORIES set (see the module docstring contracts)."""
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


# ==========================================================================
# Background selection (real crop / flat-gradient synthetic)
# ==========================================================================
def _synthetic_bg(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Flat color or two-color linear gradient (the text category's 20% branch)."""
    w, h = size
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    if rng.uniform() < 0.5:
        return np.ascontiguousarray(np.broadcast_to(c1.round(), (h, w, 3))).astype(np.uint8)
    c2 = rng.integers(0, 256, 3).astype(np.float32)
    horizontal = rng.uniform() < 0.5
    n = w if horizontal else h
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    grad = c1[None, :] * (1 - t) + c2[None, :] * t  # (n, 3)
    arr = grad[None, :, :] if horizontal else grad[:, None, :]
    return np.ascontiguousarray(np.broadcast_to(arr, (h, w, 3)).round()).astype(np.uint8)


def _bg_crop(rng: np.random.Generator, bg_paths: list[Path], size: tuple[int, int]) -> np.ndarray:
    """A randomly cropped (w, h) patch of a random real background from the pool."""
    arr = _load_rgb_capped(bg_paths[int(rng.integers(0, len(bg_paths)))])
    bh, bw = arr.shape[:2]
    w, h = size
    scale = max(w / bw, h / bh) * float(rng.uniform(1.0, 1.4))  # cover + slight zoom
    nw, nh = max(w, round(bw * scale)), max(h, round(bh * scale))
    if (nw, nh) != (bw, bh):
        arr = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    x0 = int(rng.integers(0, nw - w + 1))
    y0 = int(rng.integers(0, nh - h + 1))
    return np.ascontiguousarray(arr[y0 : y0 + h, x0 : x0 + w])


def _pick_bg(
    rng: np.random.Generator,
    bg_paths: list[Path],
    size: tuple[int, int],
    flat_prob: float = FLAT_BG_PROB,
) -> np.ndarray:
    if not bg_paths or rng.uniform() < flat_prob:
        return _synthetic_bg(rng, size)
    return _bg_crop(rng, bg_paths, size)


# ==========================================================================
# text category — synthetic text/logo rendering
# ==========================================================================
def _load_font_paths(font_dir: Path | None) -> list[Path]:
    if not font_dir:
        return []
    font_dir = Path(font_dir)
    if not font_dir.is_dir():
        return []
    return sorted(p for p in font_dir.rglob("*") if p.suffix.lower() in FONT_EXTS)


def _renders_latin(font: ImageFont.ImageFont) -> bool:
    """Can the font actually draw Latin glyphs? Fonts without Latin coverage
    (e.g. symbol/CJK fonts in macOS Supplemental) render every letter as the
    SAME "tofu" box — if the "I" and "W" masks are identical, the font is
    unusable."""
    try:
        m_i, m_w = font.getmask("I"), font.getmask("W")
    except OSError:
        return False
    return m_i.size != m_w.size or bytes(m_i) != bytes(m_w)


def _get_font(font_paths: list[Path], rng: np.random.Generator, size: int) -> ImageFont.ImageFont:
    for _ in range(8):  # retry if a font without Latin support is picked
        if not font_paths:
            break
        path = font_paths[int(rng.integers(0, len(font_paths)))]
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            continue  # corrupt/unreadable font file -> try again
        if _renders_latin(font):
            return font
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1: load_default() takes no size parameter
        return ImageFont.load_default()


def _rand_text(rng: np.random.Generator) -> str:
    """Short brand-like string of 1-3 words mixing letters and digits."""
    words = []
    for _ in range(int(rng.integers(1, 4))):
        n = int(rng.integers(3, 9))
        words.append("".join(_CHARS[int(rng.integers(0, len(_CHARS)))] for _ in range(n)))
    return " ".join(words)


def _bright_color(rng: np.random.Generator) -> tuple[int, int, int]:
    c = rng.integers(96, 256, 3)
    c[int(rng.integers(0, 3))] = int(rng.integers(192, 256))
    return (int(c[0]), int(c[1]), int(c[2]))


def _rand_color(rng: np.random.Generator) -> tuple[int, int, int]:
    c = rng.integers(0, 256, 3)
    return (int(c[0]), int(c[1]), int(c[2]))


def _draw_text_rgba(
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int] | None,
    pad: int,
) -> Image.Image:
    """Draws the text onto its own tightly-framed RGBA layer (pad: effect overflow margin)."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return img


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int = 5) -> list[tuple[float, float]]:
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        ang = -math.pi / 2 + k * math.pi / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _add_badge(base: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simple vector badge behind the text (circle / rounded-rect / star) —
    a logo look. The badge fill may be semi-transparent (alpha 140-255)."""
    gw, gh = base.size
    m = max(4, int(0.2 * max(gw, gh)))
    group = Image.new("RGBA", (gw + 2 * m, gh + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(group)
    color = _bright_color(rng) + (int(rng.integers(140, 256)),)
    shape = int(rng.integers(0, 3))
    gw2, gh2 = group.size
    if shape == 0:
        d.ellipse([m // 2, m // 2, gw2 - m // 2, gh2 - m // 2], fill=color)
    elif shape == 1:
        d.rounded_rectangle(
            [m // 2, m // 2, gw2 - m // 2, gh2 - m // 2],
            radius=max(2, min(gw2, gh2) // 6),
            fill=color,
        )
    else:
        r_out = min(gw2, gh2) / 2 - 1
        d.polygon(_star_points(gw2 / 2, gh2 / 2, r_out, r_out * 0.45), fill=color)
    group.alpha_composite(base, (m, m))
    return group


def _text_group(rng: np.random.Generator, canvas_min: int, font_paths: list[Path]) -> Image.Image:
    """Unrotated RGBA group layer with the text (+ optional stroke and badge)."""
    text = _rand_text(rng)
    font_size = max(8, int(canvas_min * float(rng.uniform(0.05, 0.40))))
    font = _get_font(font_paths, rng, font_size)
    fill = (_bright_color(rng) if rng.uniform() < 0.7 else _rand_color(rng)) + (255,)
    stroke_width, stroke_fill = 0, None
    if rng.uniform() < 0.5:
        stroke_width = max(1, font_size // 12)
        stroke_fill = _rand_color(rng)
    pad = max(6, font_size // 2)  # shadow/glow overflow margin
    base = _draw_text_rgba(text, font, fill, stroke_width, stroke_fill, pad)
    if rng.uniform() < BADGE_PROB:
        base = _add_badge(base, rng)
    return base


def _decorate(group: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Adds a drop shadow (offset + gaussian blur) and/or glow.

    MID-ALPHA GUARANTEE: if neither was randomly picked, glow is forced — the
    gt of every text sample contains intermediate alpha values outside 0/255
    (glow and blurred shadow produce semi-transparent alpha; see the module
    docstring)."""
    a = group.getchannel("A")
    out = Image.new("RGBA", group.size, (0, 0, 0, 0))
    use_shadow = rng.uniform() < 0.45
    use_glow = rng.uniform() < 0.6
    if not use_shadow and not use_glow:
        use_glow = True

    if use_shadow:
        off = max(1, int(0.02 * max(group.size) * float(rng.uniform(1.0, 3.0))))
        dx = off if rng.uniform() < 0.5 else -off
        dy = max(1, int(off * float(rng.uniform(0.5, 1.5))))
        sa = ImageChops.offset(a, dx, dy).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(0.8, 3.0)))
        )
        opacity = float(rng.uniform(0.4, 0.8))
        sa = Image.fromarray((np.asarray(sa, dtype=np.float32) * opacity).astype(np.uint8), "L")
        shadow_color = rng.integers(0, 64, 3)
        shadow = Image.new("RGBA", group.size, (int(shadow_color[0]), int(shadow_color[1]), int(shadow_color[2]), 0))
        shadow.putalpha(sa)
        out.alpha_composite(shadow)

    if use_glow:
        radius = max(1.5, 0.04 * max(group.size) * float(rng.uniform(0.5, 1.5)))
        ga = a.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius))
        strength = float(rng.uniform(0.35, 0.8))  # SEMI-TRANSPARENT — not binarized
        ga = Image.fromarray((np.asarray(ga, dtype=np.float32) * strength).astype(np.uint8), "L")
        glow_color = (255, 255, 255) if rng.uniform() < 0.5 else _bright_color(rng)
        glow = Image.new("RGBA", group.size, glow_color + (0,))
        glow.putalpha(ga)
        out.alpha_composite(glow)

    out.alpha_composite(group)
    return out


def _render_text_sample(
    rng: np.random.Generator,
    size: tuple[int, int],
    bg_paths: list[Path],
    font_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Single text sample: (composite RGB uint8, alpha float32 [0,1])."""
    w, h = size
    group = _decorate(_text_group(rng, min(w, h), font_paths), rng)
    group = group.rotate(float(rng.uniform(-30, 30)), expand=True, resample=Image.BICUBIC)

    # fit to canvas (a large font size + rotation may exceed the canvas)
    max_w, max_h = int(0.95 * w), int(0.95 * h)
    if group.width > max_w or group.height > max_h:
        s = min(max_w / group.width, max_h / group.height)
        group = group.resize(
            (max(1, int(group.width * s)), max(1, int(group.height * s))), Image.LANCZOS
        )

    x0 = int(rng.integers(0, w - group.width + 1))
    y0 = int(rng.integers(0, h - group.height + 1))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(group, (x0, y0))

    fg = np.asarray(canvas, dtype=np.float32)
    alpha = fg[..., 3] / 255.0
    bg = _pick_bg(rng, bg_paths, (w, h)).astype(np.float32)
    rgb = fg[..., :3] * alpha[..., None] + bg * (1 - alpha[..., None])
    return rgb.round().clip(0, 255).astype(np.uint8), alpha.astype(np.float32)


def gen_text(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    bg_paths: list[Path],
    font_paths: list[Path],
    seed: int,
    existing_ids: set[str],
    canvas_range: tuple[int, int] = (448, 768),
) -> tuple[list[dict], int, int]:
    """Returns (manifest rows, number of pairs generated, number of pairs skipped)."""
    new_rows: list[dict] = []
    generated = skipped = 0
    lo, hi = canvas_range
    for i in range(count):
        stem = f"text_{i:05d}_c00"
        img_path = out_im_dir / f"{stem}.jpg"
        gt_path = out_gt_dir / f"{stem}.png"
        row = {"id": stem, "category": "text"}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if stem not in existing_ids:
                new_rows.append(row)  # file exists, manifest line missing -> line only
            continue
        rng = _item_rng(seed, stem)
        w = int(rng.integers(lo, hi + 1))
        h = int(rng.integers(lo, hi + 1))
        rgb, alpha = _render_text_sample(rng, (w, h), bg_paths, font_paths)
        _save_pair(rgb, alpha, img_path, gt_path)
        new_rows.append(row)
        generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# fx category — procedural VFX glow around a foreground
# ==========================================================================
def _fx_color(rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(_FX_PALETTE[int(rng.integers(0, len(_FX_PALETTE)))], dtype=np.float32)
    return np.clip(base + rng.uniform(-0.08, 0.08, 3).astype(np.float32), 0.0, 1.0)


def _add_spot(
    acc: np.ndarray,
    rng: np.random.Generator,
    region: tuple[float, float, float, float] | None = None,
) -> None:
    """Adds a gaussian sparkle or 4-armed star onto acc (H, W float) (via max).

    If `region` (x0, y0, x1, y1) is given, the sparkle is placed inside that
    rectangle — v5: sparkles are concentrated in the object's expanded bbox
    (the "vfx around the object" scenario; free-floating spots in far corners
    of the background were feeding the ghosting signal)."""
    h, w = acc.shape
    sigma = max(0.6, min(h, w) * float(rng.uniform(0.004, 0.02)))
    if region is not None:
        rx0, ry0, rx1, ry1 = region
        cx, cy = float(rng.uniform(rx0, rx1)), float(rng.uniform(ry0, ry1))
    else:
        cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    peak = float(rng.uniform(0.3, FX_ALPHA_MAX))  # semi-transparent mid values
    star = rng.uniform() < 0.5
    r = int(6 * sigma) + 1
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
    if star:
        s_long, s_short = 4.0 * sigma, 0.5 * sigma
        k = np.exp(-(xx**2 / (2 * s_long**2) + yy**2 / (2 * s_short**2)))
        k = np.maximum(k, np.exp(-(xx**2 / (2 * s_short**2) + yy**2 / (2 * s_long**2))))
        k = np.maximum(k, np.exp(-(xx**2 + yy**2) / (2 * sigma**2)))
    else:
        k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    np.maximum(acc[y0:y1, x0:x1], peak * k, out=acc[y0:y1, x0:x1])


def _streaks(
    rng: np.random.Generator,
    h: int,
    w: int,
    region: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Thin, blurred lens-flare-like light streaks (H, W float [0,1])."""
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    diag = math.hypot(w, h)
    # v5 FIX (2026-07-13): in v4 the streaks were 20-70% of the diagonal —
    # low-alpha lines crossing the whole image fed ghosting. They are now
    # short (5-18%), fewer in number, and concentrated in the object region.
    for _ in range(int(rng.integers(1, 3))):
        if region is not None:
            rx0, ry0, rx1, ry1 = region
            cx, cy = float(rng.uniform(rx0, rx1)), float(rng.uniform(ry0, ry1))
        else:
            cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        ang = float(rng.uniform(0, math.pi))
        half = diag * float(rng.uniform(0.05, 0.18)) / 2
        dx, dy = math.cos(ang) * half, math.sin(ang) * half
        val = int(255 * float(rng.uniform(0.15, 0.5)))
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=val, width=int(rng.integers(1, 3)))
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.8, 2.5))))
    return np.asarray(layer, dtype=np.float32) / 255.0


def _render_fx_sample(
    rng: np.random.Generator,
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    bg_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Single fx sample: (composite RGB uint8, alpha float32 [0,1]).

    alpha = max(fg_alpha, fx_alpha); RGB = the fg's composite onto a real bg
    with the sparkle energy screen-blended on top (see module docstring)."""
    h, w = fg_alpha.shape
    elements: list[tuple[np.ndarray, np.ndarray]] = []  # (alpha map, color)

    # 1) glow halo — on EVERY sample (the fx leg of the mid-alpha guarantee):
    # an outward MaxFilter+GaussianBlur copy of the object alpha. v5 FIX
    # (2026-07-13): in v4 the radius was 2-8% of the image (a giant 96px halo
    # at 1200px) — the model learned that "wide soft-alpha layers are normal"
    # and started GHOSTING solid objects (benchmark: complex mid-alpha
    # 4.5%->5.9%, ping-pong table 35%). The halo is now a narrow band AT the
    # object boundary: radius capped in absolute pixels + lower peak alpha.
    pil_a = Image.fromarray((fg_alpha * 255).astype(np.uint8), mode="L")
    ksz = 3 + 2 * int(rng.integers(0, 2))  # 3 / 5
    radius = min(10.0, max(1.5, min(h, w) * float(rng.uniform(0.004, 0.012))))
    halo = pil_a.filter(ImageFilter.MaxFilter(ksz)).filter(ImageFilter.GaussianBlur(radius))
    # only the part spilling OUTSIDE the object counts as halo; the interior is already alpha=1
    halo_a = (np.asarray(halo, dtype=np.float32) / 255.0) * float(rng.uniform(0.15, 0.35))
    elements.append((halo_a, _fx_color(rng)))

    # 2) particle sparkles (gaussian kernel / 4-armed star) — v5:
    # concentrated in the object's bbox expanded by 40%.
    ys, xs = np.nonzero(fg_alpha > 0.1)
    if len(xs):
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        mx, my = 0.4 * (bx1 - bx0 + 1), 0.4 * (by1 - by0 + 1)
        region = (max(0.0, bx0 - mx), max(0.0, by0 - my),
                  min(float(w), bx1 + mx), min(float(h), by1 + my))
    else:
        region = None
    spots = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(5, 26))):
        _add_spot(spots, rng, region=region)
    elements.append((spots, _fx_color(rng)))

    # 3) light streaks
    if rng.uniform() < 0.5:
        elements.append((_streaks(rng, h, w, region=region), _fx_color(rng)))

    fx_alpha = np.zeros((h, w), dtype=np.float32)
    fx_energy = np.zeros((h, w, 3), dtype=np.float32)
    for a_map, color in elements:
        fx_alpha = 1 - (1 - fx_alpha) * (1 - a_map)  # alpha union
        fx_energy = 1 - (1 - fx_energy) * (1 - a_map[..., None] * color[None, None, :])
    fx_alpha = fx_alpha.clip(0.0, FX_ALPHA_MAX)  # transparency ceiling (0.15-0.9 band)

    bg = _pick_bg(rng, bg_paths, (w, h), flat_prob=0.0).astype(np.float32) / 255.0
    base = fg_rgb.astype(np.float32) / 255.0 * fg_alpha[..., None] + bg * (1 - fg_alpha[..., None])
    out = 1 - (1 - base) * (1 - fx_energy)  # screen: sparkles appear additive
    out_alpha = np.maximum(fg_alpha, fx_alpha)
    return (out * 255).round().clip(0, 255).astype(np.uint8), out_alpha.astype(np.float32)


def gen_fx(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    pairs: list[tuple[Path, Path]],
    bg_paths: list[Path],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """Distributes copies evenly across the source fg pairs: index = source
    ordinal, copy = that source's copy ordinal. Returns (rows, generated, skipped)."""
    if count > 0 and not pairs:
        raise SystemExit("no source im/gt pairs found for fx (--fg-dirs roots must contain im/ + gt/)")
    base_copies, rem = divmod(count, len(pairs))
    assert base_copies + (1 if rem else 0) <= 100, (
        f"fx copy count cannot exceed 100 per source (count={count}, sources={len(pairs)}): "
        f"the 2-digit `_c<NN>` naming would overflow."
    )
    new_rows: list[dict] = []
    generated = skipped = 0
    for idx, (im_path, gt_src) in enumerate(pairs):
        n_copies = base_copies + (1 if idx < rem else 0)
        if n_copies == 0:
            continue
        pending: list[str] = []
        for ci in range(n_copies):
            stem = f"fx_{idx:05d}_c{ci:02d}"
            if (out_im_dir / f"{stem}.jpg").exists() and (out_gt_dir / f"{stem}.png").exists():
                skipped += 1
                if stem not in existing_ids:
                    new_rows.append({"id": stem, "category": "fx"})
                continue
            pending.append(stem)
        if not pending:
            continue
        fg_rgb = _load_rgb_capped(im_path)
        fg_alpha = _load_alpha(gt_src, (fg_rgb.shape[1], fg_rgb.shape[0]))
        for stem in pending:
            rng = _item_rng(seed, stem)
            rgb, alpha = _render_fx_sample(rng, fg_rgb, fg_alpha, bg_paths)
            _save_pair(rgb, alpha, out_im_dir / f"{stem}.jpg", out_gt_dir / f"{stem}.png")
            new_rows.append({"id": stem, "category": "fx"})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# illustration category — composites + originals from ready-made ToonOut pairs
# ==========================================================================
def gen_illustration(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    pairs: list[tuple[Path, Path]],
    bg_paths: list[Path],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """3 copies per pair: c00/c01 compose+augment (bgr.compositing —
    the make_composites pattern), c02 original image (NO compose, augment
    only; make_composites `_o00` logic). Returns (rows, generated, skipped)."""
    if count > 0 and not pairs:
        raise SystemExit("no ToonOut im/gt pairs found for illustration (--toonout-dir/im + /gt)")
    if count > 0 and not bg_paths:
        raise SystemExit("illustration compositing requires a background pool (--bg-dir)")
    n_pairs = min(len(pairs), math.ceil(count / ILLUSTRATION_COPIES))
    new_rows: list[dict] = []
    generated = skipped = emitted = 0
    for idx in range(n_pairs):
        im_path, gt_src = pairs[idx]
        stems: list[tuple[str, int]] = []
        for ci in range(ILLUSTRATION_COPIES):
            if emitted >= count:
                break
            stems.append((f"illustration_{idx:05d}_c{ci:02d}", ci))
            emitted += 1
        pending: list[tuple[str, int]] = []
        for stem, ci in stems:
            if (out_im_dir / f"{stem}.jpg").exists() and (out_gt_dir / f"{stem}.png").exists():
                skipped += 1
                if stem not in existing_ids:
                    new_rows.append({"id": stem, "category": "illustration"})
                continue
            pending.append((stem, ci))
        if not pending:
            continue
        fg_rgb = _load_rgb_capped(im_path)
        alpha = _load_alpha(gt_src, (fg_rgb.shape[1], fg_rgb.shape[0]))
        for stem, ci in pending:
            rng = _item_rng(seed, stem)
            if ci < ILLUSTRATION_COPIES - 1:  # c00/c01: composite onto a real bg
                bg_rgb = _load_rgb_capped(bg_paths[int(rng.integers(0, len(bg_paths)))])
                out_rgb, out_alpha = compose(fg_rgb, alpha, bg_rgb, rng)
            else:  # c02: original image (whatever the raw is) — augment only
                out_rgb, out_alpha = fg_rgb, alpha
            out_rgb, out_alpha = augment(out_rgb, out_alpha, rng)
            _save_pair(out_rgb, out_alpha, out_im_dir / f"{stem}.jpg", out_gt_dir / f"{stem}.png")
            new_rows.append({"id": stem, "category": "illustration"})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# Orchestration
# ==========================================================================
def run(
    out_dir: Path,
    bg_dir: Path | None = None,
    fg_dirs: list[Path] | None = None,
    toonout_dir: Path | None = None,
    font_dir: Path | None = None,
    seed: int = 42,
    counts: dict[str, int] | None = None,
    out_manifest: Path | None = None,
    text_canvas: tuple[int, int] = (448, 768),
) -> dict[str, int]:
    """Runs the generators for the 3 categories; returns category -> number of
    newly generated pairs (only entries >0 — same pattern as make_composites.run()).

    A category missing from `counts` or set to 0 is skipped entirely (its
    inputs are not even scanned). `text_canvas`: the canvas side range for the
    text category (parametric so tests can run fast with small values)."""
    out_dir = Path(out_dir)
    counts = dict(DEFAULT_COUNTS) if counts is None else counts
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    bg_paths = _list_images(Path(bg_dir)) if bg_dir else []
    font_paths = _load_font_paths(Path(font_dir) if font_dir else None)

    all_rows: list[dict] = []
    result: dict[str, int] = {}
    total_skipped = 0

    if counts.get("text", 0) > 0:
        rows, generated, skipped = gen_text(
            counts["text"], out_im_dir, out_gt_dir, bg_paths, font_paths, seed,
            existing_ids, canvas_range=text_canvas,
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["text"] = generated

    if counts.get("fx", 0) > 0:
        pairs: list[tuple[Path, Path]] = []
        for d in fg_dirs or []:
            pairs += _pairs_from_dir(Path(d))
        rows, generated, skipped = gen_fx(
            counts["fx"], out_im_dir, out_gt_dir, pairs, bg_paths, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["fx"] = generated

    if counts.get("illustration", 0) > 0:
        pairs = _pairs_from_dir(Path(toonout_dir)) if toonout_dir else []
        rows, generated, skipped = gen_illustration(
            counts["illustration"], out_im_dir, out_gt_dir, pairs, bg_paths, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["illustration"] = generated

    # only new ids go to the manifest (including an in-run safety dedup)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in all_rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{sum(result.values())} new pairs written, {total_skipped} already existed (skipped)")
    for category, n in sorted(result.items()):
        print(f"{category}: {n}")
    return result


def _parse_counts(spec: str) -> dict[str, int]:
    """'text=4000,fx=3500' -> dict; missing categories default to 0 (skipped)."""
    counts = {k: 0 for k in DEFAULT_COUNTS}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        if key not in DEFAULT_COUNTS or not value:
            raise SystemExit(
                f"invalid --counts segment: {part!r} (expected: {'|'.join(DEFAULT_COUNTS)}=N)"
            )
        counts[key] = int(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="output root (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--bg-dir", default=None, help="real background pool (BG-20k)")
    parser.add_argument(
        "--fg-dirs", nargs="*", default=[],
        help="fx source roots; each root must contain im/ + gt/ subdirectories (matched by stem)",
    )
    parser.add_argument("--toonout-dir", default=None, help="ToonOut root (im/ + gt/)")
    parser.add_argument("--font-dir", default=None, help=".ttf/.otf/.ttc font pool (PIL default if absent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts", default="text=4000,fx=3500,illustration=3600")
    parser.add_argument("--out-manifest", default=None, help="default: <out-dir>/manifest.jsonl")
    args = parser.parse_args()
    run(
        Path(args.out_dir),
        bg_dir=Path(args.bg_dir) if args.bg_dir else None,
        fg_dirs=[Path(d) for d in args.fg_dirs],
        toonout_dir=Path(args.toonout_dir) if args.toonout_dir else None,
        font_dir=Path(args.font_dir) if args.font_dir else None,
        seed=args.seed,
        counts=_parse_counts(args.counts),
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
    )


if __name__ == "__main__":
    main()
