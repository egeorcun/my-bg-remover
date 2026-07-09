"""Foreground/arka plan compositing ve augmentasyon.

Sözleşme: RGB `np.uint8 (H, W, 3)` [0, 255]; alpha `np.float32 (H, W)` [0, 1]
(bkz. `bgr/segmenter.py`). Tüm rastgelelik çağırana ait `np.random.Generator`
üzerinden akar — kütüphane içinde global seed (random.seed/np.random.seed)
KULLANILMAZ, aynı seed'le her zaman aynı çıktı üretilir (determinism).
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _check_shapes(rgb: np.ndarray, alpha: np.ndarray) -> None:
    if rgb.shape[:2] != alpha.shape:
        raise ValueError(f"rgb {rgb.shape[:2]} != alpha {alpha.shape}")


def _resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h)."""
    return np.asarray(Image.fromarray(rgb, mode="RGB").resize(size, Image.BILINEAR), dtype=np.uint8)


def _resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h). PIL 'F' modu 32-bit float tek kanalı destekler."""
    out = Image.fromarray(alpha.astype(np.float32), mode="F").resize(size, Image.BILINEAR)
    return np.asarray(out, dtype=np.float32).clip(0, 1)


def compose(
    fg_rgb: np.ndarray,
    alpha: np.ndarray,
    bg_rgb: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.4, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """fg'yi rastgele ölçek/konumla bg'ye alpha-blend eder.

    bg, fg'den küçükse (herhangi bir boyutta) önce büyütülür (canvas her zaman
    fg'yi tam olarak içerecek kadar büyük olur). Döndürülen alpha, canvas
    üzerinde yerleştirilen (ölçeklenmiş) fg alpha'sı dışında her yerde 0'dır.
    """
    _check_shapes(fg_rgb, alpha)
    fh, fw = fg_rgb.shape[:2]
    bh, bw = bg_rgb.shape[:2]

    grow = max(fw / bw, fh / bh, 1.0)
    if grow > 1.0:
        import math

        bw, bh = math.ceil(bw * grow), math.ceil(bh * grow)
        bg_rgb = _resize_rgb(bg_rgb, (bw, bh))

    canvas = bg_rgb.astype(np.float32).copy()

    lo, hi = scale_range
    hi = min(hi, bw / fw, bh / fh)
    lo = min(lo, hi)
    scale = float(rng.uniform(lo, hi)) if hi > lo else hi

    new_w = min(bw, max(1, int(round(fw * scale))))
    new_h = min(bh, max(1, int(round(fh * scale))))
    if (new_w, new_h) == (fw, fh):
        fg_resized, alpha_resized = fg_rgb, alpha
    else:
        fg_resized = _resize_rgb(fg_rgb, (new_w, new_h))
        alpha_resized = _resize_alpha(alpha, (new_w, new_h))

    max_x, max_y = bw - new_w, bh - new_h
    x0 = int(rng.integers(0, max_x + 1))
    y0 = int(rng.integers(0, max_y + 1))

    out_alpha = np.zeros((bh, bw), dtype=np.float32)
    out_alpha[y0 : y0 + new_h, x0 : x0 + new_w] = alpha_resized

    a = alpha_resized[..., None]
    region = canvas[y0 : y0 + new_h, x0 : x0 + new_w]
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = a * fg_resized.astype(np.float32) + (1 - a) * region

    return canvas.clip(0, 255).astype(np.uint8), out_alpha


def augment(
    rgb: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
    jpeg_quality_range: tuple[int, int] = (40, 95),
    blur_prob: float = 0.3,
    flip_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Renk/parlaklık jitter, JPEG artifact, hafif blur, yatay flip uygular.

    Alpha'ya YALNIZ geometrik dönüşüm (flip) uygulanır — renk jitter, blur ve
    JPEG artifact yalnız RGB'yi etkiler.
    """
    _check_shapes(rgb, alpha)
    out_rgb = rgb
    out_alpha = alpha

    # 1) renk/parlaklık jitter (yalnız RGB)
    im = Image.fromarray(out_rgb, mode="RGB")
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.8, 1.2)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.8, 1.2)))
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(0.7, 1.3)))

    # 2) hafif blur (yalnız RGB)
    if rng.uniform() < blur_prob:
        radius = float(rng.uniform(0.3, 1.2))
        im = im.filter(ImageFilter.GaussianBlur(radius))

    # 3) JPEG artifact: encode/decode döngüsü (yalnız RGB)
    quality = int(rng.integers(jpeg_quality_range[0], jpeg_quality_range[1] + 1))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    out_rgb = np.asarray(im, dtype=np.uint8)

    # 4) yatay flip (RGB + alpha, tek geometrik dönüşüm)
    if rng.uniform() < flip_prob:
        out_rgb = out_rgb[:, ::-1, :]
        out_alpha = out_alpha[:, ::-1]

    return np.ascontiguousarray(out_rgb), np.ascontiguousarray(out_alpha.clip(0, 1).astype(np.float32))
