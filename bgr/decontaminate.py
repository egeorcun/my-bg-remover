"""Edge color-spill cleanup (color decontamination).

Edge pixels are mixed with the old background; pymatting's multi-level
foreground estimation solves for the pure subject color at every pixel.
Alpha is left untouched — only the RGB channels are cleaned.
"""
import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml


def decontaminate(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if alpha.shape != rgb.shape[:2]:
        raise ValueError(f"alpha shape {alpha.shape} != image {rgb.shape[:2]}")
    fg = estimate_foreground_ml(rgb, alpha.astype(np.float64))
    out = np.dstack([np.clip(fg, 0, 1), alpha.clip(0, 1)])
    return Image.fromarray(np.round(out * 255).astype(np.uint8), mode="RGBA")
