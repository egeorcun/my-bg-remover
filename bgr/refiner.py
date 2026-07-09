"""CGM tarzı kenar rafinesi: modelin emin olamadığı bölgeleri kırpıp
aynı modele yüksek efektif çözünürlükte yeniden sorar, sonucu yalnız
belirsiz bantta feather'lı harmanlar.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from bgr.segmenter import Segmenter


def _regions(band: np.ndarray, min_region: int, max_patches: int) -> list[tuple[int, int, int, int]]:
    labels, num = ndimage.label(ndimage.binary_dilation(band, iterations=4))
    if num == 0:
        return []
    sizes = ndimage.sum(band, labels, range(1, num + 1))
    order = np.argsort(sizes)[::-1]
    boxes = ndimage.find_objects(labels)
    out = []
    for i in order[:max_patches]:
        if sizes[i] < min_region:
            break
        sl = boxes[i]
        out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


def refine_alpha(
    segmenter: Segmenter,
    image: Image.Image,
    alpha: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    min_region: int = 256,
    context: float = 0.35,
    max_patches: int = 6,
) -> np.ndarray:
    h, w = alpha.shape
    band = (alpha > low) & (alpha < high)
    out = alpha.copy()
    for y0, y1, x0, x1 in _regions(band, min_region, max_patches):
        cy, cx = int((y1 - y0) * context), int((x1 - x0) * context)
        yy0, yy1 = max(0, y0 - cy), min(h, y1 + cy)
        xx0, xx1 = max(0, x0 - cx), min(w, x1 + cx)
        crop = image.convert("RGB").crop((xx0, yy0, xx1, yy1))
        refined = segmenter.predict_alpha(crop)
        # feather: bant maskesini yumuşat, yalnız bant içinde harmanla
        local_band = band[yy0:yy1, xx0:xx1].astype(np.float32)
        weight = ndimage.gaussian_filter(local_band, 2).clip(0, 1)
        weight[local_band == 0] = 0.0
        region = out[yy0:yy1, xx0:xx1]
        out[yy0:yy1, xx0:xx1] = weight * refined + (1 - weight) * region
    return out.clip(0, 1).astype(np.float32)
