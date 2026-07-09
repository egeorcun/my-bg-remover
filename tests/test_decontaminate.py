import numpy as np
import pytest
from PIL import Image

from bgr.decontaminate import decontaminate


@pytest.fixture
def red_on_green():
    """Kırmızı kare, yeşil zemin, kenarda 3px yumuşak (karışmış) geçiş."""
    w = h = 64
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :] = (0.0, 0.8, 0.0)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    from scipy import ndimage
    alpha = ndimage.gaussian_filter(alpha, 1.5).clip(0, 1).astype(np.float32)
    comp = alpha[..., None] * np.array([0.9, 0.1, 0.1]) + (1 - alpha[..., None]) * img
    pil = Image.fromarray((comp * 255).astype(np.uint8))
    return pil, alpha


def test_returns_rgba_same_size(red_on_green):
    pil, alpha = red_on_green
    out = decontaminate(pil, alpha)
    assert out.mode == "RGBA"
    assert out.size == pil.size


def test_edge_pixels_lose_green_spill(red_on_green):
    pil, alpha = red_on_green
    out = np.asarray(decontaminate(pil, alpha), dtype=np.float64) / 255.0
    band = (alpha > 0.2) & (alpha < 0.8)
    naive_rgb = np.asarray(pil, dtype=np.float64) / 255.0
    # kenar bandında yeşil kanal, naive kompozite göre belirgin azalmalı
    assert out[..., 1][band].mean() < naive_rgb[..., 1][band].mean() - 0.05
    # opak iç bölge değişmemeli (kırmızı kalmalı)
    core = alpha > 0.99
    assert abs(out[..., 0][core].mean() - naive_rgb[..., 0][core].mean()) < 0.05


def test_shape_mismatch_raises(red_on_green):
    pil, _ = red_on_green
    with pytest.raises(ValueError):
        decontaminate(pil, np.zeros((8, 8), dtype=np.float32))
