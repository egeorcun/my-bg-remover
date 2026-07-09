import numpy as np
from PIL import Image

from bgr.refiner import refine_alpha


class SharpFakeSeg:
    """Kırpıntıda 'keskin' alpha döndürür: sol yarı 1, sağ yarı 0."""
    name = "sharp-fake"

    def __init__(self):
        self.calls = []

    def predict_alpha(self, image):
        w, h = image.size
        self.calls.append((w, h))
        a = np.zeros((h, w), dtype=np.float32)
        a[:, : w // 2] = 1.0
        return a


def _blurry_alpha(h=128, w=128):
    a = np.zeros((h, w), dtype=np.float32)
    a[:, : w // 2] = 1.0
    from scipy import ndimage
    return ndimage.gaussian_filter(a, 6).clip(0, 1).astype(np.float32)


def test_confident_alpha_untouched():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (64, 64))
    a = np.ones((64, 64), dtype=np.float32)  # tamamen emin
    out = refine_alpha(seg, img, a)
    assert seg.calls == []  # hiç patch koşmadı
    np.testing.assert_array_equal(out, a)


def test_uncertain_band_gets_sharper():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (128, 128))
    blurry = _blurry_alpha()
    out = refine_alpha(seg, img, blurry)
    assert len(seg.calls) >= 1
    band = (blurry > 0.05) & (blurry < 0.95)
    # rafine sonrası bantta ara-değerli piksel sayısı azalmalı (keskinleşme)
    mid_before = ((blurry > 0.2) & (blurry < 0.8) & band).sum()
    mid_after = ((out > 0.2) & (out < 0.8) & band).sum()
    assert mid_after < mid_before
    # emin bölgeler değişmedi
    np.testing.assert_allclose(out[~band], blurry[~band], atol=1e-6)


def test_contract_preserved():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (96, 80))
    out = refine_alpha(seg, img, _blurry_alpha(80, 96))
    assert out.dtype == np.float32 and out.shape == (80, 96)
    assert out.min() >= 0 and out.max() <= 1
