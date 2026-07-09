import numpy as np
import pytest
from PIL import Image, ImageDraw

from bgr.registry import MODEL_SPECS, get_segmenter


def test_known_model_names():
    assert set(MODEL_SPECS) == {"birefnet-hr", "rmbg-2.0"}


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        get_segmenter("yok-boyle-model")


@pytest.mark.slow
def test_rmbg2_alpha_contract():
    img = Image.new("RGB", (320, 240), (200, 200, 200))
    ImageDraw.Draw(img).rectangle([100, 60, 220, 180], fill=(20, 20, 160))
    seg = get_segmenter("rmbg-2.0")
    alpha = seg.predict_alpha(img)
    assert alpha.dtype == np.float32
    assert alpha.shape == (240, 320)
    assert float(alpha.max()) <= 1.0 and float(alpha.min()) >= 0.0
