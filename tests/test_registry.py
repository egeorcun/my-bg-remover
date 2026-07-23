from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from bgr.registry import MODEL_SPECS, get_segmenter


def test_known_model_names():
    assert set(MODEL_SPECS) == {
        "birefnet-hr", "rmbg-2.0", "bgr-v1", "bgr-v2", "bgr-v3", "bgr-v4",
        "lucida-v5", "lucida-v6", "lucida-v7", "lucida-v8", "lucida-v9", "lucida-v10", "lucida-v11", "lucida-v11probe",
        "inspyrenet", "lucida",
    }


def test_lucida_spec_fields():
    spec = MODEL_SPECS["lucida"]
    assert spec["model_id"] == "egeorcun/lucida"
    assert spec["input_size"] == 1024
    assert "ckpt" not in spec  # downloads from HF, needs no local checkpoint


def test_bgr_v1_spec_fields():
    spec = MODEL_SPECS["bgr-v1"]
    assert spec["ckpt"] == "data/checkpoints/epoch_1.pth"
    assert spec["arch_id"] == "ZhengPeng7/BiRefNet_HR"
    assert spec["input_size"] == 1024


def test_bgr_v1_uses_local_segmenter_with_spec_args():
    with patch("bgr.registry.LocalBiRefNetSegmenter") as m:
        m.return_value.name = "bgr-v1"
        seg = get_segmenter("bgr-v1")
    m.assert_called_once_with(
        ckpt_path="data/checkpoints/epoch_1.pth",
        input_size=1024,
        name="bgr-v1",
        arch_id="ZhengPeng7/BiRefNet_HR",
    )
    assert seg.name == "bgr-v1"


def test_bgr_v1_refine_variant_composes():
    with patch("bgr.registry.LocalBiRefNetSegmenter") as m:
        m.return_value.name = "bgr-v1"
        seg = get_segmenter("bgr-v1+refine")
    assert seg.name == "bgr-v1+refine"


def test_bgr_v1_unknown_variant_raises_before_model_load():
    with patch("bgr.registry.LocalBiRefNetSegmenter") as m:
        with pytest.raises(KeyError):
            get_segmenter("bgr-v1+nope")
    m.assert_not_called()


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        get_segmenter("no-such-model")


def test_unknown_variant_raises_before_model_load():
    """Known base name + unknown variant: fast KeyError before any model weights load."""
    with pytest.raises(KeyError):
        get_segmenter("rmbg-2.0+refime")


@pytest.mark.slow
def test_rmbg2_alpha_contract():
    img = Image.new("RGB", (320, 240), (200, 200, 200))
    ImageDraw.Draw(img).rectangle([100, 60, 220, 180], fill=(20, 20, 160))
    seg = get_segmenter("rmbg-2.0")
    alpha = seg.predict_alpha(img)
    assert alpha.dtype == np.float32
    assert alpha.shape == (240, 320)
    assert float(alpha.max()) <= 1.0 and float(alpha.min()) >= 0.0
