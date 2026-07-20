from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from bgr.segmenter import BiRefNetSegmenter, LocalBiRefNetSegmenter, get_device


@pytest.fixture(scope="module")
def toy_image():
    img = Image.new("RGB", (640, 480), (30, 120, 30))
    d = ImageDraw.Draw(img)
    d.ellipse([200, 100, 440, 380], fill=(220, 60, 60))
    return img


def test_get_device_is_mps():
    assert get_device().type == "mps"


@pytest.mark.slow
def test_birefnet_hr_alpha_contract(toy_image):
    seg = BiRefNetSegmenter(
        model_id="ZhengPeng7/BiRefNet_HR", input_size=2048, name="birefnet-hr"
    )
    alpha = seg.predict_alpha(toy_image)
    assert alpha.dtype == np.float32
    assert alpha.shape == (480, 640)
    assert 0.0 <= alpha.min() and alpha.max() <= 1.0
    # the ellipse center must be subject, the corner background
    assert alpha[240, 320] > 0.5
    assert alpha[10, 10] < 0.5


class _FakeArch(torch.nn.Module):
    """Tiny fake architecture standing in for `from_pretrained`, with real
    state_dict/load_state_dict behavior (for hermetic tests)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def _patch_from_pretrained(fake_model):
    return patch(
        "transformers.AutoModelForImageSegmentation.from_pretrained",
        return_value=fake_model,
    )


class TestLocalBiRefNetSegmenter:
    """`torch.load`/`from_pretrained` are mocked: no 2.6GB real checkpoint required."""

    def test_unwraps_model_key_and_loads(self):
        fake_model = _FakeArch()
        payload = {
            "model": fake_model.state_dict(),
            "optimizer": {},
            "lr_scheduler": {},
            "epoch": 1,
        }
        with (
            _patch_from_pretrained(fake_model),
            patch("torch.load", return_value=payload) as mock_load,
        ):
            seg = LocalBiRefNetSegmenter(
                ckpt_path="fake/epoch_1.pth", input_size=1024, name="bgr-v1"
            )
        mock_load.assert_called_once_with(
            "fake/epoch_1.pth", map_location="cpu", weights_only=False
        )
        assert seg.name == "bgr-v1"
        assert seg.input_size == 1024

    def test_strips_orig_mod_prefix(self):
        fake_model = _FakeArch()
        prefixed = {f"_orig_mod.{k}": v for k, v in fake_model.state_dict().items()}
        payload = {"model": prefixed, "optimizer": {}, "lr_scheduler": {}, "epoch": 1}
        with (
            _patch_from_pretrained(fake_model),
            patch("torch.load", return_value=payload),
        ):
            # if the prefix were not stripped, strict load_state_dict would raise RuntimeError
            LocalBiRefNetSegmenter(
                ckpt_path="fake/epoch_1.pth", input_size=1024, name="bgr-v1"
            )

    def test_missing_model_key_raises_keyerror(self):
        fake_model = _FakeArch()
        payload = {"optimizer": {}, "lr_scheduler": {}, "epoch": 1}
        with (
            _patch_from_pretrained(fake_model),
            patch("torch.load", return_value=payload),
            pytest.raises(KeyError, match="model"),
        ):
            LocalBiRefNetSegmenter(
                ckpt_path="fake/epoch_1.pth", input_size=1024, name="bgr-v1"
            )

    def test_strict_mismatch_raises_loudly_with_key_diff(self):
        fake_model = _FakeArch()
        bad_state = dict(fake_model.state_dict())
        del bad_state["linear.bias"]  # missing key -> strict load must fail
        bad_state["extra.unexpected"] = torch.zeros(1)  # unexpected key
        payload = {
            "model": bad_state,
            "optimizer": {},
            "lr_scheduler": {},
            "epoch": 1,
        }
        with (
            _patch_from_pretrained(fake_model),
            patch("torch.load", return_value=payload),
            pytest.raises(RuntimeError) as excinfo,
        ):
            LocalBiRefNetSegmenter(
                ckpt_path="fake/epoch_1.pth", input_size=1024, name="bgr-v1"
            )
        msg = str(excinfo.value)
        assert "linear.bias" in msg
        assert "extra.unexpected" in msg
