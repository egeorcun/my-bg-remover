import numpy as np
from unittest.mock import patch
from PIL import Image

from bgr.cli import main
from bgr.pipeline import PipelineSegmenter


class FakeSeg:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def predict_alpha(self, image):
        self.calls += 1
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def test_remove_writes_rgba(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (16, 16), (10, 120, 200)).save(src)
    dst = tmp_path / "out.png"
    with patch("bgr.cli.get_segmenter", return_value=FakeSeg()):
        main(["remove", str(src), "-o", str(dst), "--no-decontaminate"])
    out = Image.open(dst)
    assert out.mode == "RGBA" and out.size == (16, 16)


def test_remove_does_not_double_wrap_pipeline(tmp_path):
    """get_segmenter zaten refine'lı bir PipelineSegmenter döndürdüğünde,
    --refine bayrağı CLI'da ikinci bir sarmalama yaratmamalı."""
    src = tmp_path / "in.jpg"
    Image.new("RGB", (16, 16), (10, 120, 200)).save(src)
    dst = tmp_path / "out.png"
    fake = FakeSeg()
    pipeline = PipelineSegmenter(fake, refine=True)
    with patch("bgr.cli.get_segmenter", return_value=pipeline):
        main(["remove", str(src), "-o", str(dst), "--refine", "--no-decontaminate"])
    out = Image.open(dst)
    assert out.mode == "RGBA" and out.size == (16, 16)
    # tüm alpha 1.0 (tam emin) olduğu için refiner sıfır patch koşturur;
    # çift sarmalanmışsa taban modele birden fazla çağrı gitmez ama
    # ikinci bir refine katmanı ekstra maliyete/side effect'e yol açardı.
    assert fake.calls == 1
