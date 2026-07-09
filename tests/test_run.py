import json
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.run import run_benchmark


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _make_testset(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(img)
    gt = tmp_path / "a.png"
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(gt)
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "general", "gt_alpha": str(gt),
    }) + "\n")
    return manifest


def test_run_benchmark_outputs_and_metrics(tmp_path):
    manifest = _make_testset(tmp_path)
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        result = run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert (tmp_path / "out/fake/a.png").exists()
    assert result["per_image"]["fake"]["a"]["sad"] == 0.0  # tam isabet
    assert result["overall"]["fake"]["mae"] == 0.0
    assert (tmp_path / "out/metrics.json").exists()


class SpySeg(FakeSeg):
    def __init__(self):
        self.calls = 0

    def predict_alpha(self, image):
        self.calls += 1
        return super().predict_alpha(image)


def test_resume_skips_existing_outputs(tmp_path):
    manifest = _make_testset(tmp_path)
    spy = SpySeg()
    with patch("benchmark.run.get_segmenter", return_value=spy):
        run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
        first = (tmp_path / "out/metrics.json").read_text()
        run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert spy.calls == 1  # ikinci koşu var olan PNG'yi yeniden üretmez
    assert (tmp_path / "out/metrics.json").read_text() == first


def test_metrics_json_merges_across_invocations(tmp_path):
    manifest = _make_testset(tmp_path)
    manifest_path = str(manifest)
    out_dir = str(tmp_path / "out")

    class OtherSeg(FakeSeg):
        name = "other"

    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        run_benchmark(["fake"], manifest_path, out_dir)
    with patch("benchmark.run.get_segmenter", return_value=OtherSeg()):
        run_benchmark(["other"], manifest_path, out_dir)

    metrics = json.loads((tmp_path / "out/metrics.json").read_text())
    assert set(metrics["overall"]) == {"fake", "other"}
    assert set(metrics["per_image"]) == {"fake", "other"}
    assert set(metrics["per_category"]) == {"fake", "other"}


def test_gtless_row_gets_alpha_but_no_metrics(tmp_path):
    manifest = _make_testset(tmp_path)
    img_b = tmp_path / "b.jpg"
    Image.new("RGB", (8, 8), (20, 20, 20)).save(img_b)
    with manifest.open("a") as f:
        f.write(json.dumps({
            "id": "b", "image": str(img_b), "category": "general", "gt_alpha": None,
        }) + "\n")
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        result = run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert (tmp_path / "out/fake/a.png").exists()
    assert (tmp_path / "out/fake/b.png").exists()
    assert set(result["per_image"]["fake"]) == {"a"}  # GT'siz satıra metrik yok
