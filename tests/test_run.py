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
    assert result["per_image"]["fake"]["a"]["sad"] == 0.0  # perfect hit
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
    assert spy.calls == 1  # the second run does not regenerate the existing PNG
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


def test_rgba_flag_produces_decontaminated_output(tmp_path):
    manifest = _make_testset(tmp_path)
    fake_rgba = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()), \
         patch("benchmark.run.decontaminate", return_value=fake_rgba) as mock_decon:
        run_benchmark(["fake"], str(manifest), str(tmp_path / "out"), rgba=True)
    rgba_path = tmp_path / "out/fake/rgba/a.png"
    assert rgba_path.exists()
    with Image.open(rgba_path) as img:
        assert img.mode == "RGBA"
    assert mock_decon.call_count == 1

    # idempotent: on the second run decontaminate is not called again (nor is the segmenter)
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()), \
         patch("benchmark.run.decontaminate", return_value=fake_rgba) as mock_decon2:
        run_benchmark(["fake"], str(manifest), str(tmp_path / "out"), rgba=True)
    assert mock_decon2.call_count == 0


def test_rgba_default_false_produces_no_rgba_dir(tmp_path):
    manifest = _make_testset(tmp_path)
    with patch("benchmark.run.get_segmenter", return_value=FakeSeg()):
        run_benchmark(["fake"], str(manifest), str(tmp_path / "out"))
    assert not (tmp_path / "out/fake/rgba").exists()


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
    assert set(result["per_image"]["fake"]) == {"a"}  # no metrics for the GT-less row
