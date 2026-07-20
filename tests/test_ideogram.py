from unittest.mock import patch

from PIL import Image

from benchmark.ideogram import fetch_reference


def test_skips_existing_output(tmp_path):
    out = tmp_path / "x.png"
    Image.new("RGBA", (4, 4)).save(out)
    with patch("benchmark.ideogram.fal_client") as m:
        fetch_reference("not-real.jpg", str(out))
        m.subscribe.assert_not_called()


def test_calls_fal_and_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_KEY", "dummy")
    src = tmp_path / "in.jpg"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(src)
    out = tmp_path / "out.png"
    fake_png = tmp_path / "fake_result.png"
    Image.new("RGBA", (4, 4), (0, 255, 0, 128)).save(fake_png)
    with (
        patch("benchmark.ideogram.fal_client") as m,
        patch("benchmark.ideogram._download") as dl,
    ):
        m.upload_file.return_value = "https://fal.example/in.jpg"
        m.subscribe.return_value = {"image": {"url": "https://fal.example/out.png"}}
        dl.side_effect = lambda url, path: fake_png.rename(path)
        fetch_reference(str(src), str(out))
    assert out.exists()
