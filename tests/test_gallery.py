import json

import numpy as np
from PIL import Image

from benchmark.gallery import build_gallery


def test_gallery_contains_rows_and_images(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    (tmp_path / "results/m1").mkdir(parents=True)
    Image.fromarray(np.full((8, 8), 200, np.uint8)).save(tmp_path / "results/m1/a.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "hair", "gt_alpha": None,
    }) + "\n")
    out = tmp_path / "results/gallery.html"
    build_gallery(str(manifest), str(tmp_path / "results"), ["m1"], str(out))
    html = out.read_text()
    assert "hair" in html and 'id="a"' in html
    # the model cell must embed the RGBA composite, not the raw mask
    assert "m1/a" not in html
    assert "m1/composites/a" in html
    composite = tmp_path / "results/m1/composites/a.png"
    assert composite.exists()
    with Image.open(composite) as comp:
        assert comp.mode == "RGBA"


def test_gallery_prefers_rgba_over_composite(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    (tmp_path / "results/m1/rgba").mkdir(parents=True)
    Image.fromarray(np.full((8, 8), 200, np.uint8)).save(tmp_path / "results/m1/a.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "results/m1/rgba/a.png")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "id": "a", "image": str(img), "category": "hair", "gt_alpha": None,
    }) + "\n")
    out = tmp_path / "results/gallery.html"
    build_gallery(str(manifest), str(tmp_path / "results"), ["m1"], str(out))
    html = out.read_text()
    # if rgba exists it must be embedded, no composite should be produced
    assert "m1/rgba/a" in html
    assert "m1/composites/a" not in html
    assert not (tmp_path / "results/m1/composites").exists()
