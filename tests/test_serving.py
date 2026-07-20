import io

import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _client():
    from serving.app import app
    return TestClient(app)


def test_index_serves_html():
    r = _client().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Lucida" in r.text
    assert "/remove" in r.text  # the UI POSTs to the existing endpoint


def test_models_endpoint():
    r = _client().get("/models")
    assert r.status_code == 200
    data = r.json()
    assert data["default"] == "lucida"
    assert "lucida" in data["models"]
    assert data["models"] == sorted(data["models"])


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_remove_returns_png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (50, 60, 70)).save(buf, format="JPEG")
    buf.seek(0)
    with patch("serving.app._load_segmenter", return_value=FakeSeg()):
        r = _client().post(
            "/remove?decontaminate=false",
            files={"file": ("in.jpg", buf, "image/jpeg")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = Image.open(io.BytesIO(r.content))
    assert out.mode == "RGBA" and out.size == (16, 16)


def test_unknown_model_400():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    r = _client().post("/remove?model=no-such-model", files={"file": ("x.png", buf, "image/png")})
    assert r.status_code == 400


def test_invalid_upload_400():
    with patch("serving.app._load_segmenter", return_value=FakeSeg()):
        r = _client().post(
            "/remove",
            files={"file": ("garbage.png", io.BytesIO(b"this is not an image"), "image/png")},
        )
    assert r.status_code == 400
