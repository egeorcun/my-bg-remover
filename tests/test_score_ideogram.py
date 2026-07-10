"""`scripts/score_ideogram.py` için hermetik testler: sahte küçük RGBA ideogram
çıktıları -> metrik hesaplama + `metrics.json`'a birleştirme (merge)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import score_ideogram as si  # noqa: E402


def _write_alpha_png(path: Path, alpha_row: list[int], rgba: bool) -> None:
    """2x2'lik minik bir görsel; `alpha_row` iki değer -> [[a0,a0],[a1,a1]] alfa deseni."""
    h, w = 2, 2
    arr = np.zeros((h, w, 4 if rgba else 1), dtype=np.uint8)
    for y in range(h):
        val = alpha_row[y]
        if rgba:
            arr[y, :, 0] = 200  # R (alaka yok, decontaminate testi değil)
            arr[y, :, 1] = 100
            arr[y, :, 2] = 50
            arr[y, :, 3] = val
        else:
            arr[y, :, 0] = val
    mode = "RGBA" if rgba else "L"
    img = Image.fromarray(arr.squeeze(-1) if not rgba else arr, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _make_fixture(tmp_path: Path, with_missing: bool = False) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.jsonl"
    gt_dir = tmp_path / "gt"
    ideogram_dir = tmp_path / "ideogram"

    rows = [
        {"id": "a1", "image": "unused.jpg", "category": "camouflage", "gt_alpha": str(gt_dir / "a1.png")},
        {"id": "a2", "image": "unused.jpg", "category": "transparent", "gt_alpha": str(gt_dir / "a2.png")},
        {"id": "a3", "image": "unused.jpg", "category": "camouflage", "gt_alpha": ""},  # GT'siz -> atlanmalı
    ]
    if with_missing:
        rows.append(
            {"id": "a4", "image": "unused.jpg", "category": "hair", "gt_alpha": str(gt_dir / "a4.png")}
        )

    manifest_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # a1: ideogram GT ile BİREBİR aynı (MAE=0 beklenir).
    _write_alpha_png(gt_dir / "a1.png", [255, 255], rgba=False)
    _write_alpha_png(ideogram_dir / "a1.png", [255, 255], rgba=True)

    # a2: ideogram GT'den FARKLI (MAE>0 beklenir).
    _write_alpha_png(gt_dir / "a2.png", [255, 255], rgba=False)
    _write_alpha_png(ideogram_dir / "a2.png", [0, 0], rgba=True)

    if with_missing:
        _write_alpha_png(gt_dir / "a4.png", [128, 128], rgba=False)
        # a4.png ideogram dizininde KASITLI OLARAK YOK (fal API başarısız simülasyonu).

    return manifest_path, ideogram_dir


def test_score_ideogram_computes_metrics_for_gt_rows(tmp_path):
    manifest_path, ideogram_dir = _make_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.json"

    result = si.score_ideogram(str(ideogram_dir), str(manifest_path), str(metrics_path))

    assert set(result["per_image"]["ideogram"]) == {"a1", "a2"}  # a3 (GT'siz) dahil değil
    assert result["per_image"]["ideogram"]["a1"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["per_image"]["ideogram"]["a2"]["mae"] == pytest.approx(1.0, abs=1e-6)

    per_cat = result["per_category"]["ideogram"]
    assert per_cat["camouflage"]["mae"] == pytest.approx(0.0, abs=1e-6)  # yalnız a1 (camouflage)
    assert per_cat["transparent"]["mae"] == pytest.approx(1.0, abs=1e-6)  # yalnız a2 (transparent)

    overall = result["overall"]["ideogram"]
    assert overall["mae"] == pytest.approx(0.5, abs=1e-6)  # (0.0 + 1.0) / 2

    assert metrics_path.exists()
    on_disk = json.loads(metrics_path.read_text())
    assert on_disk == result


def test_score_ideogram_skips_missing_output_without_crashing(tmp_path, capsys):
    manifest_path, ideogram_dir = _make_fixture(tmp_path, with_missing=True)
    metrics_path = tmp_path / "metrics.json"

    result = si.score_ideogram(str(ideogram_dir), str(manifest_path), str(metrics_path))

    assert "a4" not in result["per_image"]["ideogram"]  # ideogram çıktısı yok -> atlandı
    assert set(result["per_image"]["ideogram"]) == {"a1", "a2"}
    captured = capsys.readouterr()
    assert "a4" in captured.out
    assert "UYARI" in captured.out


def test_score_ideogram_merges_into_existing_metrics_json(tmp_path):
    manifest_path, ideogram_dir = _make_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.json"

    existing = {
        "per_image": {"birefnet-hr": {"a1": {"mae": 0.05}}},
        "per_category": {"birefnet-hr": {"camouflage": {"mae": 0.05}}},
        "overall": {"birefnet-hr": {"mae": 0.05}},
    }
    metrics_path.write_text(json.dumps(existing))

    result = si.score_ideogram(str(ideogram_dir), str(manifest_path), str(metrics_path))

    # var olan birefnet-hr girdileri KORUNMALI, yalnız ideogram eklenmeli.
    assert result["overall"]["birefnet-hr"] == {"mae": 0.05}
    assert "ideogram" in result["overall"]
    assert "ideogram" in result["per_category"]
    assert "ideogram" in result["per_image"]

    on_disk = json.loads(metrics_path.read_text())
    assert on_disk == result


def test_score_ideogram_no_gt_rows_produces_empty_overall(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"id": "z1", "image": "unused.jpg", "category": "general", "gt_alpha": ""}) + "\n"
    )
    metrics_path = tmp_path / "metrics.json"
    ideogram_dir = tmp_path / "ideogram"
    ideogram_dir.mkdir()

    result = si.score_ideogram(str(ideogram_dir), str(manifest_path), str(metrics_path))
    assert result["per_image"]["ideogram"] == {}
    assert result["overall"]["ideogram"] == {}
