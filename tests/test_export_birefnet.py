import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import export_birefnet as eb  # noqa: E402

from benchmark.testset import append_entries  # noqa: E402


def _make_pair(img_dir: Path, gt_dir: Path, stem: str, size=(20, 10), alpha_val=200):
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{stem}.png"
    gt_path = gt_dir / f"{stem}_gt.png"
    Image.new("RGB", size, (10, 20, 30)).save(img_path)
    a = np.full((size[1], size[0]), alpha_val, dtype=np.uint8)
    Image.fromarray(a, mode="L").save(gt_path)
    return img_path, gt_path


@pytest.fixture
def fake_manifest(tmp_path):
    img_dir = tmp_path / "src" / "im"
    gt_dir = tmp_path / "src" / "gt"
    manifest = tmp_path / "manifest.jsonl"
    rows = []

    # (id, kategori, boyut, alpha_deger) — alpha 0.05<a<0.95 aralığı "soft" sayılır.
    specs = [
        ("cam1", "camouflage", (20, 10), 200),  # 200/255=0.78 -> soft
        ("cam2", "camouflage", (20, 10), 255),  # 1.0 -> hard (soft değil)
        ("trans1", "transparent", (30, 15), 128),  # 0.50 -> soft
        ("hair1", "hair", (16, 16), 0),  # 0.0 -> hard
    ]
    for stem, cat, size, aval in specs:
        img_path, gt_path = _make_pair(img_dir, gt_dir, stem, size=size, alpha_val=aval)
        rows.append(
            {"id": stem, "image": str(img_path), "category": cat, "gt_alpha": str(gt_path)}
        )
    append_entries(str(manifest), rows)
    return {"manifest": manifest, "rows": rows}


def test_export_creates_expected_layout(tmp_path, fake_manifest):
    out = tmp_path / "out"
    stats = eb.export(fake_manifest["manifest"], out, split_name="TRAIN")

    im_dir = out / "TRAIN" / "im"
    gt_dir = out / "TRAIN" / "gt"
    for row in fake_manifest["rows"]:
        stem = row["id"]
        img_path = im_dir / f"{stem}.jpg"
        gt_path = gt_dir / f"{stem}.png"
        assert img_path.exists()
        assert gt_path.exists()
        with Image.open(img_path) as im:
            assert im.mode == "RGB"
            assert im.format == "JPEG"
        with Image.open(gt_path) as gt:
            assert gt.mode == "L"

    assert stats["total"] == 4
    assert (out / "stats.json").exists()
    on_disk = json.loads((out / "stats.json").read_text())
    assert on_disk == stats


def test_export_default_split_name_is_train(tmp_path, fake_manifest):
    out = tmp_path / "out"
    eb.export(fake_manifest["manifest"], out)
    assert (out / "TRAIN" / "im" / "cam1.jpg").exists()


def test_stats_category_counts(tmp_path, fake_manifest):
    out = tmp_path / "out"
    stats = eb.export(fake_manifest["manifest"], out, split_name="TRAIN")
    assert stats["category_counts"] == {"camouflage": 2, "transparent": 1, "hair": 1}


def test_stats_resolution_percentiles(tmp_path, fake_manifest):
    out = tmp_path / "out"
    stats = eb.export(fake_manifest["manifest"], out, split_name="TRAIN")
    p = stats["resolution_short_side_percentiles"]
    # kısa kenarlar: cam1/cam2 -> 10, trans1 -> 15, hair1 -> 16
    assert p["p10"] <= p["p50"] <= p["p90"]
    assert 10 <= p["p10"]
    assert p["p90"] <= 16


def test_stats_soft_alpha_ratio(tmp_path, fake_manifest):
    out = tmp_path / "out"
    stats = eb.export(fake_manifest["manifest"], out, split_name="TRAIN")
    ratios = stats["soft_alpha_ratio_by_category"]
    assert ratios["camouflage"] == pytest.approx(0.5)  # cam1 soft(1.0) + cam2 hard(0.0) -> ort 0.5
    assert ratios["transparent"] == pytest.approx(1.0)
    assert ratios["hair"] == pytest.approx(0.0)


def test_export_idempotent_skips_existing(tmp_path, fake_manifest):
    out = tmp_path / "out"
    eb.export(fake_manifest["manifest"], out, split_name="TRAIN")
    img_path = out / "TRAIN" / "im" / "cam1.jpg"
    mtime1 = img_path.stat().st_mtime_ns
    stats2 = eb.export(fake_manifest["manifest"], out, split_name="TRAIN")
    mtime2 = img_path.stat().st_mtime_ns
    assert mtime1 == mtime2  # yeniden yazılmadı
    assert stats2["total"] == 4  # stats yine de doğru hesaplanır


def test_export_duplicate_stem_raises(tmp_path):
    manifest = tmp_path / "dup_manifest.jsonl"
    img_dir = tmp_path / "src" / "im"
    gt_dir = tmp_path / "src" / "gt"
    img_path, gt_path = _make_pair(img_dir, gt_dir, "dupA")
    row = {"id": "dup", "image": str(img_path), "category": "product", "gt_alpha": str(gt_path)}
    # manuel olarak aynı id iki kez yazılır (append_entries kendi başına tekrar
    # kontrolü yapmaz — load_manifest okurken "tekrarlanan id" hatası verir).
    with open(manifest, "w") as f:
        f.write(json.dumps(row) + "\n")
        f.write(json.dumps(row) + "\n")

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="tekrarlanan"):
        eb.export(manifest, out, split_name="TRAIN")


def test_export_skips_rows_without_gt(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    img_dir = tmp_path / "src" / "im"
    gt_dir = tmp_path / "src" / "gt"
    img_path, gt_path = _make_pair(img_dir, gt_dir, "withgt")
    rows = [
        {"id": "withgt", "image": str(img_path), "category": "product", "gt_alpha": str(gt_path)},
        {"id": "nogt", "image": str(img_path), "category": "product", "gt_alpha": None},
    ]
    append_entries(str(manifest), rows)
    out = tmp_path / "out"
    stats = eb.export(manifest, out, split_name="TRAIN")
    assert stats["total"] == 1
    assert not (out / "TRAIN" / "im" / "nogt.jpg").exists()
