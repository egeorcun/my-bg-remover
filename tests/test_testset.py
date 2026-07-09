import json

import pytest

from benchmark.testset import append_entries, load_manifest


def test_roundtrip(tmp_path):
    p = tmp_path / "m.jsonl"
    rows = [
        {"id": "a1", "image": "data/x/a1.jpg", "category": "hair", "gt_alpha": "data/x/a1.png"},
        {"id": "b2", "image": "data/x/b2.jpg", "category": "product", "gt_alpha": None},
    ]
    append_entries(str(p), rows)
    assert load_manifest(str(p)) == rows


def test_invalid_category_raises(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": "x", "image": "i.jpg", "category": "ucan-kus", "gt_alpha": None}) + "\n")
    with pytest.raises(ValueError, match="kategori"):
        load_manifest(str(p))


def test_missing_key_raises(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": "x", "image": "i.jpg"}) + "\n")
    with pytest.raises(ValueError, match="anahtar"):
        load_manifest(str(p))


def test_duplicate_id_raises(tmp_path):
    p = tmp_path / "m.jsonl"
    row = {"id": "dup", "image": "i.jpg", "category": "hair", "gt_alpha": None}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="dup"):
        load_manifest(str(p))
