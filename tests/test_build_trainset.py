import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_trainset as bt  # noqa: E402
from benchmark.testset import append_entries, load_manifest  # noqa: E402


def _make_pair(img_dir: Path, gt_dir: Path, stem: str) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "red").save(img_dir / f"{stem}.jpg")
    Image.new("L", (4, 4), 128).save(gt_dir / f"{stem}.png")


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    src_img = tmp_path / "raw" / "im"
    src_gt = tmp_path / "raw" / "gt"
    for i in range(5):
        _make_pair(src_img, src_gt, f"item{i}")

    out_img = tmp_path / "train" / "images"
    out_gt = tmp_path / "train" / "gt"
    manifest = tmp_path / "train" / "manifest.jsonl"
    out_img.mkdir(parents=True)
    out_gt.mkdir(parents=True)

    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "OUT_IMG", out_img)
    monkeypatch.setattr(bt, "OUT_GT", out_gt)
    monkeypatch.setattr(bt, "MANIFEST", manifest)

    return {
        "img_glob": "raw/im/*",
        "gt_glob": "raw/gt/*",
        "src_img": src_img,
        "src_gt": src_gt,
        "manifest": manifest,
    }


def test_sample_source_produces_manifest_rows(fake_root):
    rows = bt.sample_source("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 3)
    assert len(rows) == 3
    for row in rows:
        assert row["category"] == "product"
        assert row["id"].startswith("fake_")
        assert row["gt_alpha"] is not None


def test_sample_source_links_resolve_to_original_files(fake_root):
    rows = bt.sample_source("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)
    for row in rows:
        dst_i = bt.ROOT / row["image"]
        dst_g = bt.ROOT / row["gt_alpha"]
        assert dst_i.is_symlink()
        assert dst_g.is_symlink()
        assert dst_i.resolve().parent == fake_root["src_img"].resolve()
        assert dst_g.resolve().parent == fake_root["src_gt"].resolve()
        # the symlink's content must be readable (not a copy; it points to the real file)
        Image.open(dst_i).verify()


def test_sample_source_writes_valid_manifest(fake_root):
    rows = bt.sample_source("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)
    append_entries(str(fake_root["manifest"]), rows)
    loaded = load_manifest(str(fake_root["manifest"]))
    assert loaded == rows


def test_duplicate_id_rejected_on_reload(fake_root):
    rows = bt.sample_source("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)
    append_entries(str(fake_root["manifest"]), rows)
    append_entries(str(fake_root["manifest"]), rows)  # the same ids were added again
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(str(fake_root["manifest"]))


def test_sample_disvd_tokens_classifies_thin_and_complex(tmp_path, monkeypatch):
    src_img = tmp_path / "raw" / "im"
    src_gt = tmp_path / "raw" / "gt"
    thin_stem = "20#Sports#8#Racket#1234_abcd_o"
    complex_stem = "11#Furniture#4#Chair#5678_efgh_o"
    _make_pair(src_img, src_gt, thin_stem)
    _make_pair(src_img, src_gt, complex_stem)

    out_img = tmp_path / "train" / "images"
    out_gt = tmp_path / "train" / "gt"
    manifest = tmp_path / "train" / "manifest.jsonl"
    out_img.mkdir(parents=True)
    out_gt.mkdir(parents=True)
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "OUT_IMG", out_img)
    monkeypatch.setattr(bt, "OUT_GT", out_gt)
    monkeypatch.setattr(bt, "MANIFEST", manifest)

    rows = bt.sample_disvd_tokens("dis5ktr", "raw/im/*", "raw/gt/*", n=2)
    categories = {r["id"]: r["category"] for r in rows}
    assert any(cat == "thin" for cat in categories.values())
    assert any(cat == "complex" for cat in categories.values())


def test_add_source_appends_known_source(fake_root, monkeypatch):
    monkeypatch.setattr(
        bt,
        "SOURCES",
        [("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)],
    )
    bt.add_source("fake")
    loaded = load_manifest(str(fake_root["manifest"]))
    assert len(loaded) == 2


def test_add_source_unknown_raises(fake_root):
    with pytest.raises(SystemExit):
        bt.add_source("nope")


def test_sample_source_gt_stem_suffix_strips_before_pairing(tmp_path, monkeypatch):
    """In sources like HIM2K the GT file is '<stem>_matte.png', the image '<stem>.jpg' —
    the default (exact stem) pairing NEVER catches this (0 pairs); with gt_stem_suffix
    the suffix must be stripped from the GT stem and the pairing established."""
    src_img = tmp_path / "raw" / "im"
    src_gt = tmp_path / "raw" / "gt"
    src_img.mkdir(parents=True)
    src_gt.mkdir(parents=True)
    for stem in ("foo", "bar"):
        Image.new("RGB", (4, 4), "red").save(src_img / f"{stem}.jpg")
        Image.new("L", (4, 4), 128).save(src_gt / f"{stem}_matte.png")

    out_img = tmp_path / "train" / "images"
    out_gt = tmp_path / "train" / "gt"
    manifest = tmp_path / "train" / "manifest.jsonl"
    out_img.mkdir(parents=True)
    out_gt.mkdir(parents=True)
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "OUT_IMG", out_img)
    monkeypatch.setattr(bt, "OUT_GT", out_gt)
    monkeypatch.setattr(bt, "MANIFEST", manifest)

    # Default (no gt_stem_suffix): 0 pairs, because "foo" != "foo_matte".
    rows_default = bt.sample_source("him2k", "raw/im/*", "raw/gt/*", "general")
    assert rows_default == []

    # gt_stem_suffix="_matte": the suffix is stripped from the GT stem, 2 pairs match.
    rows = bt.sample_source("him2k", "raw/im/*", "raw/gt/*", "general", gt_stem_suffix="_matte")
    assert len(rows) == 2
    for row in rows:
        assert row["category"] == "general"
        dst_g = bt.ROOT / row["gt_alpha"]
        assert dst_g.resolve().name.endswith("_matte.png")


def test_sample_source_default_mode_uses_symlinks(fake_root):
    rows = bt.sample_source("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)
    for row in rows:
        assert (bt.ROOT / row["image"]).is_symlink()
        assert (bt.ROOT / row["gt_alpha"]).is_symlink()


def test_sample_source_copy_mode_creates_real_files_not_symlinks(fake_root):
    """--copy for full data materialization on Colab: a real file instead of a
    symlink (the link cannot break during the move/zip to Drive)."""
    rows = bt.sample_source(
        "fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2, copy=True
    )
    assert len(rows) == 2
    for row in rows:
        dst_i = bt.ROOT / row["image"]
        dst_g = bt.ROOT / row["gt_alpha"]
        assert not dst_i.is_symlink()
        assert not dst_g.is_symlink()
        assert dst_i.is_file()
        assert dst_g.is_file()
        Image.open(dst_i).verify()


def test_sample_disvd_tokens_copy_mode_creates_real_files(tmp_path, monkeypatch):
    src_img = tmp_path / "raw" / "im"
    src_gt = tmp_path / "raw" / "gt"
    stem = "20#Sports#8#Racket#1234_abcd_o"
    _make_pair(src_img, src_gt, stem)

    out_img = tmp_path / "train" / "images"
    out_gt = tmp_path / "train" / "gt"
    manifest = tmp_path / "train" / "manifest.jsonl"
    out_img.mkdir(parents=True)
    out_gt.mkdir(parents=True)
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "OUT_IMG", out_img)
    monkeypatch.setattr(bt, "OUT_GT", out_gt)
    monkeypatch.setattr(bt, "MANIFEST", manifest)

    rows = bt.sample_disvd_tokens("dis5ktr", "raw/im/*", "raw/gt/*", n=1, copy=True)
    assert len(rows) == 1
    assert not (bt.ROOT / rows[0]["image"]).is_symlink()
    assert (bt.ROOT / rows[0]["image"]).is_file()


def test_add_source_copy_mode_threads_through(fake_root, monkeypatch):
    monkeypatch.setattr(
        bt,
        "SOURCES",
        [("fake", fake_root["img_glob"], fake_root["gt_glob"], "product", 2)],
    )
    bt.add_source("fake", copy=True)
    loaded = load_manifest(str(fake_root["manifest"]))
    assert len(loaded) == 2
    for row in loaded:
        assert not (bt.ROOT / row["image"]).is_symlink()


def test_sources_derived_from_source_specs():
    """SOURCES must be derived from the single source of truth SOURCE_SPECS — the
    Colab notebook uses the same definitions without n (full set); no manual
    copies/drift allowed."""
    assert bt.SOURCES, "SOURCES must not be empty"
    for name, img_glob, gt_glob, category, n in bt.SOURCES:
        spec = bt.SOURCE_SPECS[name]
        assert spec["img_glob"] == img_glob
        assert spec["gt_glob"] == gt_glob
        assert spec["category"] == category
        assert category != "disvd_tokens"  # the token rule never enters sample_source
        assert n == bt.LOCAL_SAMPLE_N[name]


def test_source_specs_complete_and_dis5ktr_uses_token_rule():
    for name, spec in bt.SOURCE_SPECS.items():
        assert set(spec) == {"img_glob", "gt_glob", "category"}, name
        assert name in bt.LOCAL_SAMPLE_N, f"{name}: LOCAL_SAMPLE_N missing"
    assert bt.SOURCE_SPECS["dis5ktr"]["category"] == "disvd_tokens"
    assert bt.DIS5KTR_IMG_GLOB == bt.SOURCE_SPECS["dis5ktr"]["img_glob"]
    assert bt.DIS5KTR_GT_GLOB == bt.SOURCE_SPECS["dis5ktr"]["gt_glob"]
    assert bt.DIS5KTR_N == bt.LOCAL_SAMPLE_N["dis5ktr"]
