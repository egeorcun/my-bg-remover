"""scripts/make_textfx.py testleri — küçük (<=128px) sahte kaynaklarla hızlı koşu.

Doğrulanan sözleşmeler (bkz. make_textfx modül docstring'i):
- her kategoriden im/gt çifti + `{category}_{index:05d}_c{copy:02d}` stem kalıbı,
- gt'de ARA alpha değerleri (0/255 dışı piksel > 0 — glow/parıltı yarı saydamlık şartı),
- manifest satırları `{"id", "category"}`,
- aynı seed'le ikinci koşu bit-identical (resume güvenliği),
- idempotent atlama (var olan dosya yeniden üretilmez, manifest'te tekrar yok).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_textfx as mt  # noqa: E402

COUNTS = {"text": 2, "fx": 2, "illustration": 3}  # illustration=3 -> c02 (orijinal kopya) da üretilir
CANVAS = (96, 128)
STEM_RE = re.compile(r"^(text|fx|illustration)_\d{5}_c\d{2}$")


def _write_solid(path: Path, size, color, mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


def _write_alpha(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture
def env(tmp_path):
    """Sahte arka plan havuzu + fx foreground'ları + ToonOut çiftleri."""
    bg_dir = tmp_path / "backgrounds"
    for i in range(3):
        _write_solid(bg_dir / f"bg{i}.jpg", (64, 64), (255, 0, 255))  # magenta

    fg_root = tmp_path / "fg"  # fx kaynağı: im/gt çiftli 2 foreground (binary alpha)
    for i in range(2):
        _write_solid(fg_root / "im" / f"obj{i}.jpg", (96, 96), (0, 200, 0))
        a = np.zeros((96, 96), dtype=np.uint8)
        a[24:72, 24:72] = 255
        _write_alpha(fg_root / "gt" / f"obj{i}.png", a)

    toon_dir = tmp_path / "toonout"  # illustration kaynağı: gt'de ara değerli bant
    for i in range(2):
        _write_solid(toon_dir / "im" / f"toon{i}.jpg", (96, 96), (30, 60, 200))
        a = np.full((96, 96), 255, dtype=np.uint8)
        a[:32, :] = 0
        a[32:48, :] = 128  # ara alpha kaynaktan gelir (AA kenar simülasyonu)
        _write_alpha(toon_dir / "gt" / f"toon{i}.png", a)

    return {"bg": bg_dir, "fg": [fg_root], "toon": toon_dir, "out": tmp_path / "out"}


def _run(env, out_dir=None, seed=42, counts=None):
    return mt.run(
        out_dir if out_dir is not None else env["out"],
        bg_dir=env["bg"],
        fg_dirs=env["fg"],
        toonout_dir=env["toon"],
        font_dir=None,  # PIL varsayılan fontuna düşer
        seed=seed,
        counts=dict(counts or COUNTS),
        text_canvas=CANVAS,
    )


def _manifest_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pair_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    return out_dir / "im" / f"{stem}.jpg", out_dir / "gt" / f"{stem}.png"


def test_run_generates_pairs_and_manifest(env):
    counts = _run(env)
    assert counts == {"text": 2, "fx": 2, "illustration": 3}

    rows = _manifest_rows(env["out"])
    assert len(rows) == 7
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))  # tekrar yok
    per_cat: dict[str, int] = {}
    for row in rows:
        assert set(row) == {"id", "category"}
        assert STEM_RE.match(row["id"]), row["id"]
        assert row["id"].startswith(row["category"] + "_")
        per_cat[row["category"]] = per_cat.get(row["category"], 0) + 1
    assert per_cat == {"text": 2, "fx": 2, "illustration": 3}
    # illustration çift başına 3 kopya: c00/c01 kompozit + c02 orijinal
    assert {"illustration_00000_c00", "illustration_00000_c01", "illustration_00000_c02"} <= set(ids)

    for stem in ids:
        img_path, gt_path = _pair_paths(env["out"], stem)
        assert img_path.exists() and gt_path.exists()
        img = Image.open(img_path)
        assert img.mode == "RGB"
        gt = Image.open(gt_path)
        assert gt.mode == "L"


def test_gt_has_intermediate_alpha_values(env):
    """Her kategori gt'sinde 0/255 DIŞI ara alpha piksel olmalı (glow/parıltı
    yarı saydamlık şartı — binarize edilmediğinin kanıtı)."""
    _run(env)
    for row in _manifest_rows(env["out"]):
        _, gt_path = _pair_paths(env["out"], row["id"])
        arr = np.asarray(Image.open(gt_path))
        intermediate = int(((arr > 0) & (arr < 255)).sum())
        assert intermediate > 0, f"{row['id']}: gt'de ara alpha değeri yok (binarize edilmiş?)"


def test_deterministic_same_seed_bit_identical(env):
    counts1 = _run(env, out_dir=env["out"] / "a")
    counts2 = _run(env, out_dir=env["out"] / "b")
    assert counts1 == counts2
    rows1 = {r["id"] for r in _manifest_rows(env["out"] / "a")}
    rows2 = {r["id"] for r in _manifest_rows(env["out"] / "b")}
    assert rows1 == rows2
    for stem in rows1:
        img1, gt1 = _pair_paths(env["out"] / "a", stem)
        img2, gt2 = _pair_paths(env["out"] / "b", stem)
        assert img1.read_bytes() == img2.read_bytes(), f"{stem}: aynı seed farklı image üretti"
        assert gt1.read_bytes() == gt2.read_bytes(), f"{stem}: aynı seed farklı gt üretti"


def test_different_seed_changes_output(env):
    _run(env, out_dir=env["out"] / "a", seed=42)
    _run(env, out_dir=env["out"] / "b", seed=7)
    img1, _ = _pair_paths(env["out"] / "a", "text_00000_c00")
    img2, _ = _pair_paths(env["out"] / "b", "text_00000_c00")
    assert img1.read_bytes() != img2.read_bytes()


def test_idempotent_skips_existing(env):
    counts1 = _run(env)
    assert sum(counts1.values()) == 7
    counts2 = _run(env)
    assert counts2 == {}  # ikinci koşuda hiçbir yeni öğe üretilmedi
    rows = _manifest_rows(env["out"])
    assert len(rows) == 7  # manifest'te tekrar/duplicate yok


def test_partial_then_resume_matches_full_run(env):
    """Kesinti simülasyonu: tam koşu sonrası bazı çiftler (dosyalar + manifest
    satırları) silinir ve yeniden koşulur — devam koşusu kesintisiz koşuyla
    bit-birebir aynı dosyaları üretmeli (SeedSequence alt-seed'leri sıradan
    bağımsız)."""
    dir_full = env["out"] / "full"
    dir_resume = env["out"] / "resume"
    _run(env, out_dir=dir_full)
    _run(env, out_dir=dir_resume)

    rows = _manifest_rows(dir_resume)
    keep, drop = rows[::2], rows[1::2]
    assert drop
    for row in drop:
        img_path, gt_path = _pair_paths(dir_resume, row["id"])
        img_path.unlink()
        gt_path.unlink()
    manifest = dir_resume / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))

    counts = _run(env, out_dir=dir_resume)
    assert sum(counts.values()) == len(drop)  # yalnız silinenler yeniden üretildi

    full_ids = {r["id"] for r in _manifest_rows(dir_full)}
    resume_ids = {r["id"] for r in _manifest_rows(dir_resume)}
    assert full_ids == resume_ids
    for stem in full_ids:
        img_f, gt_f = _pair_paths(dir_full, stem)
        img_r, gt_r = _pair_paths(dir_resume, stem)
        assert img_f.read_bytes() == img_r.read_bytes(), f"{stem}: devam koşusu image'ı farklı"
        assert gt_f.read_bytes() == gt_r.read_bytes(), f"{stem}: devam koşusu gt'si farklı"


def test_file_exists_but_manifest_row_missing_gets_completed(env):
    """Dosya kaydı ile manifest append'i arasında kesinti: dosyalar durur,
    satır eksik — yeniden koşuda dosya ÜRETİLMEZ, yalnız satır tamamlanır."""
    _run(env)
    manifest = env["out"] / "manifest.jsonl"
    rows = _manifest_rows(env["out"])
    dropped = rows[0]
    img_path, _ = _pair_paths(env["out"], dropped["id"])
    before_bytes = img_path.read_bytes()
    before_mtime = img_path.stat().st_mtime_ns
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[1:]))

    counts = _run(env)
    assert counts == {}  # dosya zaten vardı -> üretim sayılmadı
    after = _manifest_rows(env["out"])
    assert {r["id"] for r in after} == {r["id"] for r in rows}
    assert img_path.read_bytes() == before_bytes
    assert img_path.stat().st_mtime_ns == before_mtime  # dosyaya hiç dokunulmadı


def test_zero_count_category_is_skipped_without_inputs(env, tmp_path):
    """counts'ta 0 olan kategori girdileri hiç aranmadan atlanır — yalnız text
    üretimi fg/toonout OLMADAN çalışabilmeli."""
    out = tmp_path / "only_text"
    counts = mt.run(
        out, bg_dir=env["bg"], fg_dirs=None, toonout_dir=None, font_dir=None,
        seed=42, counts={"text": 2, "fx": 0, "illustration": 0}, text_canvas=CANVAS,
    )
    assert counts == {"text": 2}
    assert len(_manifest_rows(out)) == 2


def test_fx_requires_sources(env, tmp_path):
    with pytest.raises(SystemExit, match="fx"):
        mt.run(
            tmp_path / "o", bg_dir=env["bg"], fg_dirs=[], toonout_dir=None,
            seed=42, counts={"text": 0, "fx": 2, "illustration": 0}, text_canvas=CANVAS,
        )


def test_gt_alpha_covers_fg_and_fx_glow(env):
    """fx gt'si: nesne çekirdeği tam opak (fg alpha korunur) + nesne DIŞINDA
    yarı saydam glow halkası (max(fg, fx) birleşimi)."""
    _run(env)
    arr = np.asarray(Image.open(_pair_paths(env["out"], "fx_00000_c00")[1]))
    assert arr[48, 48] == 255  # nesne merkezi (24:72 karesi) tam opak
    outside = arr.copy()
    outside[24:72, 24:72] = 0
    assert int(((outside > 0) & (outside < 255)).sum()) > 0  # dışarıda yarı saydam parıltı


def test_parse_counts():
    assert mt._parse_counts("text=10,fx=5") == {"text": 10, "fx": 5, "illustration": 0}
    assert mt._parse_counts("illustration=3600") == {"text": 0, "fx": 0, "illustration": 3600}
    with pytest.raises(SystemExit):
        mt._parse_counts("bogus=1")
    with pytest.raises(SystemExit):
        mt._parse_counts("text")
