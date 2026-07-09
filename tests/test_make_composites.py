import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_composites as mc  # noqa: E402

from benchmark.testset import append_entries, load_manifest  # noqa: E402


def _write_solid(path: Path, size, color, mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


@pytest.fixture
def env(tmp_path):
    """Sahte manifest + sahte arka plan havuzu (belirgin magenta renk, kompozit
    tespitini kolaylaştırmak için)."""
    src = tmp_path / "src"
    bg_dir = tmp_path / "backgrounds"
    for i in range(3):
        _write_solid(bg_dir / f"bg{i}.jpg", (20, 20), (255, 0, 255))  # magenta

    manifest = tmp_path / "train_manifest.jsonl"
    rows = []

    def _add(name, category, alpha_partial=True, with_gt=True):
        _write_solid(src / f"{name}.jpg", (16, 16), (0, 200, 0))  # yesil fg
        gt_alpha = None
        if with_gt:
            gt_path = src / f"{name}_gt.png"
            a = np.full((16, 16), 255, dtype=np.uint8)
            if alpha_partial:
                a[:8, :] = 128  # ust yari kismi saydam -> compose izi burada gorulur
            Image.fromarray(a, mode="L").save(gt_path)
            gt_alpha = str(gt_path)
        rows.append(
            {"id": name, "image": str(src / f"{name}.jpg"), "category": category, "gt_alpha": gt_alpha}
        )

    _add("cam1", "camouflage")
    _add("trans1", "transparent")
    _add("hair1", "hair")
    _add("nogt1", "product", with_gt=False)

    append_entries(str(manifest), rows)

    return {"manifest": manifest, "backgrounds": bg_dir, "src": src, "out": tmp_path / "out"}


def test_multiplier_values():
    assert mc.multiplier("transparent") == 10
    assert mc.multiplier("camouflage") == 2
    assert mc.multiplier("hair") == 1
    assert mc.multiplier("complex") == 1


def test_run_counts_follow_category_multipliers(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    assert counts["camouflage"] == 2
    assert counts["transparent"] == 10
    assert counts["hair"] == 1
    assert "product" not in counts  # gt_alpha=None -> compositing'e dahil edilmez


def test_run_per_image_multiplies_category_factor(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=3, seed=42, out_dir=env["out"])
    assert counts["camouflage"] == 6
    assert counts["transparent"] == 30
    assert counts["hair"] == 3


def test_run_writes_valid_manifest(env):
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    out_manifest = env["out"] / "manifest.jsonl"
    loaded = load_manifest(str(out_manifest))
    assert len(loaded) == 2 + 10 + 1  # camouflage + transparent + hair
    for row in loaded:
        assert Path(row["image"]).exists()
        assert Path(row["gt_alpha"]).exists()
        img = Image.open(row["image"])
        assert img.mode == "RGB"
        alpha = np.asarray(Image.open(row["gt_alpha"]).convert("L"), dtype=np.float32) / 255.0
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_run_camouflage_skips_compose_no_bg_contamination(env):
    """Kamuflaj kategorisinde compose ATLANIR: kısmi saydam (alpha=0.5) bölgede
    magenta arka plan havuzundan hiçbir sızıntı olmamalı (yalnız augment uygulanır,
    orijinal arka plan/renk korunur)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    cam_rows = [r for r in loaded if r["category"] == "camouflage"]
    assert cam_rows
    for row in cam_rows:
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
        # magenta bg (255,0,255) karisimis olsaydi R+B kanallari cok yuksek olurdu;
        # yesil fg + jitter/jpeg varyansiyla bile R+B toplami dusuk kalmali.
        assert rgb[..., 0].mean() + rgb[..., 2].mean() < 150


def test_run_transparent_does_compose_bg_contamination_present(env):
    """Saydam kategori compose'lu: kısmi saydam bölgede magenta arka plandan
    belirgin sızıntı olmalı (yeşil fg + magenta bg karışımı -> yüksek R+B)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    trans_rows = [r for r in loaded if r["category"] == "transparent"]
    assert trans_rows
    contaminated = False
    for row in trans_rows:
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
        if rgb[..., 0].mean() + rgb[..., 2].mean() > 150:
            contaminated = True
    assert contaminated, "compose uygulanan hiçbir transparent kopyada bg sızıntısı görülmedi"


def test_run_deterministic_same_seed(env):
    counts1 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"] / "a")
    counts2 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"] / "b")
    assert counts1 == counts2
    rows1 = {r["id"]: r for r in load_manifest(str(env["out"] / "a" / "manifest.jsonl"))}
    rows2 = {r["id"]: r for r in load_manifest(str(env["out"] / "b" / "manifest.jsonl"))}
    assert rows1.keys() == rows2.keys()
    for rid in rows1:
        img1 = np.asarray(Image.open(rows1[rid]["image"]))
        img2 = np.asarray(Image.open(rows2[rid]["image"]))
        assert np.array_equal(img1, img2), f"{rid}: aynı seed farklı çıktı üretti"


def test_run_idempotent_skips_existing(env):
    counts1 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    total1 = sum(counts1.values())
    assert total1 > 0
    counts2 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    assert counts2 == {}  # ikinci koşuda hiçbir yeni öğe üretilmedi
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    assert len(loaded) == total1  # manifest'te tekrar/duplicate yok


def test_partial_then_resume_matches_full_run(env):
    """Kesinti simülasyonu: tam koşu sonrası çıktının YARISI (dosyalar + manifest
    satırları) silinir ve yeniden koşulur. Devam koşusu, kesintisiz tam koşuyla
    bit-birebir aynı dosyaları üretmeli (SeedSequence alt-seed'leri sıradan bağımsız)."""
    dir_full = env["out"] / "full"
    dir_resume = env["out"] / "resume"
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_full)
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_resume)

    # yarıyı sil: manifest'ten her ikinci satır + o satırların dosyaları
    resume_manifest = dir_resume / "manifest.jsonl"
    rows = load_manifest(str(resume_manifest))
    keep, drop = rows[::2], rows[1::2]
    assert drop, "test anlamlı değil: silinecek satır yok"
    for row in drop:
        Path(row["image"]).unlink()
        Path(row["gt_alpha"]).unlink()
    resume_manifest.unlink()
    append_entries(str(resume_manifest), keep)

    # devam koşusu: yalnız silinenler yeniden üretilmeli
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_resume)
    assert sum(counts.values()) == len(drop)

    full_rows = {r["id"]: r for r in load_manifest(str(dir_full / "manifest.jsonl"))}
    resume_rows = {r["id"]: r for r in load_manifest(str(resume_manifest))}
    assert full_rows.keys() == resume_rows.keys()
    for rid, full_row in full_rows.items():
        resume_row = resume_rows[rid]
        assert Path(full_row["image"]).read_bytes() == Path(resume_row["image"]).read_bytes(), (
            f"{rid}: devam koşusu image'ı tam koşudan farklı"
        )
        assert Path(full_row["gt_alpha"]).read_bytes() == Path(resume_row["gt_alpha"]).read_bytes(), (
            f"{rid}: devam koşusu gt'si tam koşudan farklı"
        )


def test_run_limit_caps_source_rows(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"], limit=1)
    assert sum(counts.values()) < 2 + 4 + 1
