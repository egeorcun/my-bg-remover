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
    assert counts["camouflage"] == 2  # NO_COMPOSE_CATEGORIES -> _o00 yok
    assert counts["transparent"] == 10 + 1  # _v x10 + v3 _o00 x1
    assert counts["hair"] == 1 + 1  # _v x1 + v3 _o00 x1
    assert "product" not in counts  # gt_alpha=None -> compositing'e dahil edilmez


def test_run_per_image_multiplies_category_factor(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=3, seed=42, out_dir=env["out"])
    assert counts["camouflage"] == 6
    # ORIGINAL_BG_COPIES (_o00) per_image ile ÖLÇEKLENMEZ -- sabit +1.
    assert counts["transparent"] == 30 + 1
    assert counts["hair"] == 3 + 1


def test_run_writes_valid_manifest(env):
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    out_manifest = env["out"] / "manifest.jsonl"
    loaded = load_manifest(str(out_manifest))
    # camouflage(2) + transparent(10 _v + 1 _o00) + hair(1 _v + 1 _o00)
    assert len(loaded) == 2 + 11 + 2
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


# ============================================================================
# v3 — orijinal arka plan kopyaları (_o<NN>)
# ============================================================================
def test_run_generates_o00_for_non_camouflage_categories(env):
    """camouflage HARİÇ her kategori (transparent, hair) için 1 adet ekstra
    `_o00` kopyası üretilmeli; camouflage zaten compose'suz olduğundan _o00
    üretilmemeli (redundant)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert "trans1_o00" in ids
    assert "hair1_o00" in ids
    assert "cam1_o00" not in ids  # NO_COMPOSE_CATEGORIES -> _o00 YOK


def test_run_o00_keeps_original_background_no_compose_contamination(env):
    """_o00 kopyası camouflage'ın yolunu izler: compose YOK, yalnız augment —
    magenta arka plan havuzundan sızıntı olmamalı."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = {r["id"]: r for r in load_manifest(str(env["out"] / "manifest.jsonl"))}
    row = loaded["trans1_o00"]
    rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
    assert rgb[..., 0].mean() + rgb[..., 2].mean() < 150


def test_run_o00_respects_exclude_source_ids(env):
    """`exclude_source_ids`'teki kaynak id'ler için _o00 üretilmemeli (VAL sızıntı
    koruması) — _v<NN> kopyaları ETKİLENMEMELİ."""
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        exclude_source_ids={"trans1"},
    )
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert "trans1_o00" not in ids
    assert "hair1_o00" in ids  # hariç tutulmayan diğer kaynak etkilenmedi
    assert counts["transparent"] == 10  # yalnız _v<NN>'ler (10 tane), _o00 yok


def test_run_only_original_bg_skips_all_v_copies(env):
    """`only_original_bg=True`: _v<NN> kopyaları TAMAMEN atlanır, yalnız _o00 seti
    üretilir (taze bir VM'de tüm kompozit setini yeniden üretmeden hızlı devam)."""
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        only_original_bg=True,
    )
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert ids == {"trans1_o00", "hair1_o00"}  # camouflage hariç, _v<NN> hiç yok
    assert counts == {"transparent": 1, "hair": 1}


def test_run_only_original_bg_with_exclusion_produces_nothing_for_excluded(env):
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        only_original_bg=True, exclude_source_ids={"trans1", "hair1"},
    )
    assert counts == {}
    # hiç yeni girdi yazılmadığından manifest dosyası hiç oluşturulmaz (append_entries
    # yalnız new_entries doluysa çağrılır) -- boş bir manifest.jsonl DEĞİL, yok.
    assert not (env["out"] / "manifest.jsonl").exists()


def test_run_o00_naming_never_collides_with_v_copies(env):
    """`_o00` isim alanı `_v<NN>` ile ASLA çakışmaz (ayrı bir son ek) — normal (v3
    öncesi gibi) bir koşuda üretilen tüm id'ler arasında hem _v hem _o kopyaları
    bağımsız var olabilir, hiçbiri diğerinin üzerine yazmaz."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = [r["id"] for r in loaded]
    assert len(ids) == len(set(ids))  # tekrar yok
    trans_v = [i for i in ids if i.startswith("trans1_v")]
    trans_o = [i for i in ids if i.startswith("trans1_o")]
    assert len(trans_v) == 10
    assert trans_o == ["trans1_o00"]


def test_run_o00_deterministic_same_seed(env):
    counts1 = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"] / "a",
        only_original_bg=True,
    )
    counts2 = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"] / "b",
        only_original_bg=True,
    )
    assert counts1 == counts2
    rows1 = {r["id"]: r for r in load_manifest(str(env["out"] / "a" / "manifest.jsonl"))}
    rows2 = {r["id"]: r for r in load_manifest(str(env["out"] / "b" / "manifest.jsonl"))}
    assert rows1.keys() == rows2.keys()
    for rid in rows1:
        img1 = np.asarray(Image.open(rows1[rid]["image"]))
        img2 = np.asarray(Image.open(rows2[rid]["image"]))
        assert np.array_equal(img1, img2), f"{rid}: aynı seed farklı _o00 çıktısı üretti"


def test_run_rejects_copy_counts_that_overflow_two_digit_suffix(env):
    """per_image x en büyük kategori çarpanı > 99 -> AssertionError: `{ci:02d}`
    3 haneye taşar ve VAL sızıntı korumasının `_[vo]\\d{2}$` son ek deseni
    (train_colab_lib.strip_composite_copy_suffix) o id'lerle eşleşmez olurdu."""
    with pytest.raises(AssertionError, match="99"):
        # transparent x10 çarpanıyla per_image=10 -> 100 kopya (>99).
        mc.run(env["manifest"], env["backgrounds"], per_image=10, seed=42, out_dir=env["out"])


def test_run_idempotent_rerun_adds_only_missing_o00(env):
    """Var olan bir _v<NN> koşusu üzerine, sonradan _o00 eklemek için tekrar
    koşulduğunda YALNIZ eksik _o00'lar üretilir — mevcut _v<NN> çıktıları
    değişmeden kalır (bkz. modül docstring'i idempotentlik notu)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    before = {r["id"]: Path(r["image"]).read_bytes() for r in load_manifest(str(env["out"] / "manifest.jsonl"))}

    counts2 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    assert counts2 == {}  # her şey zaten vardı, yeni üretim yok

    after = load_manifest(str(env["out"] / "manifest.jsonl"))
    assert len(after) == len(before)
    for row in after:
        assert Path(row["image"]).read_bytes() == before[row["id"]]
