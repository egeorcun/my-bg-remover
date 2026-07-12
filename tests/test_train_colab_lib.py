"""`training/train_colab_lib.py` için saf-Python simülasyon testleri (görev
madde 6: sampler/oversampling + resume-tespiti mantığının GPU/Colab olmadan
lokal doğrulanması). Gerçek Colab/torch/Drive ortamı gerektirmez, `slow`
değildir."""
import json
from pathlib import Path

import pytest

from training.train_colab_lib import (
    SAMPLER_PRESET_V1,
    SAMPLER_PRESET_V2,
    SAMPLER_PRESET_V3,
    SAMPLER_PRESET_V4,
    SAMPLER_PRESETS,
    apply_config_patches,
    compute_expected_shares,
    compute_sample_weights,
    copy_pairs,
    derive_val_excluded_source_ids,
    deterministic_val_split,
    effective_lr,
    ensure_manifest_pairs,
    find_latest_checkpoint,
    fixed_eval_subset,
    load_or_create_val_split,
    load_stem_categories,
    merge_composite_manifest,
    prune_old_checkpoints,
    resolve_sampler_num_samples,
    should_apply_finetune_reweight,
    strip_composite_copy_suffix,
)


def _synthetic_stems(counts: dict[str, int]) -> tuple[list[str], dict[str, str]]:
    stems: list[str] = []
    stem_category: dict[str, str] = {}
    for category, n in counts.items():
        for i in range(n):
            stem = f"{category}_{i:04d}"
            stems.append(stem)
            stem_category[stem] = category
    return stems, stem_category


# ============================================================================
# 1) Kategori ağırlıklı örnekleme
# ============================================================================
def test_compute_sample_weights_hits_target_share():
    counts = {"transparent": 50, "camouflage": 80, "hair": 9000, "general": 3000, "thin": 800, "complex": 2000}
    stems, stem_category = _synthetic_stems(counts)
    target = {"transparent": 0.20, "camouflage": 0.20}

    weights = compute_sample_weights(stems, stem_category, target)
    assert len(weights) == len(stems)

    shares = compute_expected_shares(weights, stems, stem_category)
    assert shares["transparent"] == pytest.approx(0.20, abs=1e-9)
    assert shares["camouflage"] == pytest.approx(0.20, abs=1e-9)
    # Kalan %60'lık pay, hedefsiz kategoriler arasında HAM sayılarıyla orantılı kalmalı.
    remaining = {c: shares[c] for c in ("hair", "general", "thin", "complex")}
    assert sum(remaining.values()) == pytest.approx(0.60, abs=1e-9)
    total_other = counts["hair"] + counts["general"] + counts["thin"] + counts["complex"]
    for cat in remaining:
        expected = 0.60 * counts[cat] / total_other
        assert remaining[cat] == pytest.approx(expected, abs=1e-9)


def test_compute_sample_weights_missing_target_category_is_ignored():
    # transparent hiç yoksa (bu batch'te 0 örnek), hedef payı sessizce düşürülmeli
    # (ValueError FIRLATILMAMALI) — kalan tüm pay diğer kategorilere gider.
    counts = {"camouflage": 10, "hair": 90}
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, {"transparent": 0.20, "camouflage": 0.20})
    shares = compute_expected_shares(weights, stems, stem_category)
    assert "transparent" not in shares
    assert shares["camouflage"] == pytest.approx(0.20, abs=1e-9)
    assert shares["hair"] == pytest.approx(0.80, abs=1e-9)


def test_compute_sample_weights_rejects_impossible_target():
    stems, stem_category = _synthetic_stems({"a": 5, "b": 5})
    with pytest.raises(ValueError):
        compute_sample_weights(stems, stem_category, {"a": 0.6, "b": 0.6})


# ============================================================================
# 1b) v2 sampler preset (rebalancing — v1'deki catastrophic forgetting fix)
# ============================================================================
def test_sampler_preset_v1_matches_default_target_share():
    # target_share=None (varsayılan) v1 fine-tune koşusunun (epoch_1.pth)
    # davranışıyla BİREBİR aynı olmalı — geriye dönük uyumluluk.
    counts = {"transparent": 4100, "camouflage": 8080, "hair": 9422, "complex": 2190, "thin": 810}
    stems, stem_category = _synthetic_stems(counts)
    weights_default = compute_sample_weights(stems, stem_category, None)
    weights_explicit = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V1)
    assert weights_default == weights_explicit
    assert SAMPLER_PRESET_V1 == {"transparent": 0.20, "camouflage": 0.20}


def test_sampler_presets_registry_has_v1_v2_v3_and_v4():
    assert set(SAMPLER_PRESETS) == {"v1", "v2", "v3", "v4"}
    assert SAMPLER_PRESETS["v1"] is SAMPLER_PRESET_V1
    assert SAMPLER_PRESETS["v2"] is SAMPLER_PRESET_V2
    assert SAMPLER_PRESETS["v3"] is SAMPLER_PRESET_V3
    assert SAMPLER_PRESETS["v4"] is SAMPLER_PRESET_V4
    # compute_sample_weights yalnız sum > 1.0'da ValueError fırlatır; tam 1.0'a İZİN VAR
    # (o durumda hedefsiz "_other" örneklere 0 ağırlık düşer — bkz. SAMPLER_PRESET_V2 docstring'i).
    for preset in SAMPLER_PRESETS.values():
        assert sum(preset.values()) <= 1.0 + 1e-9
    # v2 kasıtlı olarak TAM %100 dağıtır: camo 18 + transparent 20 + hair 20 +
    # complex 20 + thin 12 + general 10 (ideogram skorlaması sonrası ayar —
    # transparent en yakın kovalama hedefi olduğu için %20'de tutuldu).
    assert sum(SAMPLER_PRESET_V2.values()) == pytest.approx(1.0, abs=1e-9)
    assert SAMPLER_PRESET_V2["transparent"] == pytest.approx(0.20)
    assert SAMPLER_PRESET_V2["camouflage"] == pytest.approx(0.18)


def test_sampler_preset_v2_hits_target_shares_within_one_percent():
    # `docs/reports/2026-07-faz2-veri.md` §2'nin belgelediği ham/materyalize
    # sayılara yakın, TÜM 6 kategorinin de mevcut olduğu bir dağılım (camouflage
    # ×2, transparent ×10 fiziksel çarpanlarıyla materyalize edilmiş; general=4000
    # senaryosu — doc §2 tablosu): camouflage doğal olarak en büyük paylardan biri
    # (~%28), complex/thin ise v1'de neredeyse hiç pay alamayan küçük kategoriler
    # (bkz. v1-entegrasyon + bgr-v1-comparison raporlarındaki catastrophic
    # forgetting bulgusu). Preset toplamı tam %100 olduğundan ve tüm kategoriler
    # hedefli olduğundan gerçekleşen paylar hedeflerle BİREBİR örtüşür; tolerans
    # yine de spec'teki "within 1%" olarak bırakıldı.
    counts = {
        "camouflage": 8080,   # 4040 ham × 2
        "hair": 9422,
        "transparent": 4100,  # 410 ham × 10
        "complex": 2190,
        "thin": 810,
        "general": 4000,
    }
    stems, stem_category = _synthetic_stems(counts)
    raw_total = sum(counts.values())
    raw_share = {c: n / raw_total for c, n in counts.items()}

    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V2)
    achieved = compute_expected_shares(weights, stems, stem_category)

    for cat, target in SAMPLER_PRESET_V2.items():
        if cat not in achieved:
            continue
        assert achieved[cat] == pytest.approx(target, abs=0.01), (
            f"{cat}: hedef %{target * 100:.1f}, hesaplanan %{achieved[cat] * 100:.1f}"
        )

    # v1'in kök nedenini (complex/thin'in ham paydan çok daha düşük efektif pay
    # alması) v2'nin düzelttiğini doğrula: complex/thin artık ham paylarından
    # AÇIKÇA daha yüksek örnekleniyor, camouflage ise ham payından DÜŞÜK
    # (background: "camo downweighted from its raw ~%28-36 share").
    assert achieved["complex"] > raw_share["complex"]
    assert achieved["thin"] > raw_share["thin"]
    assert achieved["camouflage"] < raw_share["camouflage"]


def test_sampler_preset_v2_includes_general_when_present():
    counts = {
        "camouflage": 8080,
        "hair": 9422,
        "transparent": 4100,
        "complex": 2190,
        "thin": 810,
        "general": 4000,
    }
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V2)
    achieved = compute_expected_shares(weights, stems, stem_category)
    # Preset toplamı tam %100 ve tüm 6 kategori mevcut/hedefli — gerçekleşen
    # paylar hedeflerle birebir örtüşmeli, renormalizasyon yok.
    assert achieved["general"] == pytest.approx(0.10, abs=1e-9)
    assert sum(achieved.values()) == pytest.approx(1.0, abs=1e-9)


def test_sampler_preset_v2_gives_zero_weight_to_unknown_stems():
    # Preset toplamı tam 1.0 iken manifest'te kategorisi bulunamayan ("_other")
    # stem'lere SIFIR ağırlık düşmeli (hiç örneklenmezler) — bilinçli tercih,
    # bkz. SAMPLER_PRESET_V2 docstring'i. ValueError FIRLATILMAMALI (yalnız
    # sum > 1.0 hatadır).
    counts = {"camouflage": 100, "transparent": 100, "hair": 100, "complex": 100, "thin": 100, "general": 100}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["gizemli_stem_0001", "gizemli_stem_0002"]  # manifest'te yok

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V2)
    assert weights[-1] == 0.0
    assert weights[-2] == 0.0
    assert all(w > 0 for w in weights[:-2])

    achieved = compute_expected_shares(weights, stems_with_unknown, stem_category)
    assert achieved.get("_other", 0.0) == 0.0
    for cat, target in SAMPLER_PRESET_V2.items():
        assert achieved[cat] == pytest.approx(target, abs=1e-9)


# ============================================================================
# 1c) v3 sampler preset (v2 gerçek benchmark sonrası ayar — bkz. modül docstring'i)
# ============================================================================
def test_sampler_preset_v3_values_and_sum_to_one():
    assert SAMPLER_PRESET_V3 == {
        "camouflage": 0.16,
        "transparent": 0.24,
        "hair": 0.18,
        "complex": 0.20,
        "thin": 0.12,
        "general": 0.10,
    }
    assert sum(SAMPLER_PRESET_V3.values()) == pytest.approx(1.0, abs=1e-9)


def test_sampler_preset_v3_pushes_transparent_above_v2():
    # v2->v3'te transparent MAE kötüleşti (0.0437->0.0481, ideogram hedefi 0.0343) --
    # v3 transparent payını v2'nin %20'sinden (bkz. SAMPLER_PRESET_V2 kaydı, güncel
    # değer %20) %24'e YÜKSELTMELİ, en büyük tek pay olmalı.
    assert SAMPLER_PRESET_V3["transparent"] > SAMPLER_PRESET_V2["transparent"]
    assert SAMPLER_PRESET_V3["transparent"] == max(SAMPLER_PRESET_V3.values())


def test_sampler_preset_v3_hits_target_shares_within_one_percent():
    counts = {
        "camouflage": 8080,
        "hair": 9422,
        "transparent": 4100,
        "complex": 2190,
        "thin": 810,
        "general": 4000,
    }
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V3)
    achieved = compute_expected_shares(weights, stems, stem_category)
    for cat, target in SAMPLER_PRESET_V3.items():
        if cat not in achieved:
            continue
        assert achieved[cat] == pytest.approx(target, abs=0.01), (
            f"{cat}: hedef %{target * 100:.1f}, hesaplanan %{achieved[cat] * 100:.1f}"
        )


def test_sampler_preset_v3_gives_zero_weight_to_unknown_stems():
    counts = {"camouflage": 100, "transparent": 100, "hair": 100, "complex": 100, "thin": 100, "general": 100}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["gizemli_stem_0001"]  # örn. yeni bir _o00 ama manifest satırı eksik

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V3)
    assert weights[-1] == 0.0
    assert all(w > 0 for w in weights[:-1])


# ============================================================================
# 1c-2) v4 sampler preset (v3 benchmark sonrası: odak complex+thin + yeni
# yetenekler text/fx/illustration — bkz. SAMPLER_PRESET_V4 docstring'i)
# ============================================================================
def test_sampler_preset_v4_values_and_sum_to_one():
    assert SAMPLER_PRESET_V4 == {
        "camouflage": 0.12,
        "transparent": 0.18,
        "hair": 0.08,
        "complex": 0.19,
        "thin": 0.13,
        "general": 0.04,
        "text": 0.10,
        "fx": 0.08,
        "illustration": 0.08,
    }
    # toplam TAM %100 — hedefsiz "_other" stem'lere 0 ağırlık düşer
    # (bkz. SAMPLER_PRESET_V2 docstring'i, aynı bilinçli tercih).
    assert sum(SAMPLER_PRESET_V4.values()) == pytest.approx(1.0, abs=1e-9)


def test_sampler_preset_v4_uses_only_known_categories():
    # v4'ün TÜM kategorileri bilinen kümede olmalı: eski 6 kategori + v4'ün
    # üç yeni yeteneği (text/fx/illustration — v4_veri_guncelleme_hucresi.py
    # + scripts/make_textfx.py üretir). Yazım hatası (ör. "ilustration")
    # sampler'da sessizce 0 örnekli hedef olarak kaybolurdu — burada yakalanır.
    known = {
        "camouflage", "transparent", "hair", "complex", "thin", "general",
        "text", "fx", "illustration",
    }
    assert set(SAMPLER_PRESET_V4) == known


def test_sampler_preset_v4_shifts_shares_from_v3():
    # v3 benchmark sonrası yön: camo payı düşer (marj devasa: 0.0304 vs
    # Ideogram 0.1179), hair payı düşer (0.0067 MAE, rmbg 0.0045'e yakın),
    # transparent v3'ün %24'ünden iner ama korunur (%18 — kovalamaca sürüyor),
    # yeni yetenekler toplamda anlamlı pay alır.
    assert SAMPLER_PRESET_V4["camouflage"] < SAMPLER_PRESET_V3["camouflage"]
    assert SAMPLER_PRESET_V4["hair"] < SAMPLER_PRESET_V3["hair"]
    assert SAMPLER_PRESET_V4["transparent"] < SAMPLER_PRESET_V3["transparent"]
    new_share = sum(SAMPLER_PRESET_V4[c] for c in ("text", "fx", "illustration"))
    assert new_share == pytest.approx(0.26, abs=1e-9)


def test_sampler_preset_v4_hits_target_shares_within_one_percent():
    counts = {
        "camouflage": 8080,
        "hair": 9422,
        "transparent": 4100,
        "complex": 2190,
        "thin": 810,
        "general": 4000,
        "text": 4000,
        "fx": 3500,
        "illustration": 900,
    }
    stems, stem_category = _synthetic_stems(counts)
    weights = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V4)
    achieved = compute_expected_shares(weights, stems, stem_category)
    for cat, target in SAMPLER_PRESET_V4.items():
        if cat not in achieved:
            continue
        assert achieved[cat] == pytest.approx(target, abs=0.01), (
            f"{cat}: hedef %{target * 100:.1f}, hesaplanan %{achieved[cat] * 100:.1f}"
        )


def test_sampler_preset_v4_gives_zero_weight_to_unknown_stems():
    counts = {c: 100 for c in SAMPLER_PRESET_V4}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["gizemli_stem_0001"]  # örn. manifest satırı eksik yeni bir textfx stem'i

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V4)
    assert weights[-1] == 0.0
    assert all(w > 0 for w in weights[:-1])


# ============================================================================
# 1d) v3 sabit epoch uzunluğu (`resolve_sampler_num_samples`)
# ============================================================================
def test_resolve_sampler_num_samples_defaults_to_dataset_len():
    # num_samples=None -> v1/v2 davranışı BİREBİR: dataset büyüklüğü döner.
    assert resolve_sampler_num_samples(27715) == 27715
    assert resolve_sampler_num_samples(41830) == 41830


def test_resolve_sampler_num_samples_uses_fixed_value_when_given():
    # v3: dataset ~14k _o00 ile büyüse bile (41830), sabit 27715 (v2 epoch
    # parite) döner -- epoch maliyeti değişmez.
    assert resolve_sampler_num_samples(41830, num_samples=27715) == 27715
    assert resolve_sampler_num_samples(1000, num_samples=27715) == 27715  # dataset küçük olsa bile sabit değer


def test_resolve_sampler_num_samples_rejects_non_positive():
    with pytest.raises(ValueError):
        resolve_sampler_num_samples(1000, num_samples=0)
    with pytest.raises(ValueError):
        resolve_sampler_num_samples(1000, num_samples=-5)


def test_load_stem_categories(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "x_v00", "image": "im/x_v00.jpg", "category": "transparent", "gt_alpha": "gt/x_v00.png"},
        {"id": "y_v00", "image": "im/y_v00.jpg", "category": "camouflage", "gt_alpha": "gt/y_v00.png"},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = load_stem_categories(manifest)
    assert result == {"x_v00": "transparent", "y_v00": "camouflage"}


# ============================================================================
# 2) Checkpoint keşfi / budama
# ============================================================================
def test_find_latest_checkpoint_picks_max_epoch(tmp_path):
    for name in ("epoch_3.pth", "epoch_10.pth", "epoch_1.pth", "garbage.txt", "epoch_x.pth"):
        (tmp_path / name).write_text("x")
    result = find_latest_checkpoint(tmp_path)
    assert result is not None
    path, epoch = result
    assert epoch == 10
    assert path.endswith("epoch_10.pth")


def test_find_latest_checkpoint_empty_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "does-not-exist") is None


def test_prune_old_checkpoints_keeps_only_last_n(tmp_path):
    for n in (1, 2, 3, 4, 5):
        (tmp_path / f"epoch_{n}.pth").write_text("x")
    removed = prune_old_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["epoch_4.pth", "epoch_5.pth"]
    assert sorted(removed) == [str(tmp_path / "epoch_1.pth"), str(tmp_path / "epoch_2.pth"), str(tmp_path / "epoch_3.pth")]


def test_prune_old_checkpoints_noop_when_fewer_than_keep_n(tmp_path):
    (tmp_path / "epoch_1.pth").write_text("x")
    removed = prune_old_checkpoints(tmp_path, keep_last_n=5)
    assert removed == []
    assert (tmp_path / "epoch_1.pth").exists()


# ============================================================================
# 3) Deterministik TRAIN/VAL bölünmesi + sabit hızlı-değerlendirme alt kümesi
# ============================================================================
def test_deterministic_val_split_is_reproducible_and_covers_all():
    stems = [f"id_{i:05d}" for i in range(2000)]
    train_a, val_a = deterministic_val_split(stems, seed=42, val_fraction=0.02)
    train_b, val_b = deterministic_val_split(list(reversed(stems)), seed=42, val_fraction=0.02)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 40  # 2000 * 0.02
    assert set(train_a) | set(val_a) == set(stems)
    assert set(train_a) & set(val_a) == set()


def test_deterministic_val_split_different_seed_differs():
    stems = [f"id_{i:05d}" for i in range(500)]
    _, val_a = deterministic_val_split(stems, seed=1, val_fraction=0.02)
    _, val_b = deterministic_val_split(stems, seed=2, val_fraction=0.02)
    assert val_a != val_b


def test_fixed_eval_subset_deterministic_and_bounded():
    val_stems = [f"val_{i:03d}" for i in range(560)]
    a = fixed_eval_subset(val_stems, seed=7, n=24)
    b = fixed_eval_subset(val_stems, seed=7, n=24)
    assert a == b
    assert len(a) == 24
    assert set(a).issubset(set(val_stems))


def test_fixed_eval_subset_capped_by_available_size():
    val_stems = [f"val_{i:03d}" for i in range(10)]
    result = fixed_eval_subset(val_stems, seed=7, n=24)
    assert len(result) == 10


# ============================================================================
# 4) BiRefNet resmi mantık parçaları
# ============================================================================
@pytest.mark.parametrize(
    "epoch,total_epochs,finetune_last_epochs,expected",
    [
        (90, 100, -10, False),
        (91, 100, -10, True),
        (100, 100, -10, True),
        (1, 100, 0, False),  # finetune_last_epochs=0 -> "choose 0 to skip" (config.py yorumu), hep False
        (100, 100, 0, False),
        # Kısa koşu koruması: EPOCHS <= |ft| -> hile TAMAMEN atlanır (pencere epoch 1'in
        # öncesine düşer, decay üssü daha ilk epoch'ta n>1 olurdu — review Critical 1 knock-on).
        (1, 6, -10, False),
        (6, 6, -10, False),
        (10, 10, -10, False),
        # EPOCHS > |ft| -> resmi koşul aynen geçerli; üs otomatik olarak n>=1'den başlar.
        (10, 20, -10, False),
        (11, 20, -10, True),
    ],
)
def test_should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs, expected):
    assert should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs) is expected


def test_finetune_reweight_exponent_starts_at_one_when_applicable():
    # Hile uygulandığı İLK epoch'ta üs n=1 olmalı (0.9^1) — hiçbir kısayolda n>1 ile başlamamalı.
    total_epochs, ft = 20, -10
    first_applicable = next(
        e for e in range(1, total_epochs + 1) if should_apply_finetune_reweight(e, total_epochs, ft)
    )
    assert first_applicable - (total_epochs + ft) == 1


def test_effective_lr_dis5k_vs_other_task():
    lr_dis5k = effective_lr("DIS5K", batch_size=2, accum_steps=4)
    lr_matting = effective_lr("Matting", batch_size=2, accum_steps=4)
    assert lr_dis5k == pytest.approx(1e-4 * (8 / 4) ** 0.5)
    assert lr_matting == pytest.approx(1e-5 * (8 / 4) ** 0.5)
    assert lr_dis5k == pytest.approx(lr_matting * 10)


def test_effective_lr_override_bypasses_formula():
    assert effective_lr("Matting", batch_size=2, accum_steps=4, base_lr_override=3e-5) == 3e-5


# ============================================================================
# 5) config.py yaması (idempotentlik — review Critical 2)
# ============================================================================
# BiRefNet main dalındaki gerçek satırların birebir kopyası (curl ile doğrulandı).
_CONFIG_SNIPPET = """\
class Config():
    def __init__(self) -> None:
        self.batch_size = 8                                     # Multi-GPU+BF16 training...
        self.sys_home_dir = [os.path.expanduser('~'), '/workspace'][1]   # Default, custom
        self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][0]
"""


def test_apply_config_patches_basic():
    out = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    assert "self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'][5]" in out
    assert "self.sys_home_dir = [os.path.expanduser('~'), '/content/dis_data'][1]" in out
    assert "self.batch_size = 2 " in out or "self.batch_size = 2\n" in out or "self.batch_size = 2" in out


def test_apply_config_patches_is_idempotent():
    once = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    twice = apply_config_patches(once, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    assert once == twice


def test_apply_config_patches_reparameterizable_after_previous_patch():
    # Aynı VM'de kullanıcı BATCH/görev değiştirip yeniden koşarsa da çalışmalı.
    once = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    again = apply_config_patches(once, task="General", sys_home_dir="/content/other", batch_size=4)
    assert "'Matting'][3]" in again
    assert "'/content/other'" in again
    assert "self.batch_size = 4" in again


def test_apply_config_patches_raises_on_unknown_source():
    with pytest.raises(ValueError):
        apply_config_patches("class Config: pass", task="Matting", sys_home_dir="/x", batch_size=2)
    with pytest.raises(ValueError):
        apply_config_patches(_CONFIG_SNIPPET, task="YokBoyleGorev", sys_home_dir="/x", batch_size=2)


# ============================================================================
# 6) copy_pairs (boyut doğrulamalı kopyalama — review Important 3)
# ============================================================================
def _make_pair_tree(tmp_path, stems, im_content=b"IMDATA-123", gt_content=b"GTDATA-456"):
    src_im, src_gt = tmp_path / "src_im", tmp_path / "src_gt"
    dst_im, dst_gt = tmp_path / "dst_im", tmp_path / "dst_gt"
    for d in (src_im, src_gt, dst_im, dst_gt):
        d.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (src_im / f"{stem}.jpg").write_bytes(im_content)
        (src_gt / f"{stem}.png").write_bytes(gt_content)
    return src_im, src_gt, dst_im, dst_gt


def test_copy_pairs_copies_and_is_idempotent(tmp_path):
    stems = ["a", "b", "c"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 3
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 0  # ikinci koşu no-op
    for stem in stems:
        assert (dst_im / f"{stem}.jpg").read_bytes() == b"IMDATA-123"
        assert (dst_gt / f"{stem}.png").read_bytes() == b"GTDATA-456"


def test_copy_pairs_repairs_truncated_gt(tmp_path):
    # im tam ama gt kesik (yarım kalmış Drive kopyası) -> çift YENİDEN kopyalanmalı.
    stems = ["a"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    (dst_gt / "a.png").write_bytes(b"GT")  # kesik gt simülasyonu (im boyutu hâlâ doğru)
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 1
    assert (dst_gt / "a.png").read_bytes() == b"GTDATA-456"


def test_copy_pairs_repairs_truncated_im(tmp_path):
    stems = ["a"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    (dst_im / "a.jpg").write_bytes(b"IM")
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 1
    assert (dst_im / "a.jpg").read_bytes() == b"IMDATA-123"


def test_copy_pairs_parallel_matches_serial(tmp_path):
    # Paralel (varsayılan max_workers=16) sonucun, tek-iş-parçacıklı (max_workers=1)
    # koşumla BİREBİR aynı dizin ağacını ürettiğini kanıtlar — aynı fixture (200 çift,
    # her biri kendine özgü içerik), iki ayrı hedef ağaca kopyalanır, sonra karşılaştırılır.
    stems = [f"stem_{i:04d}" for i in range(200)]
    src_im, src_gt = tmp_path / "src_im", tmp_path / "src_gt"
    src_im.mkdir()
    src_gt.mkdir()
    for stem in stems:
        (src_im / f"{stem}.jpg").write_bytes(f"IMG-DATA-{stem}".encode())
        (src_gt / f"{stem}.png").write_bytes(f"GT-DATA-{stem}".encode())

    dst_im_serial, dst_gt_serial = tmp_path / "dst_im_serial", tmp_path / "dst_gt_serial"
    dst_im_parallel, dst_gt_parallel = tmp_path / "dst_im_parallel", tmp_path / "dst_gt_parallel"
    for d in (dst_im_serial, dst_gt_serial, dst_im_parallel, dst_gt_parallel):
        d.mkdir()

    n_serial = copy_pairs(stems, src_im, src_gt, dst_im_serial, dst_gt_serial, max_workers=1)
    n_parallel = copy_pairs(stems, src_im, src_gt, dst_im_parallel, dst_gt_parallel, max_workers=16)
    assert n_serial == n_parallel == len(stems)

    def _tree(d):
        return {p.name: p.read_bytes() for p in d.iterdir()}

    assert _tree(dst_im_serial) == _tree(dst_im_parallel)
    assert _tree(dst_gt_serial) == _tree(dst_gt_parallel)

    # İkinci koşum (idempotentlik) her iki modda da no-op olmalı.
    assert copy_pairs(stems, src_im, src_gt, dst_im_serial, dst_gt_serial, max_workers=1) == 0
    assert copy_pairs(stems, src_im, src_gt, dst_im_parallel, dst_gt_parallel, max_workers=16) == 0


def test_copy_pairs_collects_errors_and_raises_first_with_count(tmp_path):
    # Kaynakta OLMAYAN bir stem varsa o çiftin kopyalanması hata verir; ama diğer
    # TÜM çiftler yine de işlenmeli (kısmi ilerleme kaybolmamalı) ve sonda İLK hata
    # toplam hata sayısıyla birlikte fırlatılmalı.
    stems = ["a", "missing", "b"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, ["a", "b"])
    with pytest.raises(RuntimeError, match=r"1/3.*missing"):
        copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    # "a" ve "b" hatasız çiftler olduğu için yine de kopyalanmış olmalı.
    assert (dst_im / "a.jpg").exists()
    assert (dst_im / "b.jpg").exists()


# ============================================================================
# 7) Kalıcı VAL bölünmesi (review Important 2)
# ============================================================================
def test_load_or_create_val_split_first_run_persists(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    train, val = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)
    assert persist.exists()
    saved = json.loads(persist.read_text())
    assert saved["val_stems"] == val
    assert len(val) == 20
    assert set(train) | set(val) == set(stems)


def test_load_or_create_val_split_loads_existing_and_keeps_new_stems_in_train(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    _, val_first = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)

    # Veri seti BÜYÜDÜ (Faz 2 yeniden koştu, 200 yeni çift) — val kümesi DEĞİŞMEMELİ,
    # yeni stem'lerin tamamı train'e gitmeli (belgelenmiş tercih, sızıntı yok).
    grown = stems + [f"new_{i:05d}" for i in range(200)]
    train2, val2 = load_or_create_val_split(grown, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first
    assert all(s in train2 for s in (f"new_{i:05d}" for i in range(200)))
    assert set(train2) & set(val2) == set()


def test_load_or_create_val_split_drops_vanished_val_stems(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    _, val_first = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)
    shrunk = [s for s in stems if s != val_first[0]]  # bir val görseli diskten silindi
    _, val2 = load_or_create_val_split(shrunk, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first[1:]


# ============================================================================
# 7) v3 — VAL sızıntı hariç tutma + kompozit manifest merge
#    (bkz. training/v3_veri_guncelleme_hucresi.py)
# ============================================================================
def test_strip_composite_copy_suffix_strips_v_and_o_suffixes():
    assert strip_composite_copy_suffix("camo_00365_v03") == "camo_00365"
    assert strip_composite_copy_suffix("trans1_o00") == "trans1"
    assert strip_composite_copy_suffix("hair_0042_v00") == "hair_0042"


def test_strip_composite_copy_suffix_leaves_unmatched_stems_unchanged():
    # Eşleşmeyen bir stem (son eksiz kaynak id, ya da 3 haneli indeks) OLDUĞU
    # GİBİ döner — bu SIZINTI RİSKİDİR (docstring: son ekli hali hariç-tutma
    # kümesine girer, hiçbir kaynak id ile eşleşmez, koruma o kaynak için
    # BAYPAS olur); derive_val_excluded_source_ids bu durumu ayrıca raporlar.
    assert strip_composite_copy_suffix("bare_source_id") == "bare_source_id"
    assert strip_composite_copy_suffix("id_v100") == "id_v100"  # 3 haneli indeks desene UYMAZ


def test_derive_val_excluded_source_ids_from_val_stems_list():
    val_stems = ["camo_00365_v03", "trans1_o00", "hair_0042_v00", "hair_0042_v01"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    # aynı kaynağın birden çok kopyası (hair_0042_v00/_v01) TEK bir kaynak id'e düşer.
    assert excluded == {"camo_00365", "trans1", "hair_0042"}
    assert unmatched == []


def test_derive_val_excluded_source_ids_reports_unmatched_stems():
    # Son ek deseniyle eşleşmeyen stem'ler koruma-baypas TEŞHİSİ için ayrıca
    # döndürülmeli — v3 hücresi (stage_composites_o) boş-olmayan listede yüksek
    # sesli uyarı basar (reviewer bulgusu #3).
    val_stems = ["a_v00", "garip_stem", "b_o00", "id_v100"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    assert unmatched == ["garip_stem", "id_v100"]
    assert {"a", "b"} <= excluded
    # eşleşmeyenler kümeye SON EKLİ/yanlış haliyle girer (dokümante davranış) —
    # kaynak manifest'te bu id'ler bulunmayacağından koruma onlar için baypas.
    assert "garip_stem" in excluded
    assert "id_v100" in excluded


def test_derive_val_excluded_source_ids_empty_list():
    assert derive_val_excluded_source_ids([]) == (set(), [])


def test_merge_composite_manifest_appends_only_new_ids(tmp_path):
    local = tmp_path / "local_o00_manifest.jsonl"
    drive = tmp_path / "drive_composites_manifest.jsonl"

    # Drive'da ZATEN v1/v2'nin _v<NN> satırları var.
    drive_rows = [
        {"id": "a_v00", "image": "im/a_v00.jpg", "category": "transparent", "gt_alpha": "gt/a_v00.png"},
    ]
    drive.write_text("\n".join(json.dumps(r) for r in drive_rows) + "\n")

    # Yerelde yalnız yeni _o00 satırları var.
    local_rows = [
        {"id": "a_o00", "image": "im/a_o00.jpg", "category": "transparent", "gt_alpha": "gt/a_o00.png"},
        {"id": "b_o00", "image": "im/b_o00.jpg", "category": "hair", "gt_alpha": "gt/b_o00.png"},
    ]
    local.write_text("\n".join(json.dumps(r) for r in local_rows) + "\n")

    n_added = merge_composite_manifest(local, drive)
    assert n_added == 2

    merged_ids = [json.loads(line)["id"] for line in drive.read_text().splitlines() if line.strip()]
    assert merged_ids == ["a_v00", "a_o00", "b_o00"]  # eski satırlar KORUNDU, yeni satırlar EKLENDİ


def test_merge_composite_manifest_idempotent_second_call_adds_nothing(tmp_path):
    local = tmp_path / "local_o00_manifest.jsonl"
    drive = tmp_path / "drive_composites_manifest.jsonl"
    local_rows = [{"id": "a_o00", "image": "im/a_o00.jpg", "category": "transparent", "gt_alpha": "gt/a_o00.png"}]
    local.write_text("\n".join(json.dumps(r) for r in local_rows) + "\n")

    n1 = merge_composite_manifest(local, drive)
    n2 = merge_composite_manifest(local, drive)
    assert n1 == 1
    assert n2 == 0
    merged_ids = [json.loads(line)["id"] for line in drive.read_text().splitlines() if line.strip()]
    assert merged_ids == ["a_o00"]  # tekrar yok


def test_merge_composite_manifest_missing_local_returns_zero(tmp_path):
    local = tmp_path / "does_not_exist.jsonl"
    drive = tmp_path / "drive_composites_manifest.jsonl"
    assert merge_composite_manifest(local, drive) == 0
    assert not drive.exists()


# ============================================================================
# 7c) boş-manifest guard'ı (ensure_manifest_pairs) — canlı v3 koşusu dersi:
#     ham veri inmemişken manifest 0 çiftle kuruldu, hata ancak export'ta
#     (SEMPTOM olarak) göründü; guard NEDENİ manifest kurulumunda yakalar.
# ============================================================================
def test_ensure_manifest_pairs_returns_count_when_nonempty(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "a", "image": "im/a.jpg", "category": "transparent", "gt_alpha": "gt/a.png"},
        {"id": "b", "image": "im/b.jpg", "category": "hair", "gt_alpha": "gt/b.png"},
        {"id": "c", "image": "im/c.jpg", "category": "product", "gt_alpha": None},  # GT'siz -> sayılmaz
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert ensure_manifest_pairs(manifest) == 2


def test_ensure_manifest_pairs_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="manifest dosyası yok"):
        ensure_manifest_pairs(tmp_path / "yok.jsonl")


def test_ensure_manifest_pairs_raises_on_empty_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")  # 0 satır — canlı koşudaki durum
    with pytest.raises(RuntimeError, match="GEÇİLMEYECEK"):
        ensure_manifest_pairs(manifest)


def test_ensure_manifest_pairs_raises_when_all_rows_lack_gt(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"id": "a", "image": "im/a.jpg", "category": "product", "gt_alpha": None}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="0 GT'li çift"):
        ensure_manifest_pairs(manifest)


# ============================================================================
# 7b) _o00 uçtan uca simülasyon: küçük bir fixture üzerinde make_composites.run()
#     -> exclude_source_ids (val_stems.json'dan türetilen) -> merge_composite_
#     manifest ile Drive manifestine merge -- v3_veri_guncelleme_hucresi.py'nin
#     composites_o + drive_copy aşamalarının hermetik simülasyonu.
# ============================================================================
def test_o00_end_to_end_simulation_with_val_exclusion_and_drive_merge(tmp_path):
    import sys

    scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import make_composites as mc
    from benchmark.testset import append_entries
    from PIL import Image

    src_dir = tmp_path / "src"
    bg_dir = tmp_path / "backgrounds"
    src_dir.mkdir()
    bg_dir.mkdir()
    Image.new("RGB", (20, 20), (255, 0, 255)).save(bg_dir / "bg0.jpg")

    source_manifest = tmp_path / "train_manifest.jsonl"
    rows = []
    for name, category in (("a", "transparent"), ("b", "hair"), ("c", "transparent")):
        Image.new("RGB", (16, 16), (0, 200, 0)).save(src_dir / f"{name}.jpg")
        Image.new("L", (16, 16), 255).save(src_dir / f"{name}_gt.png")
        rows.append({
            "id": name, "image": str(src_dir / f"{name}.jpg"), "category": category,
            "gt_alpha": str(src_dir / f"{name}_gt.png"),
        })
    append_entries(str(source_manifest), rows)

    # val_stems.json: kaynak "a"nın BİR _v kopyası VAL'e düşmüş -- "a" tamamen
    # hariç tutulmalı (make_composites hâlâ _v'ler için "a"yı işler, ama _o00
    # üretiminden dışlanır).
    # Boş-manifest guard'ı (v3 hücresinin "manifest" aşaması sonu — canlı koşu
    # dersi): dolu kaynak manifest'te guard GEÇER ve GT'li çift sayısını döner;
    # boş/eksik manifest'te (ham veri inmemiş senaryosu) RuntimeError fırlatıp
    # composites_o/export'a GEÇİLMESİNİ engeller.
    assert ensure_manifest_pairs(source_manifest) == 3
    empty_manifest = tmp_path / "empty_manifest.jsonl"
    empty_manifest.write_text("")
    with pytest.raises(RuntimeError, match="GEÇİLMEYECEK"):
        ensure_manifest_pairs(empty_manifest)
    with pytest.raises(RuntimeError, match="manifest dosyası yok"):
        ensure_manifest_pairs(tmp_path / "hic_kurulmadi.jsonl")

    val_stems = ["a_v03"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    assert excluded == {"a"}
    assert unmatched == []

    out_dir = tmp_path / "composites_o"
    counts = mc.run(
        source_manifest, bg_dir, per_image=1, seed=42, out_dir=out_dir,
        exclude_source_ids=excluded, only_original_bg=True,
    )
    # yalnız b ve c için _o00 üretildi (a hariç tutuldu); toplam = eligible x ORIGINAL_BG_COPIES.
    assert sum(counts.values()) == 2 * mc.ORIGINAL_BG_COPIES
    from benchmark.testset import load_manifest
    o00_ids = {r["id"] for r in load_manifest(str(out_dir / "manifest.jsonl"))}
    assert o00_ids == {"b_o00", "c_o00"}

    # Drive tarafı: v1/v2'nin _v<NN> satırlarını zaten içeren bir manifest'e merge.
    drive_manifest = tmp_path / "drive_train_composites_manifest.jsonl"
    drive_manifest.write_text(json.dumps(
        {"id": "a_v00", "image": "im/a_v00.jpg", "category": "transparent", "gt_alpha": "gt/a_v00.png"}
    ) + "\n")
    n_added = merge_composite_manifest(out_dir / "manifest.jsonl", drive_manifest)
    assert n_added == 2  # yalnız yeni _o00 satırları eklendi

    final_ids = [json.loads(line)["id"] for line in drive_manifest.read_text().splitlines() if line.strip()]
    assert set(final_ids) == {"a_v00", "b_o00", "c_o00"}

    # idempotentlik: aynı merge tekrar çağrılırsa 0 satır eklenir.
    assert merge_composite_manifest(out_dir / "manifest.jsonl", drive_manifest) == 0
