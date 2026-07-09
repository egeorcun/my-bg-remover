"""`training/train_colab_lib.py` için saf-Python simülasyon testleri (görev
madde 6: sampler/oversampling + resume-tespiti mantığının GPU/Colab olmadan
lokal doğrulanması). Gerçek Colab/torch/Drive ortamı gerektirmez, `slow`
değildir."""
import json

import pytest

from training.train_colab_lib import (
    compute_expected_shares,
    compute_sample_weights,
    deterministic_val_split,
    effective_lr,
    find_latest_checkpoint,
    fixed_eval_subset,
    load_stem_categories,
    prune_old_checkpoints,
    should_apply_finetune_reweight,
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
        (1, 100, 0, False),  # finetune_last_epochs=0 -> epoch<=total_epochs olduğu sürece hep False ("0 seç -> atla", config.py yorumu)
        (100, 100, 0, False),
    ],
)
def test_should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs, expected):
    assert should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs) is expected


def test_effective_lr_dis5k_vs_other_task():
    lr_dis5k = effective_lr("DIS5K", batch_size=2, accum_steps=4)
    lr_matting = effective_lr("Matting", batch_size=2, accum_steps=4)
    assert lr_dis5k == pytest.approx(1e-4 * (8 / 4) ** 0.5)
    assert lr_matting == pytest.approx(1e-5 * (8 / 4) ** 0.5)
    assert lr_dis5k == pytest.approx(lr_matting * 10)


def test_effective_lr_override_bypasses_formula():
    assert effective_lr("Matting", batch_size=2, accum_steps=4, base_lr_override=3e-5) == 3e-5
