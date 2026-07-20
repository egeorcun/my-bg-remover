"""Pure-Python simulation tests for `training/train_colab_lib.py` (task item
6: local validation of the sampler/oversampling + resume-detection logic
without GPU/Colab). Requires no real Colab/torch/Drive environment, and is not
`slow`."""
import ast
import json
from pathlib import Path

import pytest

from training.train_colab_lib import (
    SAMPLER_PRESET_V1,
    SAMPLER_PRESET_V2,
    SAMPLER_PRESET_V3,
    SAMPLER_PRESET_V4,
    SAMPLER_PRESETS,
    SAMPLER_PRESET_V5,
    SAMPLER_PRESET_V7,
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
    split_stems_to_shards,
    strip_composite_copy_suffix,
    tar_shard_name,
    validate_tar_manifest,
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
# 1) Category-weighted sampling
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
    # The remaining 60% share must stay proportional to the RAW counts of the untargeted categories.
    remaining = {c: shares[c] for c in ("hair", "general", "thin", "complex")}
    assert sum(remaining.values()) == pytest.approx(0.60, abs=1e-9)
    total_other = counts["hair"] + counts["general"] + counts["thin"] + counts["complex"]
    for cat in remaining:
        expected = 0.60 * counts[cat] / total_other
        assert remaining[cat] == pytest.approx(expected, abs=1e-9)


def test_compute_sample_weights_missing_target_category_is_ignored():
    # If transparent is entirely absent (0 samples in this batch), its target share
    # must be dropped silently (NO ValueError) — all the remaining share goes to
    # the other categories.
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
# 1b) v2 sampler preset (rebalancing — fix for the catastrophic forgetting in v1)
# ============================================================================
def test_sampler_preset_v1_matches_default_target_share():
    # target_share=None (default) must be IDENTICAL to the behavior of the v1
    # fine-tune run (epoch_1.pth) — backward compatibility.
    counts = {"transparent": 4100, "camouflage": 8080, "hair": 9422, "complex": 2190, "thin": 810}
    stems, stem_category = _synthetic_stems(counts)
    weights_default = compute_sample_weights(stems, stem_category, None)
    weights_explicit = compute_sample_weights(stems, stem_category, SAMPLER_PRESET_V1)
    assert weights_default == weights_explicit
    assert SAMPLER_PRESET_V1 == {"transparent": 0.20, "camouflage": 0.20}


def test_sampler_presets_registry_has_v1_v2_v3_and_v4():
    assert set(SAMPLER_PRESETS) == {"v1", "v2", "v3", "v4", "v5", "v7"}
    assert SAMPLER_PRESETS["v1"] is SAMPLER_PRESET_V1
    assert SAMPLER_PRESETS["v2"] is SAMPLER_PRESET_V2
    assert SAMPLER_PRESETS["v3"] is SAMPLER_PRESET_V3
    assert SAMPLER_PRESETS["v4"] is SAMPLER_PRESET_V4
    assert SAMPLER_PRESETS["v5"] is SAMPLER_PRESET_V5
    # compute_sample_weights only raises ValueError on sum > 1.0; exactly 1.0 IS allowed
    # (in that case untargeted "_other" samples get 0 weight — see the SAMPLER_PRESET_V2 docstring).
    for preset in SAMPLER_PRESETS.values():
        assert sum(preset.values()) <= 1.0 + 1e-9
    # v2 deliberately distributes EXACTLY 100%: camo 18 + transparent 20 + hair 20 +
    # complex 20 + thin 12 + general 10 (adjustment after the ideogram scoring —
    # transparent was kept at 20% because it is the closest chase target).
    assert sum(SAMPLER_PRESET_V2.values()) == pytest.approx(1.0, abs=1e-9)
    assert SAMPLER_PRESET_V2["transparent"] == pytest.approx(0.20)
    assert SAMPLER_PRESET_V2["camouflage"] == pytest.approx(0.18)


def test_sampler_preset_v2_hits_target_shares_within_one_percent():
    # A distribution close to the raw/materialized counts documented in
    # the project's internal phase report (removed from the repo) §2, with ALL 6 categories present
    # (materialized with the physical camouflage x2, transparent x10 multipliers;
    # the general=4000 scenario — doc §2 table): camouflage is naturally one of the
    # largest shares (~28%), while complex/thin are the small categories that got
    # almost no share in v1 (see the catastrophic forgetting finding in the
    # v1-integration + bgr-v1-comparison reports). Since the preset sums to exactly
    # 100% and all categories are targeted, the achieved shares match the targets
    # EXACTLY; the tolerance is nevertheless left at the spec's "within 1%".
    counts = {
        "camouflage": 8080,   # 4040 raw x 2
        "hair": 9422,
        "transparent": 4100,  # 410 raw x 10
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
            f"{cat}: target {target * 100:.1f}%, computed {achieved[cat] * 100:.1f}%"
        )

    # Verify that v2 fixes v1's root cause (complex/thin getting a far lower
    # effective share than their raw share): complex/thin are now sampled CLEARLY
    # above their raw shares, while camouflage is BELOW its raw share
    # (background: "camo downweighted from its raw ~28-36% share").
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
    # The preset sums to exactly 100% and all 6 categories are present/targeted —
    # the achieved shares must match the targets exactly, no renormalization.
    assert achieved["general"] == pytest.approx(0.10, abs=1e-9)
    assert sum(achieved.values()) == pytest.approx(1.0, abs=1e-9)


def test_sampler_preset_v2_gives_zero_weight_to_unknown_stems():
    # When the preset sums to exactly 1.0, stems whose category is missing from
    # the manifest ("_other") must get ZERO weight (never sampled) — a deliberate
    # choice, see the SAMPLER_PRESET_V2 docstring. NO ValueError must be raised
    # (only sum > 1.0 is an error).
    counts = {"camouflage": 100, "transparent": 100, "hair": 100, "complex": 100, "thin": 100, "general": 100}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["mystery_stem_0001", "mystery_stem_0002"]  # not in the manifest

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V2)
    assert weights[-1] == 0.0
    assert weights[-2] == 0.0
    assert all(w > 0 for w in weights[:-2])

    achieved = compute_expected_shares(weights, stems_with_unknown, stem_category)
    assert achieved.get("_other", 0.0) == 0.0
    for cat, target in SAMPLER_PRESET_V2.items():
        assert achieved[cat] == pytest.approx(target, abs=1e-9)


# ============================================================================
# 1c) v3 sampler preset (adjustment after v2's real benchmark — see the module docstring)
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
    # transparent MAE got worse from v2 to v3 (0.0437->0.0481, ideogram target 0.0343) --
    # v3 must RAISE the transparent share from v2's 20% (see the SAMPLER_PRESET_V2
    # record, current value 20%) to 24%, and it must be the single largest share.
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
            f"{cat}: target {target * 100:.1f}%, computed {achieved[cat] * 100:.1f}%"
        )


def test_sampler_preset_v3_gives_zero_weight_to_unknown_stems():
    counts = {"camouflage": 100, "transparent": 100, "hair": 100, "complex": 100, "thin": 100, "general": 100}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["mystery_stem_0001"]  # e.g. a new _o00 whose manifest row is missing

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V3)
    assert weights[-1] == 0.0
    assert all(w > 0 for w in weights[:-1])


# ============================================================================
# 1c-2) v4 sampler preset (after the v3 benchmark: focus on complex+thin + the
# new capabilities text/fx/illustration — see the SAMPLER_PRESET_V4 docstring)
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
    # sums to EXACTLY 100% — untargeted "_other" stems get 0 weight
    # (see the SAMPLER_PRESET_V2 docstring, same deliberate choice).
    assert sum(SAMPLER_PRESET_V4.values()) == pytest.approx(1.0, abs=1e-9)


def test_sampler_preset_v4_uses_only_known_categories():
    # ALL of v4's categories must be in the known set: the old 6 categories plus
    # v4's three new capabilities (text/fx/illustration — produced by
    # v4_veri_guncelleme_hucresi.py + scripts/make_textfx.py). A typo
    # (e.g. "ilustration") would silently vanish in the sampler as a target with
    # 0 samples — it gets caught here.
    known = {
        "camouflage", "transparent", "hair", "complex", "thin", "general",
        "text", "fx", "illustration",
    }
    assert set(SAMPLER_PRESET_V4) == known


def test_sampler_preset_v4_shifts_shares_from_v3():
    # Direction after the v3 benchmark: the camo share drops (the margin is
    # enormous: 0.0304 vs Ideogram 0.1179), the hair share drops (0.0067 MAE,
    # close to rmbg's 0.0045), transparent comes down from v3's 24% but is
    # protected (18% — the chase continues), and the new capabilities get a
    # meaningful combined share.
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
            f"{cat}: target {target * 100:.1f}%, computed {achieved[cat] * 100:.1f}%"
        )


def test_sampler_preset_v4_gives_zero_weight_to_unknown_stems():
    counts = {c: 100 for c in SAMPLER_PRESET_V4}
    stems, stem_category = _synthetic_stems(counts)
    stems_with_unknown = stems + ["mystery_stem_0001"]  # e.g. a new textfx stem whose manifest row is missing

    weights = compute_sample_weights(stems_with_unknown, stem_category, SAMPLER_PRESET_V4)
    assert weights[-1] == 0.0
    assert all(w > 0 for w in weights[:-1])


# ============================================================================
# 1d) v3 fixed epoch length (`resolve_sampler_num_samples`)
# ============================================================================
def test_resolve_sampler_num_samples_defaults_to_dataset_len():
    # num_samples=None -> IDENTICAL to the v1/v2 behavior: returns the dataset size.
    assert resolve_sampler_num_samples(27715) == 27715
    assert resolve_sampler_num_samples(41830) == 41830


def test_resolve_sampler_num_samples_uses_fixed_value_when_given():
    # v3: even if the dataset grows by ~14k _o00 (41830), the fixed 27715 (v2 epoch
    # parity) is returned -- the epoch cost does not change.
    assert resolve_sampler_num_samples(41830, num_samples=27715) == 27715
    assert resolve_sampler_num_samples(1000, num_samples=27715) == 27715  # fixed value even if the dataset is small


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
# 2) Checkpoint discovery / pruning
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
# 3) Deterministic TRAIN/VAL split + fixed quick-evaluation subset
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
# 4) Pieces of the official BiRefNet logic
# ============================================================================
@pytest.mark.parametrize(
    "epoch,total_epochs,finetune_last_epochs,expected",
    [
        (90, 100, -10, False),
        (91, 100, -10, True),
        (100, 100, -10, True),
        (1, 100, 0, False),  # finetune_last_epochs=0 -> "choose 0 to skip" (config.py comment), always False
        (100, 100, 0, False),
        # Short-run guard: EPOCHS <= |ft| -> the trick is skipped ENTIRELY (the window
        # would start before epoch 1 and the decay exponent would be n>1 at the very
        # first epoch — review Critical 1 knock-on).
        (1, 6, -10, False),
        (6, 6, -10, False),
        (10, 10, -10, False),
        # EPOCHS > |ft| -> the official condition applies as-is; the exponent automatically starts at n>=1.
        (10, 20, -10, False),
        (11, 20, -10, True),
    ],
)
def test_should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs, expected):
    assert should_apply_finetune_reweight(epoch, total_epochs, finetune_last_epochs) is expected


def test_finetune_reweight_exponent_starts_at_one_when_applicable():
    # At the FIRST epoch where the trick applies, the exponent must be n=1 (0.9^1) —
    # no code path may start with n>1.
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
# 5) config.py patching (idempotency — review Critical 2)
# ============================================================================
# Verbatim copy of the real lines on the BiRefNet main branch (verified with curl).
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
    # Must also work if the user changes BATCH/task on the same VM and re-runs.
    once = apply_config_patches(_CONFIG_SNIPPET, task="Matting", sys_home_dir="/content/dis_data", batch_size=2)
    again = apply_config_patches(once, task="General", sys_home_dir="/content/other", batch_size=4)
    assert "'Matting'][3]" in again
    assert "'/content/other'" in again
    assert "self.batch_size = 4" in again


def test_apply_config_patches_raises_on_unknown_source():
    with pytest.raises(ValueError):
        apply_config_patches("class Config: pass", task="Matting", sys_home_dir="/x", batch_size=2)
    with pytest.raises(ValueError):
        apply_config_patches(_CONFIG_SNIPPET, task="NoSuchTask", sys_home_dir="/x", batch_size=2)


# ============================================================================
# 6) copy_pairs (size-validated copying — review Important 3)
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
    assert copy_pairs(stems, src_im, src_gt, dst_im, dst_gt) == 0  # second run is a no-op
    for stem in stems:
        assert (dst_im / f"{stem}.jpg").read_bytes() == b"IMDATA-123"
        assert (dst_gt / f"{stem}.png").read_bytes() == b"GTDATA-456"


def test_copy_pairs_repairs_truncated_gt(tmp_path):
    # im is intact but gt is truncated (a half-finished Drive copy) -> the pair must be RE-copied.
    stems = ["a"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, stems)
    copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    (dst_gt / "a.png").write_bytes(b"GT")  # truncated gt simulation (im size still correct)
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
    # Proves that the parallel run (default max_workers=16) produces a directory tree
    # IDENTICAL to a single-threaded run (max_workers=1) — the same fixture (200 pairs,
    # each with unique content) is copied into two separate destination trees, then compared.
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

    # A second run (idempotency) must be a no-op in both modes.
    assert copy_pairs(stems, src_im, src_gt, dst_im_serial, dst_gt_serial, max_workers=1) == 0
    assert copy_pairs(stems, src_im, src_gt, dst_im_parallel, dst_gt_parallel, max_workers=16) == 0


def test_copy_pairs_collects_errors_and_raises_first_with_count(tmp_path):
    # If a stem is MISSING at the source, copying that pair fails; but ALL other
    # pairs must still be processed (partial progress must not be lost) and at the
    # end the FIRST error must be raised together with the total error count.
    stems = ["a", "missing", "b"]
    src_im, src_gt, dst_im, dst_gt = _make_pair_tree(tmp_path, ["a", "b"])
    with pytest.raises(RuntimeError, match=r"1/3.*missing"):
        copy_pairs(stems, src_im, src_gt, dst_im, dst_gt)
    # "a" and "b" are error-free pairs, so they must still have been copied.
    assert (dst_im / "a.jpg").exists()
    assert (dst_im / "b.jpg").exists()


# ============================================================================
# 7) Persistent VAL split (review Important 2)
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

    # The dataset GREW (Phase 2 re-ran, 200 new pairs) — the val set must NOT change,
    # all new stems must go to train (documented choice, no leak).
    grown = stems + [f"new_{i:05d}" for i in range(200)]
    train2, val2 = load_or_create_val_split(grown, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first
    assert all(s in train2 for s in (f"new_{i:05d}" for i in range(200)))
    assert set(train2) & set(val2) == set()


def test_load_or_create_val_split_drops_vanished_val_stems(tmp_path):
    stems = [f"id_{i:05d}" for i in range(1000)]
    persist = tmp_path / "val_stems.json"
    _, val_first = load_or_create_val_split(stems, seed=42, val_fraction=0.02, persist_path=persist)
    shrunk = [s for s in stems if s != val_first[0]]  # one val image was deleted from disk
    _, val2 = load_or_create_val_split(shrunk, seed=42, val_fraction=0.02, persist_path=persist)
    assert val2 == val_first[1:]


# ============================================================================
# 7) v3 — VAL leak exclusion + composite manifest merge
#    (see training/v3_veri_guncelleme_hucresi.py)
# ============================================================================
def test_strip_composite_copy_suffix_strips_v_and_o_suffixes():
    assert strip_composite_copy_suffix("camo_00365_v03") == "camo_00365"
    assert strip_composite_copy_suffix("trans1_o00") == "trans1"
    assert strip_composite_copy_suffix("hair_0042_v00") == "hair_0042"


def test_strip_composite_copy_suffix_leaves_unmatched_stems_unchanged():
    # A non-matching stem (a suffix-less source id, or a 3-digit index) is returned
    # AS IS — this is a LEAK RISK (docstring: the suffixed form enters the exclusion
    # set, matches no source id, and the guard is BYPASSED for that source);
    # derive_val_excluded_source_ids reports this case separately.
    assert strip_composite_copy_suffix("bare_source_id") == "bare_source_id"
    assert strip_composite_copy_suffix("id_v100") == "id_v100"  # a 3-digit index does NOT match the pattern


def test_derive_val_excluded_source_ids_from_val_stems_list():
    val_stems = ["camo_00365_v03", "trans1_o00", "hair_0042_v00", "hair_0042_v01"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    # multiple copies of the same source (hair_0042_v00/_v01) collapse to a SINGLE source id.
    assert excluded == {"camo_00365", "trans1", "hair_0042"}
    assert unmatched == []


def test_derive_val_excluded_source_ids_reports_unmatched_stems():
    # Stems that do not match the suffix pattern must also be returned for the
    # guard-bypass DIAGNOSIS — the v3 cell (stage_composites_o) prints a loud
    # warning on a non-empty list (reviewer finding #3).
    val_stems = ["a_v00", "weird_stem", "b_o00", "id_v100"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    assert unmatched == ["weird_stem", "id_v100"]
    assert {"a", "b"} <= excluded
    # non-matching stems enter the set in their SUFFIXED/wrong form (documented
    # behavior) — since these ids do not exist in the source manifest, the guard
    # is bypassed for them.
    assert "weird_stem" in excluded
    assert "id_v100" in excluded


def test_derive_val_excluded_source_ids_empty_list():
    assert derive_val_excluded_source_ids([]) == (set(), [])


def test_merge_composite_manifest_appends_only_new_ids(tmp_path):
    local = tmp_path / "local_o00_manifest.jsonl"
    drive = tmp_path / "drive_composites_manifest.jsonl"

    # Drive ALREADY contains v1/v2's _v<NN> rows.
    drive_rows = [
        {"id": "a_v00", "image": "im/a_v00.jpg", "category": "transparent", "gt_alpha": "gt/a_v00.png"},
    ]
    drive.write_text("\n".join(json.dumps(r) for r in drive_rows) + "\n")

    # Locally there are only the new _o00 rows.
    local_rows = [
        {"id": "a_o00", "image": "im/a_o00.jpg", "category": "transparent", "gt_alpha": "gt/a_o00.png"},
        {"id": "b_o00", "image": "im/b_o00.jpg", "category": "hair", "gt_alpha": "gt/b_o00.png"},
    ]
    local.write_text("\n".join(json.dumps(r) for r in local_rows) + "\n")

    n_added = merge_composite_manifest(local, drive)
    assert n_added == 2

    merged_ids = [json.loads(line)["id"] for line in drive.read_text().splitlines() if line.strip()]
    assert merged_ids == ["a_v00", "a_o00", "b_o00"]  # old rows PRESERVED, new rows APPENDED


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
    assert merged_ids == ["a_o00"]  # no duplicates


def test_merge_composite_manifest_missing_local_returns_zero(tmp_path):
    local = tmp_path / "does_not_exist.jsonl"
    drive = tmp_path / "drive_composites_manifest.jsonl"
    assert merge_composite_manifest(local, drive) == 0
    assert not drive.exists()


# ============================================================================
# 7c) empty-manifest guard (ensure_manifest_pairs) — lesson from the live v3
#     run: the manifest was built with 0 pairs while the raw data had never
#     downloaded, and the failure only surfaced at export (as a SYMPTOM);
#     the guard catches the CAUSE at manifest setup.
# ============================================================================
def test_ensure_manifest_pairs_returns_count_when_nonempty(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "a", "image": "im/a.jpg", "category": "transparent", "gt_alpha": "gt/a.png"},
        {"id": "b", "image": "im/b.jpg", "category": "hair", "gt_alpha": "gt/b.png"},
        {"id": "c", "image": "im/c.jpg", "category": "product", "gt_alpha": None},  # no GT -> not counted
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert ensure_manifest_pairs(manifest) == 2


def test_ensure_manifest_pairs_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="manifest file missing"):
        ensure_manifest_pairs(tmp_path / "missing.jsonl")


def test_ensure_manifest_pairs_raises_on_empty_manifest(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")  # 0 rows — the situation from the live run
    with pytest.raises(RuntimeError, match="NOT proceeding to export"):
        ensure_manifest_pairs(manifest)


def test_ensure_manifest_pairs_raises_when_all_rows_lack_gt(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"id": "a", "image": "im/a.jpg", "category": "product", "gt_alpha": None}]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="only 0 pairs with GT"):
        ensure_manifest_pairs(manifest)


# ============================================================================
# 7b) _o00 end-to-end simulation: make_composites.run() on a small fixture
#     -> exclude_source_ids (derived from val_stems.json) -> merge into the
#     Drive manifest with merge_composite_manifest -- a hermetic simulation of
#     the composites_o + drive_copy stages of v3_veri_guncelleme_hucresi.py.
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

    # val_stems.json: ONE _v copy of source "a" landed in VAL -- "a" must be
    # excluded entirely (make_composites still processes "a" for the _v copies,
    # but it is excluded from _o00 generation).
    # Empty-manifest guard (end of the v3 cell's "manifest" stage — live-run
    # lesson): on a populated source manifest the guard PASSES and returns the
    # GT'd pair count; on an empty/missing manifest (raw data never downloaded
    # scenario) it raises RuntimeError and blocks PROCEEDING to composites_o/export.
    assert ensure_manifest_pairs(source_manifest) == 3
    empty_manifest = tmp_path / "empty_manifest.jsonl"
    empty_manifest.write_text("")
    with pytest.raises(RuntimeError, match="NOT proceeding to export"):
        ensure_manifest_pairs(empty_manifest)
    with pytest.raises(RuntimeError, match="manifest file missing"):
        ensure_manifest_pairs(tmp_path / "never_created.jsonl")

    val_stems = ["a_v03"]
    excluded, unmatched = derive_val_excluded_source_ids(val_stems)
    assert excluded == {"a"}
    assert unmatched == []

    out_dir = tmp_path / "composites_o"
    counts = mc.run(
        source_manifest, bg_dir, per_image=1, seed=42, out_dir=out_dir,
        exclude_source_ids=excluded, only_original_bg=True,
    )
    # _o00 was generated only for b and c ("a" was excluded); total = eligible x ORIGINAL_BG_COPIES.
    assert sum(counts.values()) == 2 * mc.ORIGINAL_BG_COPIES
    from benchmark.testset import load_manifest
    o00_ids = {r["id"] for r in load_manifest(str(out_dir / "manifest.jsonl"))}
    assert o00_ids == {"b_o00", "c_o00"}

    # Drive side: merge into a manifest that already contains v1/v2's _v<NN> rows.
    drive_manifest = tmp_path / "drive_train_composites_manifest.jsonl"
    drive_manifest.write_text(json.dumps(
        {"id": "a_v00", "image": "im/a_v00.jpg", "category": "transparent", "gt_alpha": "gt/a_v00.png"}
    ) + "\n")
    n_added = merge_composite_manifest(out_dir / "manifest.jsonl", drive_manifest)
    assert n_added == 2  # only the new _o00 rows were appended

    final_ids = [json.loads(line)["id"] for line in drive_manifest.read_text().splitlines() if line.strip()]
    assert set(final_ids) == {"a_v00", "b_o00", "c_o00"}

    # idempotency: calling the same merge again appends 0 rows.
    assert merge_composite_manifest(out_dir / "manifest.jsonl", drive_manifest) == 0


# 1c-3) v5 sampler preset (after the v4 benchmark + the ghosting finding:
# transparent/hair restored, fx/text/illustration pulled back to a
# gain-protection share — see the SAMPLER_PRESET_V5 docstring).
def test_sampler_preset_v5():
    assert abs(sum(SAMPLER_PRESET_V5.values()) - 1.0) < 1e-9
    assert set(SAMPLER_PRESET_V5) == {
        "camouflage", "transparent", "hair", "complex", "thin", "general",
        "text", "fx", "illustration",
    }
    from training.train_colab_lib import SAMPLER_PRESET_V4 as V4
    # directions: transparent and hair RESTORED UP, fx/text/illustration DOWN
    assert SAMPLER_PRESET_V5["transparent"] > V4["transparent"]
    assert SAMPLER_PRESET_V5["hair"] > V4["hair"]
    assert SAMPLER_PRESET_V5["fx"] < V4["fx"]
    assert SAMPLER_PRESET_V5["text"] < V4["text"]
    assert SAMPLER_PRESET_V5["complex"] == V4["complex"]


# ============================================================================
# 8) Packing the TRAIN data into tar shards — split_stems_to_shards /
#    tar_shard_name / validate_tar_manifest (packer: training/
#    veri_tar_paketleme_hucresi.py, consumer: train_colab.ipynb cell (c)).
# ============================================================================
def test_split_stems_to_shards_is_deterministic_regardless_of_input_order():
    # Filesystem listing order can vary from run to run — the split must be
    # ORDERED and DETERMINISTIC (idempotent shard skipping is only possible that way).
    import random

    stems = [f"s_{i:05d}" for i in range(100)]
    shuffled = stems[:]
    random.Random(0).shuffle(shuffled)
    assert split_stems_to_shards(stems, 30) == split_stems_to_shards(shuffled, 30)
    assert split_stems_to_shards(list(reversed(stems)), 30) == split_stems_to_shards(stems, 30)


def test_split_stems_to_shards_preserves_total_and_chunk_sizes():
    stems = [f"s_{i:05d}" for i in range(25)]
    shards = split_stems_to_shards(stems, 7)
    assert [len(s) for s in shards] == [7, 7, 7, 4]  # the last slice may be short
    flat = [x for sh in shards for x in sh]
    assert flat == sorted(stems)  # total PRESERVED: no loss, no duplication, sorted


def test_split_stems_to_shards_empty_list_gives_no_shards():
    assert split_stems_to_shards([], 7000) == []


def test_split_stems_to_shards_rejects_non_positive_shard_size():
    with pytest.raises(ValueError):
        split_stems_to_shards(["a"], 0)
    with pytest.raises(ValueError):
        split_stems_to_shards(["a"], -5)


def test_split_stems_to_shards_real_dataset_size_gives_about_eight_shards():
    # The real dataset size (52,882 pairs) + the packing cell's SHARD_SIZE=7000
    # value -> the task target of ~8 shards, ~6-7k pairs per shard.
    shards = split_stems_to_shards([f"{i:06d}" for i in range(52_882)], 7000)
    assert len(shards) == 8
    assert [len(s) for s in shards] == [7000] * 7 + [52_882 - 7 * 7000]


def test_tar_shard_name_format_and_rejects_negative():
    assert tar_shard_name(0) == "TRAIN_shard_00.tar"
    assert tar_shard_name(7) == "TRAIN_shard_07.tar"
    assert tar_shard_name(11) == "TRAIN_shard_11.tar"
    with pytest.raises(ValueError):
        tar_shard_name(-1)


def _valid_tar_manifest() -> dict:
    return {
        "created_at": "2026-07-13T00:00:00+00:00",
        "shard_size": 3,
        "total_pairs": 5,
        "source_counts": {"im": 5, "gt": 5},
        "shards": [
            {"name": "TRAIN_shard_00.tar", "pairs": 3, "files": 6, "bytes": 111},
            {"name": "TRAIN_shard_01.tar", "pairs": 2, "files": 4, "bytes": 99},
        ],
    }


def test_validate_tar_manifest_ok_returns_total():
    manifest = _valid_tar_manifest()
    assert validate_tar_manifest(manifest) == 5
    assert validate_tar_manifest(manifest, expected_total=5) == 5


def test_validate_tar_manifest_raises_on_shard_sum_mismatch():
    manifest = _valid_tar_manifest()
    manifest["total_pairs"] = 6  # shard sum is 5
    with pytest.raises(RuntimeError, match="does not match"):
        validate_tar_manifest(manifest)


def test_validate_tar_manifest_raises_on_expected_total_mismatch():
    # The packing cell passes the source TRAIN listing length — task requirement:
    # if the total pair count does not match the TRAIN listing, RuntimeError.
    with pytest.raises(RuntimeError, match="expected source pair count"):
        validate_tar_manifest(_valid_tar_manifest(), expected_total=52_882)


def test_validate_tar_manifest_raises_on_missing_or_empty_shards():
    with pytest.raises(RuntimeError, match="shards"):
        validate_tar_manifest({"total_pairs": 5})
    with pytest.raises(RuntimeError, match="shards"):
        validate_tar_manifest({"total_pairs": 5, "shards": []})


def test_validate_tar_manifest_raises_on_bad_total_pairs():
    manifest = _valid_tar_manifest()
    for bad in (None, 0, -1, "5"):
        m = dict(manifest)
        m["total_pairs"] = bad
        with pytest.raises(RuntimeError, match="total_pairs"):
            validate_tar_manifest(m)


def test_validate_tar_manifest_raises_on_broken_shard_entry():
    for broken in (
        {"name": "TRAIN_shard_00.tar", "pairs": 0, "bytes": 111},   # pairs <= 0
        {"name": "TRAIN_shard_00.tar", "pairs": 5, "bytes": 0},     # bytes <= 0
        {"pairs": 5, "bytes": 111},                                  # no name
        {"name": "TRAIN_shard_00.tar", "bytes": 111},                # no pairs
    ):
        manifest = {"total_pairs": 5, "shards": [broken]}
        with pytest.raises(RuntimeError, match="corrupt shard entry"):
            validate_tar_manifest(manifest)


def test_validate_tar_manifest_raises_on_duplicate_shard_names():
    manifest = {
        "total_pairs": 4,
        "shards": [
            {"name": "TRAIN_shard_00.tar", "pairs": 2, "bytes": 10},
            {"name": "TRAIN_shard_00.tar", "pairs": 2, "bytes": 20},
        ],
    }
    with pytest.raises(RuntimeError, match="duplicate shard names"):
        validate_tar_manifest(manifest)


# ============================================================================
# 8b) Drift guard: by its paste-run design (so it does not require a repo
#     clone), training/veri_tar_paketleme_hucresi.py carries VERBATIM COPIES
#     of these three functions from the lib — if a copy drifts from the source
#     in the lib, this test turns red (single source of truth:
#     training/train_colab_lib.py).
# ============================================================================
_LIB_PATH = Path(__file__).resolve().parent.parent / "training" / "train_colab_lib.py"
_PACKER_CELL_PATH = Path(__file__).resolve().parent.parent / "training" / "veri_tar_paketleme_hucresi.py"


def _function_def(path: Path, name: str) -> ast.FunctionDef:
    # CAUTION: the packing cell CANNOT be imported (paste-run — importing the
    # module would execute main(), and therefore drive.mount); only its source
    # text is parsed with ast.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found in {path}: {name}")


@pytest.mark.parametrize("func_name", ["tar_shard_name", "split_stems_to_shards", "validate_tar_manifest"])
def test_packer_cell_copies_match_lib_source(func_name):
    lib_node = _function_def(_LIB_PATH, func_name)
    cell_node = _function_def(_PACKER_CELL_PATH, func_name)
    assert ast.dump(cell_node) == ast.dump(lib_node), (
        f"{func_name}: the copy in the packing cell has DRIFTED from the lib — "
        f"the single source of truth is training/train_colab_lib.py; update the copy from there."
    )


# 1c-4) v7 sampler preset (issue #2: the design category — see the docstring).
def test_sampler_preset_v7():
    assert abs(sum(SAMPLER_PRESET_V7.values()) - 1.0) < 1e-9
    assert SAMPLER_PRESET_V7["design"] == 0.08
    assert SAMPLER_PRESET_V7["transparent"] == SAMPLER_PRESET_V5["transparent"]
    assert SAMPLER_PRESET_V7["complex"] == SAMPLER_PRESET_V5["complex"]
