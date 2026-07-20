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
    """Fake manifest + fake background pool (a distinctive magenta color, to
    make composites easy to detect)."""
    src = tmp_path / "src"
    bg_dir = tmp_path / "backgrounds"
    for i in range(3):
        _write_solid(bg_dir / f"bg{i}.jpg", (20, 20), (255, 0, 255))  # magenta

    manifest = tmp_path / "train_manifest.jsonl"
    rows = []

    def _add(name, category, alpha_partial=True, with_gt=True):
        _write_solid(src / f"{name}.jpg", (16, 16), (0, 200, 0))  # green fg
        gt_alpha = None
        if with_gt:
            gt_path = src / f"{name}_gt.png"
            a = np.full((16, 16), 255, dtype=np.uint8)
            if alpha_partial:
                a[:8, :] = 128  # top half is semi-transparent -> the compose trace shows up here
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
    assert counts["camouflage"] == 2  # NO_COMPOSE_CATEGORIES -> no _o00
    assert counts["transparent"] == 10 + 1  # _v x10 + v3 _o00 x1
    assert counts["hair"] == 1 + 1  # _v x1 + v3 _o00 x1
    assert "product" not in counts  # gt_alpha=None -> not included in compositing


def test_run_per_image_multiplies_category_factor(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=3, seed=42, out_dir=env["out"])
    assert counts["camouflage"] == 6
    # ORIGINAL_BG_COPIES (_o00) does NOT scale with per_image -- a constant +1.
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
    """Compose is SKIPPED for the camouflage category: in the semi-transparent
    (alpha=0.5) region there must be no leakage from the magenta background
    pool (only augment is applied, the original background/color is preserved)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    cam_rows = [r for r in loaded if r["category"] == "camouflage"]
    assert cam_rows
    for row in cam_rows:
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
        # if the magenta bg (255,0,255) had been mixed in, the R+B channels would be very high;
        # even with the green fg + jitter/jpeg variance, the R+B sum must stay low.
        assert rgb[..., 0].mean() + rgb[..., 2].mean() < 150


def test_run_transparent_does_compose_bg_contamination_present(env):
    """The transparent category is composed: the semi-transparent region must
    show clear leakage from the magenta background (green fg + magenta bg
    mix -> high R+B)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    trans_rows = [r for r in loaded if r["category"] == "transparent"]
    assert trans_rows
    contaminated = False
    for row in trans_rows:
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
        if rgb[..., 0].mean() + rgb[..., 2].mean() > 150:
            contaminated = True
    assert contaminated, "no composed transparent copy showed bg leakage"


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
        assert np.array_equal(img1, img2), f"{rid}: same seed produced a different output"


def test_run_idempotent_skips_existing(env):
    counts1 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    total1 = sum(counts1.values())
    assert total1 > 0
    counts2 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    assert counts2 == {}  # nothing new was generated on the second run
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    assert len(loaded) == total1  # no duplicates in the manifest


def test_partial_then_resume_matches_full_run(env):
    """Interruption simulation: after a full run, HALF of the output (files +
    manifest lines) is deleted and the run is repeated. The resumed run must
    produce files bit-identical to an uninterrupted full run (SeedSequence
    sub-seeds are order-independent)."""
    dir_full = env["out"] / "full"
    dir_resume = env["out"] / "resume"
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_full)
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_resume)

    # delete half: every second line of the manifest + those lines' files
    resume_manifest = dir_resume / "manifest.jsonl"
    rows = load_manifest(str(resume_manifest))
    keep, drop = rows[::2], rows[1::2]
    assert drop, "test is meaningless: no lines to delete"
    for row in drop:
        Path(row["image"]).unlink()
        Path(row["gt_alpha"]).unlink()
    resume_manifest.unlink()
    append_entries(str(resume_manifest), keep)

    # resumed run: only the deleted ones should be regenerated
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=dir_resume)
    assert sum(counts.values()) == len(drop)

    full_rows = {r["id"]: r for r in load_manifest(str(dir_full / "manifest.jsonl"))}
    resume_rows = {r["id"]: r for r in load_manifest(str(resume_manifest))}
    assert full_rows.keys() == resume_rows.keys()
    for rid, full_row in full_rows.items():
        resume_row = resume_rows[rid]
        assert Path(full_row["image"]).read_bytes() == Path(resume_row["image"]).read_bytes(), (
            f"{rid}: resumed run image differs from the full run"
        )
        assert Path(full_row["gt_alpha"]).read_bytes() == Path(resume_row["gt_alpha"]).read_bytes(), (
            f"{rid}: resumed run gt differs from the full run"
        )


def test_run_limit_caps_source_rows(env):
    counts = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"], limit=1)
    assert sum(counts.values()) < 2 + 4 + 1


# ============================================================================
# v3 — original background copies (_o<NN>)
# ============================================================================
def test_run_generates_o00_for_non_camouflage_categories(env):
    """Every category EXCEPT camouflage (transparent, hair) must get 1 extra
    `_o00` copy; camouflage is already compose-free, so no _o00 should be
    generated for it (redundant)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert "trans1_o00" in ids
    assert "hair1_o00" in ids
    assert "cam1_o00" not in ids  # NO_COMPOSE_CATEGORIES -> NO _o00


def test_run_o00_keeps_original_background_no_compose_contamination(env):
    """The _o00 copy follows camouflage's path: NO compose, augment only —
    there must be no leakage from the magenta background pool."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = {r["id"]: r for r in load_manifest(str(env["out"] / "manifest.jsonl"))}
    row = loaded["trans1_o00"]
    rgb = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
    assert rgb[..., 0].mean() + rgb[..., 2].mean() < 150


def test_run_o00_respects_exclude_source_ids(env):
    """No _o00 should be generated for source ids in `exclude_source_ids` (VAL
    leak guard) — _v<NN> copies must NOT be affected."""
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        exclude_source_ids={"trans1"},
    )
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert "trans1_o00" not in ids
    assert "hair1_o00" in ids  # the other, non-excluded source was unaffected
    assert counts["transparent"] == 10  # only the _v<NN>s (10 of them), no _o00


def test_run_only_original_bg_skips_all_v_copies(env):
    """`only_original_bg=True`: _v<NN> copies are skipped ENTIRELY, only the
    _o00 set is generated (a quick continuation on a fresh VM without
    regenerating the whole composite set)."""
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        only_original_bg=True,
    )
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = {r["id"] for r in loaded}
    assert ids == {"trans1_o00", "hair1_o00"}  # camouflage excluded, no _v<NN> at all
    assert counts == {"transparent": 1, "hair": 1}


def test_run_only_original_bg_with_exclusion_produces_nothing_for_excluded(env):
    counts = mc.run(
        env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"],
        only_original_bg=True, exclude_source_ids={"trans1", "hair1"},
    )
    assert counts == {}
    # since no new entries were written, the manifest file is never created (append_entries
    # is only called when new_entries is non-empty) -- NOT an empty manifest.jsonl, absent.
    assert not (env["out"] / "manifest.jsonl").exists()


def test_run_o00_naming_never_collides_with_v_copies(env):
    """The `_o00` namespace NEVER collides with `_v<NN>` (a separate suffix) —
    among all ids produced by a normal (pre-v3-like) run, both _v and _o
    copies can exist independently, neither overwrites the other."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    loaded = load_manifest(str(env["out"] / "manifest.jsonl"))
    ids = [r["id"] for r in loaded]
    assert len(ids) == len(set(ids))  # no duplicates
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
        assert np.array_equal(img1, img2), f"{rid}: same seed produced a different _o00 output"


def test_run_rejects_copy_counts_that_overflow_two_digit_suffix(env):
    """per_image x the largest category multiplier > 99 -> AssertionError:
    `{ci:02d}` overflows to 3 digits and the VAL leak guard's `_[vo]\\d{2}$`
    suffix pattern (train_colab_lib.strip_composite_copy_suffix) would no
    longer match those ids."""
    with pytest.raises(AssertionError, match="99"):
        # with the transparent x10 multiplier, per_image=10 -> 100 copies (>99).
        mc.run(env["manifest"], env["backgrounds"], per_image=10, seed=42, out_dir=env["out"])


def test_run_idempotent_rerun_adds_only_missing_o00(env):
    """When re-running on top of an existing _v<NN> run to add _o00s later,
    ONLY the missing _o00s are generated — the existing _v<NN> outputs remain
    unchanged (see the idempotency note in the module docstring)."""
    mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    before = {r["id"]: Path(r["image"]).read_bytes() for r in load_manifest(str(env["out"] / "manifest.jsonl"))}

    counts2 = mc.run(env["manifest"], env["backgrounds"], per_image=1, seed=42, out_dir=env["out"])
    assert counts2 == {}  # everything already existed, nothing new generated

    after = load_manifest(str(env["out"] / "manifest.jsonl"))
    assert len(after) == len(before)
    for row in after:
        assert Path(row["image"]).read_bytes() == before[row["id"]]
