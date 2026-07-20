"""scripts/make_bokeh_copies.py — the v8 bokeh hard-negative generator.

What matters here:
- the GT of a bokeh copy is BYTE-IDENTICAL to the source GT (the whole point:
  a defocused/glowing background around a furry subject is still exactly 0),
- the subject interior keeps its original pixels, the background actually
  changes (it is blurred),
- selection: only the requested categories; _e/_m/_k derivatives and VAL
  stems are never sources; GTs without enough background are skipped,
- determinism / idempotency / resume follow the make_v6_copies contracts.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "make_bokeh_copies",
    Path(__file__).parent.parent / "scripts" / "make_bokeh_copies.py",
)
mbc = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("make_bokeh_copies", mbc)
_SPEC.loader.exec_module(mbc)

SIZE = (96, 96)


def _write_pair(im_dir: Path, gt_dir: Path, stem: str, noise_seed: int = 0) -> None:
    """A noisy-background pair: the subject square is solid, the background is
    random noise (a flat background would make the blur a no-op)."""
    rng = np.random.default_rng(noise_seed)
    rgb = rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)
    rgb[24:72, 24:72] = (0, 180, 60)
    a = np.zeros(SIZE, dtype=np.uint8)
    a[24:72, 24:72] = 255
    im_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(im_dir / f"{stem}.jpg", format="JPEG", quality=92)
    Image.fromarray(a, mode="L").save(gt_dir / f"{stem}.png")


@pytest.fixture
def env(tmp_path):
    im_dir = tmp_path / "TRAIN" / "im"
    gt_dir = tmp_path / "TRAIN" / "gt"
    cats: dict[str, str] = {}
    for i in range(3):
        stem = f"p3m_hair_{i:03d}"
        _write_pair(im_dir, gt_dir, stem, noise_seed=i)
        cats[stem] = "hair"
    _write_pair(im_dir, gt_dir, "disvd_thing_000", noise_seed=7)
    cats["disvd_thing_000"] = "complex"  # not in the default categories
    _write_pair(im_dir, gt_dir, "p3m_hair_900_e00", noise_seed=8)
    cats["p3m_hair_900_e00"] = "hair"  # derivative -> never a source
    return {"im": im_dir, "gt": gt_dir, "cats": cats, "out": tmp_path / "out"}


def _run(env, **kw):
    return mbc.run(env["im"], env["gt"], env["cats"], env["out"], seed=42, count=100, **kw)


def _manifest_rows(out_dir: Path) -> list[dict]:
    path = out_dir / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_run_generates_hair_copies_only(env):
    result = _run(env)
    assert result == {"bokeh": 3}
    rows = _manifest_rows(env["out"])
    assert {r["id"] for r in rows} == {f"p3m_hair_{i:03d}_k00" for i in range(3)}
    assert all(r["category"] == "hair" for r in rows)


def test_gt_byte_identical_and_bg_actually_blurred(env):
    _run(env)
    for i in range(3):
        stem = f"p3m_hair_{i:03d}"
        src_gt = (env["gt"] / f"{stem}.png").read_bytes()
        out_gt = (env["out"] / "gt" / f"{stem}_k00.png").read_bytes()
        assert out_gt == src_gt, "the bokeh copy's GT must not move by a byte"

        src = np.asarray(Image.open(env["im"] / f"{stem}.jpg").convert("RGB"), dtype=np.int16)
        out = np.asarray(
            Image.open(env["out"] / "im" / f"{stem}_k00.jpg").convert("RGB"), dtype=np.int16
        )
        # subject interior (away from the matte edge): pixels essentially intact
        assert float(np.abs(out[30:66, 30:66] - src[30:66, 30:66]).mean()) < 4.0
        # background: the noise must be visibly smoothed
        assert float(np.abs(out[:20, :] - src[:20, :]).mean()) > 8.0


def test_render_keeps_alpha_object_identity():
    rng = np.random.default_rng(0)
    rgb = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    _, out_alpha = mbc.render_bokeh_copy(rng, rgb, alpha)
    assert out_alpha is alpha  # unchanged, not even copied


def test_ineligible_sources_skipped(env, tmp_path):
    """A GT without enough background (subject fills the frame) is skipped."""
    stem = "p3m_hair_full"
    rgb = np.random.default_rng(9).integers(0, 256, (*SIZE, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(env["im"] / f"{stem}.jpg", quality=92)
    Image.fromarray(np.full(SIZE, 255, dtype=np.uint8), mode="L").save(env["gt"] / f"{stem}.png")
    env["cats"][stem] = "hair"
    result = _run(env)
    assert result == {"bokeh": 3}  # the frame-filling GT did not become a source


def test_exclude_stems_val_guard(env):
    result = _run(env, exclude_stems={"p3m_hair_001"})
    assert result == {"bokeh": 2}
    ids = {r["id"] for r in _manifest_rows(env["out"])}
    assert "p3m_hair_001_k00" not in ids


def test_deterministic_same_seed_bit_identical(env, tmp_path):
    _run(env, out_manifest=None)
    out2 = tmp_path / "out2"
    mbc.run(env["im"], env["gt"], env["cats"], out2, seed=42, count=100)
    for i in range(3):
        stem = f"p3m_hair_{i:03d}_k00"
        assert (env["out"] / "im" / f"{stem}.jpg").read_bytes() == (out2 / "im" / f"{stem}.jpg").read_bytes()
        assert (env["out"] / "gt" / f"{stem}.png").read_bytes() == (out2 / "gt" / f"{stem}.png").read_bytes()


def test_idempotent_rerun_writes_nothing_new(env):
    _run(env)
    mtimes = {p: p.stat().st_mtime_ns for p in (env["out"] / "im").iterdir()}
    result = _run(env)
    assert result == {}
    assert len(_manifest_rows(env["out"])) == 3  # no duplicated manifest lines
    for p, t in mtimes.items():
        assert p.stat().st_mtime_ns == t, f"{p.name} was regenerated"


def test_resume_completes_missing_manifest_line(env):
    _run(env)
    manifest = env["out"] / "manifest.jsonl"
    rows = _manifest_rows(env["out"])
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[1:]))
    _run(env)
    assert {r["id"] for r in _manifest_rows(env["out"])} == {r["id"] for r in rows}
