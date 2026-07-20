"""FINISHER CELL — Stage 5 (composite generation) of `colab_devam_hucresi.py`
COULD NOT BE COMPLETED: at ~26-27k of the 28,281 target, giant foregrounds
(Transparent-460 x10 copies and/or HIM2K general images, 100-246MP) clogged
Colab's 12GB RAM and the user interrupted the cell.

USAGE: PASTE the ENTIRE contents of this file into a new cell in the live
Colab runtime (repo already checked out at /content/my-bg-remover, Drive
mounted, `pip install -e .` done, data/train/manifest.jsonl and
data/backgrounds already prepared — Stages 0-4 previously completed by
`colab_devam_hucresi.py`) and run it.

Why `scripts/make_composites.py::run()` is not simply called again: `run()`
accumulates new entries IN MEMORY in the `new_entries` list and writes them
to the file with a SINGLE bulk `append_entries` call (AFTER all source rows
are processed) — if interrupted, the manifest rows for the tens of thousands
of image/gt files ALREADY WRITTEN to disk up to that point are never added.
Calling `run()` again as-is would trigger needless regeneration for files
that already exist on disk (because their ids are NOT in the manifest) —
both a waste of time and a risk of getting stuck on the same giant images
again. That is why the "finisher loop" below checks FILE existence (not just
the manifest) and calls `append_entries` AFTER EVERY item (interruption
resilient).

Critical invariant: for items that are NOT giant-sized, the generation path
must be EXACTLY identical to `make_composites.run()` (same `_item_rng`
substream, same `compose`/`augment` call order) — otherwise a statistical
inconsistency arises between previously generated and newly generated items.
To guarantee this, all helper functions are imported from
`scripts/make_composites.py`, NOT COPIED. Only giant images (long side >
2048px) are downscaled BEFORE compose/augment — for those items bit-exact
identity with `run()` is deliberately NOT wanted (that is the whole point:
shrink the giant canvases and prevent the RAM blow-up).

Status tracking: the SAME `report()` mechanism as `colab_devam_hucresi.py`
is used (`/content/drive/MyDrive/bg-remover-status/log.txt` + `status.json`).
Stage names: `finisher` -> `export` -> `drive_copy` -> `ALL`. An unexpected
error is reported with `stage="FATAL"` with the full traceback and re-raised.
"""

import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# Transparent-460/HIM2K contain 100MP+ images; they can exceed PIL's 179MP
# "decompression bomb" error threshold. Since the data comes from trusted
# academic datasets, the limit is removed (SAME line as
# colab_devam_hucresi.py).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np
from PIL import Image

# --- Constants (SAME as colab_devam_hucresi.py) ---------------------------
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
SEED = 42
PER_IMAGE = 1  # SAME as per_image=1 in stage5_make_composites (drift prevention)
MAX_LONG_SIDE = 2048  # fgs longer than this are downscaled BEFORE compose/augment

STATUS_DIR = Path(DRIVE_ROOT) / "bg-remover-status"
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ is not a package — we add the absolute path to sys.path (SAME
# logic as colab_devam_hucresi.py, independent of os.chdir).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries, load_manifest  # noqa: E402


# ==========================================================================
# Status reporting — VERBATIM from colab_devam_hucresi.py (same mechanism).
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    """Appends a line to log.txt + rewrites status.json (accumulating history)."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now()
    line = f"[{ts}] stage={stage} status={status}"
    if extra:
        line += " " + json.dumps(extra, ensure_ascii=False, default=str)
    print(line)

    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

    history = []
    if STATUS_PATH.exists():
        try:
            history = json.loads(STATUS_PATH.read_text()).get("history", [])
        except Exception:
            history = []
    history.append({"stage": stage, "status": status, "time": ts, "detail": extra})
    payload = {"stage": stage, "status": status, "time": ts, "detail": extra, "history": history}
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


# ==========================================================================
# Giant image downscale — BEFORE compose/augment, only if long side > 2048px.
# ==========================================================================
def downscale_giant(
    rgb: np.ndarray, alpha: np.ndarray, max_long_side: int = MAX_LONG_SIDE
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Downscales rgb+alpha TOGETHER (to the same new size) — rgb via LANCZOS
    (quality), alpha with the SAME mode='F' + BILINEAR pattern as
    `bgr/compositing.py::_resize_alpha` (float32 precision preserved). If the
    long side is already <= max_long_side, returns (the same objects) without
    doing anything -> bit-exact identity with `make_composites.run()` is NOT
    BROKEN for non-giant items."""
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return rgb, alpha, False

    scale = max_long_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    rgb_ds = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8
    )
    alpha_ds = np.asarray(
        Image.fromarray(alpha.astype(np.float32), mode="F").resize((new_w, new_h), Image.BILINEAR),
        dtype=np.float32,
    ).clip(0, 1)
    return rgb_ds, alpha_ds, True


# ==========================================================================
# Finisher stage — completes make_composites.run() from where it left off,
# downscaling giant images. Helpers are imported from the mc module (NOT COPIED).
# ==========================================================================
def stage_finisher() -> dict:
    report("finisher", "running")
    os.chdir(WORKDIR)

    import make_composites as mc  # scripts/ is on sys.path
    from bgr.compositing import augment, compose

    manifest_path = Path("data/train/manifest.jsonl")
    backgrounds_dir = Path("data/backgrounds")
    out_dir = Path("data/train_composites")
    out_img_dir = out_dir / "images"
    out_gt_dir = out_dir / "gt"
    out_manifest = out_dir / "manifest.jsonl"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    # SAME filter/order as run() (required for the id/copy/seed contract).
    rows = [r for r in load_manifest(str(manifest_path)) if r.get("gt_alpha")]

    bg_paths = sorted(p for p in backgrounds_dir.iterdir() if p.suffix.lower() in mc.IMG_EXTS)
    if not bg_paths:
        raise SystemExit(f"no backgrounds found: {backgrounds_dir}")

    manifest_ids: set[str] = set()
    if out_manifest.exists():
        manifest_ids = {r["id"] for r in load_manifest(str(out_manifest))}

    # expected total: number of copies per source-manifest row according to the category multiplier.
    expected_total = sum(PER_IMAGE * mc.multiplier(r["category"]) for r in rows)

    counts: dict[str, int] = {}
    produced = 0
    reconciled = 0
    skipped = 0
    downscaled_ids: list[str] = []

    for row in rows:
        category = row["category"]
        n_copies = PER_IMAGE * mc.multiplier(category)
        out_ids = [f"{row['id']}_v{ci:02d}" for ci in range(n_copies)]

        # look at the DISK state first (not just the manifest) — here we repair
        # run()'s bulk-append gap (file exists, row does not).
        pending: list[tuple[str, Path, Path]] = []
        for out_id in out_ids:
            img_path = out_img_dir / f"{out_id}.jpg"
            gt_path = out_gt_dir / f"{out_id}.png"
            files_ok = img_path.exists() and gt_path.exists()
            row_exists = out_id in manifest_ids

            if files_ok and row_exists:
                skipped += 1
                continue
            if files_ok and not row_exists:
                # files already written (previous run was interrupted) -> only add the row.
                entry = {"id": out_id, "image": str(img_path), "category": category, "gt_alpha": str(gt_path)}
                append_entries(str(out_manifest), [entry])
                manifest_ids.add(out_id)
                reconciled += 1
                continue
            # files missing -> to be generated (whether the row exists or not; a
            # duplicate row will never be added, re-checked below).
            pending.append((out_id, img_path, gt_path))

        if not pending:
            continue

        fg_rgb = mc._load_rgb(Path(row["image"]))
        alpha = mc._load_alpha(Path(row["gt_alpha"]), target_size=(fg_rgb.shape[1], fg_rgb.shape[0]))

        for out_id, img_path, gt_path in pending:
            rng = mc._item_rng(SEED, out_id)  # SAME substream derivation as run()

            item_fg_rgb, item_alpha, was_ds = downscale_giant(fg_rgb, alpha)
            if was_ds:
                downscaled_ids.append(out_id)

            if category in mc.NO_COMPOSE_CATEGORIES:
                out_rgb, out_alpha = item_fg_rgb, item_alpha
            else:
                bg_idx = int(rng.integers(0, len(bg_paths)))
                bg_rgb = mc._load_rgb(bg_paths[bg_idx])
                out_rgb, out_alpha = compose(item_fg_rgb, item_alpha, bg_rgb, rng)

            out_rgb, out_alpha = augment(out_rgb, out_alpha, rng)
            mc._save_pair(out_rgb, out_alpha, img_path, gt_path)

            if out_id not in manifest_ids:
                entry = {"id": out_id, "image": str(img_path), "category": category, "gt_alpha": str(gt_path)}
                append_entries(str(out_manifest), [entry])  # AFTER EVERY ITEM -> interruption resilient
                manifest_ids.add(out_id)

            counts[category] = counts.get(category, 0) + 1
            produced += 1

            if produced % 100 == 0:
                report("finisher", "progress", produced=produced, downscaled=len(downscaled_ids))

    actual_total = len(list(out_img_dir.glob("*.jpg")))
    per_category_actual: dict[str, int] = {}
    if out_manifest.exists():
        for r in load_manifest(str(out_manifest)):
            per_category_actual[r["category"]] = per_category_actual.get(r["category"], 0) + 1

    ok = actual_total == expected_total
    print(f"Finisher: {produced} newly generated, {reconciled} rows repaired, {skipped} already complete.")
    print(f"Items downscaled due to giant size: {len(downscaled_ids)}")
    print(f"Expected total: {expected_total}  Actual total (images/): {actual_total}  Match: {ok}")
    for cat, c in sorted(per_category_actual.items()):
        print(f"  {cat}: {c}")
    if not ok:
        print("WARNING: expected and actual totals do not match — inspect before proceeding to export.")

    report(
        "finisher", "done",
        produced=produced, reconciled=reconciled, skipped=skipped,
        downscaled_count=len(downscaled_ids), expected_total=expected_total,
        actual_total=actual_total, counts_match=ok, per_category=per_category_actual,
    )
    return {
        "counts": counts, "expected_total": expected_total, "actual_total": actual_total, "ok": ok,
        "downscaled_ids": downscaled_ids,
    }


# ==========================================================================
# Export + Drive copy — VERBATIM from colab_devam_hucresi.py Stage 6/7.
# ==========================================================================
def stage6_export() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ is on sys.path

    stats = eb.export(
        manifest_path="data/train_composites/manifest.jsonl",
        out_dir="/content/birefnet_format",
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


def stage7_drive_copy(stats: dict) -> None:
    report("drive_copy", "running")
    src = Path("/content/birefnet_format")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Copying: {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True)

    comp_manifest = Path("data/train_composites/manifest.jsonl")
    if comp_manifest.exists():
        shutil.copy2(comp_manifest, dst / "train_composites_manifest.jsonl")
        print(f"Composite manifest also copied: {dst / 'train_composites_manifest.jsonl'}")

    src_im = list((src / "TRAIN" / "im").iterdir())
    src_gt = list((src / "TRAIN" / "gt").iterdir())
    dst_im = list((dst / "TRAIN" / "im").iterdir())
    dst_gt = list((dst / "TRAIN" / "gt").iterdir())

    with open(src / "stats.json") as f:
        stats_on_disk = json.load(f)

    print(f"im/: source={len(src_im)}, destination={len(dst_im)}")
    print(f"gt/: source={len(src_gt)}, destination={len(dst_gt)}")
    print(f"stats.json total: {stats_on_disk['total']}")

    assert len(src_im) == len(dst_im), "im/ file count does not match in the Drive copy!"
    assert len(src_gt) == len(dst_gt), "gt/ file count does not match in the Drive copy!"
    assert len(dst_im) == len(dst_gt) == stats_on_disk["total"], "im/gt/stats.json totals are inconsistent!"

    print("\nINTEGRITY CHECK PASSED — data is ready on Drive.")
    report("drive_copy", "done", im=len(dst_im), gt=len(dst_gt), total=stats_on_disk["total"])


# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
def main() -> None:
    os.chdir(WORKDIR)
    stage_finisher()
    stats = stage6_export()
    stage7_drive_copy(stats)
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
