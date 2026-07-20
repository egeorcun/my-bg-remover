"""V6 DATA UPDATE CELL — in a fresh (FREE, CPU is enough — NO GPU REQUIRED)
Colab session, adds ONLY the NEW v6 derivative copies to the existing Drive
dataset (`bg-remover-data/TRAIN`) (a data fix for the two flaws in GitHub
issue #1):
- **frame-crop** (`{stem}_e00`, ~9,000): subjects touching the frame must not
  get erased — crops that cut the subject bbox by 20-60% on one edge (the GT
  alpha does NOT change, the part touching the frame stays solid),
- **mixed-opacity** (`{stem}_m00`/`_m01`, <= 4,000): solid parts of
  transparent objects must not turn semi-transparent — augmented copies of
  transparent pairs that have both solid and soft alpha.
The entire generation logic lives in `scripts/make_v6_copies.py` (unit
tested); this cell only orchestrates the Colab flow (mount → source →
generation → export → Drive merge). It NEVER deletes/overwrites any existing
file.

SOURCE / ATTRIBUTION: the flow pattern (Drive mount before EVERYTHING →
`report()` stage tracking → `_listdir_retry` Errno 5 guard → TRAIN-only merge
→ `drive.flush_and_unmount()` at the end of the job) was taken from
`training/v4_veri_guncelleme_hucresi.py` — the 2026-07-12 lesson applies
VERBATIM: Drive writes are buffered asynchronously; if the VM is shut down
without a flush, files are SILENTLY lost.

DIFFERENCE FROM V4 — NO DOWNLOADS: the source for v6 generation is not
external datasets but the EXISTING TRAIN on Drive itself. Instead of copying
52k+ small files one by one over Drive FUSE, the tar shards produced by
`training/veri_tar_paketleme_hucresi.py` (`bg-remover-data/tar/
TRAIN_shard_XX.tar`, containing `im/<file>` + `gt/<file>`; `_manifest.json`)
are copied locally and extracted — the exact same tar-path pattern as
`train_colab.ipynb` cell (c) (byte verification included). This local TRAIN
is the source of the generation.

VAL LEAK GUARD (lesson from v3): the tar shards contain ALL of the Drive
TRAIN — including the stems that the training side set aside for VAL. Adding
an `_e00`/`_m00` derivative of a VAL stem (or of another copy of the SAME
source image) to TRAIN would mean that image is seen in both TRAIN and VAL.
If `bg-remover-status/val_stems.json` exists: the val stems themselves + ALL
stems whose source id (`tcl.strip_composite_copy_suffix`) matches a val
stem's source are removed from the source pool
(`tcl.derive_val_excluded_source_ids`).

TARS ARE NOT REPACKED: the training side (`train_colab.ipynb` cell (c))
already completes the pairs added to Drive afterwards (the delta) via
`copy_pairs` after extracting the tars — a note is printed to the user at the
end of the cell.

PREREQUISITES: the repo must be cloned at `/content/my-bg-remover` with
`pip install -e .` done; Drive must contain `bg-remover-data/TRAIN/{im,gt}`,
`bg-remover-data/tar/_manifest.json` (the packing cell must have run) and
`bg-remover-data/train_composites_manifest.jsonl`. The repo must be
UP-TO-DATE (the env stage attempts an idempotent `git pull`):
`scripts/make_v6_copies.py` was added in the same piece of work as this cell
— if you run with an old clone, `stage_v6` stops with a clear error message.

Status tracking is the SAME mechanism as the v4 cell (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — stages: env, tar_fetch,
categories, v6, export, drive_copy, (at the end) ALL.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# The TRAIN pool may contain composites sourced from 100MP+ images; PIL's
# 179MP "decompression bomb" error threshold is removed (see the same line in
# the v4 cell).
PIL.Image.MAX_IMAGE_PIXELS = None

# --- Constants (SAME Drive layout as v4_veri_guncelleme_hucresi.py) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"
SEED = 42

# --- v6-specific constants ---
LOCAL_TRAIN_ROOT = Path("/content/v6_train_src")  # local TRAIN where the tars are extracted (im/ + gt/)
TAR_CACHE = Path("/content/tar_cache_v6")         # temporary local copy of the shards (deleted one by one)
V6_OUT_DIR = Path("data/train_v6")                # make_v6_copies.run() output (local, relative to WORKDIR)
EXPORT_DIR = "/content/birefnet_format_v6"        # export_birefnet.export() output
EDGE_COUNT = 9000                                 # frame-crop target (~9k)
MIXED_CAP = 4000                                  # mixed-opacity copy cap (sources x 2)

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ is not a package — we add the absolute path to sys.path so that
# make_v6_copies/export_birefnet can be imported (see v4_veri_guncelleme_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import training.train_colab_lib as tcl  # noqa: E402  (package installed via pip install -e .)


# ==========================================================================
# Status reporting — IDENTICAL to `v4_veri_guncelleme_hucresi.py::report`.
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
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
# Drive FUSE Errno 5 guard — copy of the _listdir_retry pattern in the v4 cell.
# ==========================================================================
def _listdir_retry(d: Path, attempts: int = 5, wait_s: int = 30) -> list[Path]:
    """Drive FUSE occasionally throws a transient 'Errno 5 I/O error' on
    directories with 50k+ files (seen in the v3/v4 runs — retrying was
    enough); waits and retries, re-raising the error as-is on the last
    attempt."""
    for i in range(attempts):
        try:
            return list(d.iterdir())
        except OSError as e:
            if i == attempts - 1:
                raise
            print(f"WARNING: {e} while listing {d} — waiting {wait_s}s before retrying "
                  f"({i + 1}/{attempts - 1}).")
            time.sleep(wait_s)
    raise AssertionError("unreachable")


def _n_files(d: Path) -> int:
    return sum(1 for p in d.iterdir() if p.is_file()) if d.is_dir() else 0


# ==========================================================================
# Stage "env" — Drive mount (before EVERYTHING, STATUS_DIR lives on Drive!) +
# repo git pull (idempotent). Source: v4 cell stage0_env_sanity — since
# make_v6_copies was added in the same piece of work as this cell, a stale
# clone is the most likely source of failure.
# ==========================================================================
def _git_pull_idempotent() -> None:
    """Updates the repo — `git pull --ff-only` is a no-op if already
    up-to-date (idempotent); on no network/conflict it prints a WARNING and
    continues (if make_v6_copies is missing, stage_v6 will stop with a clear
    message anyway)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("WARNING: git pull failed — the repo may be stale; if make_v6_copies.py "
                  "is missing, we will stop below with a clear error.")
    except Exception as e:
        print(f"WARNING: could not run git pull ({e}) — continuing with the existing clone.")


def stage0_env() -> None:
    # Drive is mounted BEFORE EVERYTHING (including report() — STATUS_DIR is
    # on Drive!); drive.mount is idempotent. Source: same stage in the v4 cell.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive could not be mounted: {DRIVE_ROOT} missing"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()

    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~35 GB needed: tar extraction + v6 output)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


# ==========================================================================
# Stage "tar_fetch" — copy the tar shards from Drive to local disk + extract.
# Source pattern: train_colab.ipynb cell (c) "fast path" (byte verification,
# one local tar per shard, delete after extraction) — NO DOWNLOADS; the
# source data is the tar package of the EXISTING TRAIN on Drive.
# ==========================================================================
def stage_tar_fetch() -> int:
    report("tar_fetch", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    manifest_path = tar_dir / "_manifest.json"
    assert manifest_path.exists(), (
        f"{manifest_path} missing — first run training/veri_tar_paketleme_hucresi.py (free CPU "
        f"Colab) to pack TRAIN into tar shards; those shards are the source for v6 generation "
        f"(copying 52k+ small files one by one over Drive FUSE would take ~75 min)."
    )
    manifest = json.loads(manifest_path.read_text())
    total_pairs = tcl.validate_tar_manifest(manifest)  # internal consistency: shard sum == total_pairs

    local_im = LOCAL_TRAIN_ROOT / "im"
    local_gt = LOCAL_TRAIN_ROOT / "gt"
    n_im, n_gt = _n_files(local_im), _n_files(local_gt)
    if n_im >= total_pairs and n_im == n_gt:
        print(f"Tar download/extract SKIPPED: {n_im} pairs already local (>= manifest {total_pairs}).")
    else:
        LOCAL_TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        TAR_CACHE.mkdir(parents=True, exist_ok=True)
        for sh in manifest["shards"]:
            src, dst = tar_dir / sh["name"], TAR_CACHE / sh["name"]
            if not (dst.exists() and dst.stat().st_size == sh["bytes"]):
                shutil.copy2(src, dst)  # a single LARGE file — fast over Drive FUSE
                if dst.stat().st_size != sh["bytes"]:
                    raise RuntimeError(
                        f"{sh['name']}: size copied from Drive ({dst.stat().st_size}) does not "
                        f"match the manifest ({sh['bytes']}) — the transfer may have been cut "
                        f"short; re-run the cell."
                    )
            with tarfile.open(dst) as tf:
                tf.extractall(LOCAL_TRAIN_ROOT, filter="data")  # members: im/<file> + gt/<file>
            dst.unlink()  # the extracted shard's local tar is deleted immediately (disk safety)
            print(f"{sh['name']}: copied + extracted ({sh['pairs']} pairs, {sh['bytes'] / 1e9:.2f} GB).")
        n_im, n_gt = _n_files(local_im), _n_files(local_gt)
        if n_im != n_gt or n_im < total_pairs:
            raise RuntimeError(
                f"tar extraction does not match the manifest: im={n_im}, gt={n_gt}, expected at "
                f"least {total_pairs} (and im == gt) — shards may be missing/corrupt; re-run the "
                f"packing cell."
            )

    print(f"Local TRAIN source ready: {n_im} pairs -> {LOCAL_TRAIN_ROOT}")
    report("tar_fetch", "done", pairs=n_im, total_pairs_manifest=total_pairs)
    return n_im


# ==========================================================================
# Stage "categories" — stem -> category map from Drive's
# train_composites_manifest.jsonl (tcl.load_stem_categories — reads
# id/category) + the set of source stems to exclude for the VAL leak guard.
# ==========================================================================
def stage_categories() -> tuple[dict[str, str], set[str]]:
    report("categories", "running")
    drive_manifest = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / "train_composites_manifest.jsonl"
    assert drive_manifest.exists(), (
        f"{drive_manifest} missing — without the category map, v6 source selection "
        f"(proportional per category + transparent filter) cannot be done; the Phase 2 / v3 / "
        f"v4 cells must have run."
    )
    category_by_stem = tcl.load_stem_categories(drive_manifest)
    print(f"Category map: {len(category_by_stem)} stems.")

    # VAL leak guard (see the module docstring): the val stems themselves +
    # other copies of the same SOURCE image are removed from the source pool.
    exclude_stems: set[str] = set()
    val_json = STATUS_DIR / "val_stems.json"
    if val_json.exists():
        val_stems = json.loads(val_json.read_text())["val_stems"]
        excluded_ids, unmatched = tcl.derive_val_excluded_source_ids(val_stems)
        if unmatched:
            print(f"WARNING: {len(unmatched)} val stems do not match the `_v/_o<NN>` suffix "
                  f"pattern (e.g. {unmatched[:5]}) — those are excluded only by their own stem; "
                  f"source-id-level protection cannot be applied for them (lesson from v3).")
        exclude_stems = set(val_stems) | {
            s for s in category_by_stem
            if tcl.strip_composite_copy_suffix(s) in excluded_ids
        }
        print(f"VAL leak guard: {len(val_stems)} val stems -> {len(exclude_stems)} "
              f"stems will be excluded from the source pool.")
    else:
        print(f"NOTE: {val_json} missing (no training may have run yet) — skipping VAL "
              f"exclusion; new stems always go to TRAIN anyway.")

    report("categories", "done", stems=len(category_by_stem), excluded=len(exclude_stems))
    return category_by_stem, exclude_stems


# ==========================================================================
# Stage "v6" — GENERATION: scripts/make_v6_copies.py (unit tested). On a
# signature/import mismatch we stop with a CLEAR error message (v4
# stage_textfx pattern); half-finished data is never produced silently.
# ==========================================================================
def stage_v6(category_by_stem: dict[str, str], exclude_stems: set[str]) -> dict[str, int]:
    report("v6", "running")

    try:
        import make_v6_copies as mv6  # scripts/ is on sys.path
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_v6_copies.py could not be imported ({e}) — is your repo up-to-date? "
            f"Try 'git -C {WORKDIR} pull' (the script was added in the same piece of work as "
            f"this cell)."
        ) from e

    try:
        counts = mv6.run(
            train_im_dir=LOCAL_TRAIN_ROOT / "im",
            train_gt_dir=LOCAL_TRAIN_ROOT / "gt",
            category_by_stem=category_by_stem,
            out_dir=V6_OUT_DIR,
            seed=SEED,
            edge_count=EDGE_COUNT,
            mixed_cap=MIXED_CAP,
            exclude_stems=exclude_stems,
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_v6_copies.run() could not be called with the expected signature ({e}) — this "
            f"cell assumes the signature run(train_im_dir, train_gt_dir, category_by_stem, "
            f"out_dir, seed, edge_count, mixed_cap, exclude_stems); check the current signature "
            f"of scripts/make_v6_copies.py and adapt the call."
        ) from e

    print("make_v6_copies.run() production:", counts)

    # Manifest guard (lesson from v3): do NOT proceed to export with an
    # empty/missing manifest. make_v6_copies' output manifest has
    # {"id","category"} rows — since export_birefnet requires the FULL testset
    # schema (image + gt_alpha), it is converted to manifest_full (exact same
    # pattern as the v4 cell's stage_textfx).
    out_manifest = V6_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} missing — make_v6_copies generation must have failed.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{out_manifest} is empty — not proceeding to export (lesson from v3).")

    full_manifest = V6_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = V6_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = V6_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"file missing for manifest row: {r['id']} — generation may have been cut short.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")

    n_edge = sum(1 for r in rows if r["id"].endswith("_e00"))
    n_mixed = len(rows) - n_edge
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"PRE-FLIGHT — {out_manifest}: total {len(rows)} pairs "
          f"(edge-crop: {n_edge}, mixed: {n_mixed}); by category:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")
    if n_edge < 100:
        print(f"WARNING: edge-crop count is very low ({n_edge} < 100) — the source pool/category "
              f"map may be incomplete; inspect the logs.")

    report("v6", "done", counts=counts, total_pairs=len(rows), edge=n_edge, mixed=n_mixed,
           by_category=by_cat)
    return by_cat


# ==========================================================================
# Stage "export" — v4 pattern: export_birefnet.export() runs against a
# fresh/empty local directory; only the new v6 files appear on disk.
# split_name="TRAIN": new stems ALWAYS go to TRAIN (existing rule).
# ==========================================================================
def stage_export_v6() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ is on sys.path

    stats = eb.export(
        manifest_path=str(V6_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR,
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — v4 pattern: MERGE into the existing Drive TRAIN
# (dirs_exist_ok=True, no deletion/overwrite; im/gt counted SEPARATELY —
# 2026-07-12 lesson: a previously interrupted upload can leave pairs whose im
# arrived but whose gt did not) + APPEND to the composite manifest
# (tcl.merge_composite_manifest, with dedupe).
# ==========================================================================
def stage_drive_copy_v6() -> None:
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Expected TRAIN data not found on Drive: {dst_train_im} / {dst_train_gt} — "
        f"this cell is only for ADDING the v6 derivatives to an EXISTING dataset."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    assert len(src_im_files) == len(src_gt_files), "im/gt counts do not match in the local v6 export!"

    # im and gt are counted SEPARATELY (v4 cell / 2026-07-12 lesson — see the stage comment).
    existing_dst_im_stems = {p.stem for p in _listdir_retry(dst_train_im)}
    existing_dst_gt_stems = {p.stem for p in _listdir_retry(dst_train_gt)}
    growth_im = len({p.stem for p in src_im_files} - existing_dst_im_stems)
    growth_gt = len({p.stem for p in src_gt_files} - existing_dst_gt_stems)

    pre_im, pre_gt = len(existing_dst_im_stems), len(existing_dst_gt_stems)
    print(f"Drive TRAIN before merge: im={pre_im}, gt={pre_gt} — expected growth: "
          f"im +{growth_im}, gt +{growth_gt}")

    # ONLY the TRAIN/ subtree is copied — the PARTIAL stats.json at the src
    # root is NOT copied so it does not clobber the authoritative FULL
    # stats.json on Drive (v3 fix).
    print(f"Copying (MERGE, no deletion, TRAIN/ only): {src / 'TRAIN'} -> {dst / 'TRAIN'}")
    shutil.copytree(src / "TRAIN", dst / "TRAIN", dirs_exist_ok=True)

    post_im, post_gt = len(_listdir_retry(dst_train_im)), len(_listdir_retry(dst_train_gt))
    print(f"Drive TRAIN after merge: im={post_im}, gt={post_gt}")

    assert post_im - pre_im == growth_im, (
        f"im/ growth does not match the expectation: {post_im - pre_im} != {growth_im}"
    )
    assert post_gt - pre_gt == growth_gt, (
        f"gt/ growth does not match the expectation: {post_gt - pre_gt} != {growth_gt}"
    )
    # The REAL integrity condition: im/gt stem counts are equal after the merge.
    assert post_im == post_gt, f"Drive TRAIN im/gt counts are not equal: {post_im} != {post_gt}"

    # manifest_full.jsonl: the load_manifest validation inside
    # merge_composite_manifest requires the FULL schema (image+gt_alpha), so
    # the raw manifest cannot be passed.
    comp_manifest_local = V6_OUT_DIR / "manifest_full.jsonl"
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} new rows appended (existing rows "
          f"PRESERVED, not overwritten). May be 0 on a repair run — not an error (lesson from v4).")

    print("\nINTEGRITY CHECK PASSED — v6 (edge-crop + mixed) data MERGED into Drive.")
    report(
        "drive_copy", "done",
        added_im=growth_im, added_gt=growth_gt, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
def main() -> None:
    stage0_env()                                   # Drive mount + git pull — before everything
    stage_tar_fetch()                              # tar shards -> local TRAIN (generation source)
    category_by_stem, exclude_stems = stage_categories()
    stage_v6(category_by_stem, exclude_stems)      # make_v6_copies.run() + manifest guard
    stage_export_v6()
    stage_drive_copy_v6()
    report("ALL", "done")
    print(
        "\nNOTE: the tar shards were NOT REPACKED — on the next training run, "
        "train_colab.ipynb cell (c) will, after extracting the tars, fill in the new ~13k pairs "
        "as a delta from Drive via copy_pairs (takes a few minutes). If you want, you can re-run "
        "training/veri_tar_paketleme_hucresi.py to reset the delta "
        "(CAUTION: because new stems slot into the ordering, most shards change and get "
        "repacked — an ~1 hour free-CPU run; the delta copy_pairs is usually cheaper)."
    )
    # CRITICAL (2026-07-12 lesson): Drive writes are buffered ASYNCHRONOUSLY —
    # if the VM is shut down before this flush finishes, files are SILENTLY
    # lost. flush_and_unmount() FORCES the buffer to drain and blocks until it
    # is done. It is called AFTER EVERYTHING that writes to Drive (report
    # included).
    print("Flushing Drive (waiting for async writes to land in the cloud)...")
    from google.colab import drive as _gdrive
    _gdrive.flush_and_unmount()
    print("Drive flush COMPLETE — the VM can now be safely shut down/swapped.")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
