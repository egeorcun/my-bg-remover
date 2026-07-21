"""V8 DATA UPDATE CELL — in a fresh (FREE, CPU is enough — NO GPU REQUIRED)
Colab session, prepares the v8 "background purity" data on Drive
(`bg-remover-data/TRAIN`). Two jobs:

1. **Bokeh hard-negatives (`{stem}_k00`, ~9,000 NEW pairs)** — the fix for the
   real-photo background smears (HF discussion #1, the cat masks: lucida
   leaves a gray haze around furry subjects where birefnet-hr outputs bit-zero
   background; measured hair-category bg_mae: v7 0.0069 vs birefnet 0.0003).
   `scripts/make_bokeh_copies.py` (unit tested) re-renders existing `hair`
   pairs with a defocused background (+ bright bokeh orbs) while the GT stays
   byte-identical — the counter-lesson: blur/glow around fur is still exactly
   0. Sources come from the tar shards (the v6 cell's tar-fetch pattern).

2. **Design regeneration (~6,000 EXISTING pairs OVERWRITTEN)** — the fix for
   the white residue between the letters of dense typography and for the
   canvas-wide glow tails. `scripts/make_design.py` was fixed in the same
   piece of work as this cell (compact glow support: exact 0 beyond 2.5 sigma;
   glow zeroed under the text bands). The SAME seed and count are used, so the
   SAME `design_NNNNN_c00` ids are produced with corrected renders — the Drive
   merge overwrites the files in place; the manifest does not change (the
   merge dedupes to 0 new rows, which is expected).

REFRESH MARKER (`bg-remover-data/tar/_refresh_stems.json`): the regenerated
design stems are recorded so that `train_colab.ipynb` cell (c) re-validates
their local copies against Drive (size-checked `copy_pairs`) AFTER the tar
extraction. Today the design stems are not inside the tar shards (the shards
were packed pre-v6) and arrive via the normal delta copy anyway; the marker
makes the flow correct even if the shards get repacked later.

SOURCE / ATTRIBUTION: the flow pattern (Drive mount before EVERYTHING →
`report()` stage tracking → `_listdir_retry` Errno 5 guard → TRAIN-only merge
→ `drive.flush_and_unmount()` at the end) is the v4/v6/v7 cell pattern — the
2026-07-12 lesson applies VERBATIM: Drive writes are buffered asynchronously;
if the VM is shut down without a flush, files are SILENTLY lost.

VAL LEAK GUARD: same as the v6/v7 cells — `bg-remover-status/val_stems.json`
stems (and every stem sharing their source id) are excluded from the bokeh
source pool; the design fg pools exclude the raw stems mapped back from VAL
composite ids.

PREREQUISITES: the repo must be cloned at `/content/my-bg-remover` with
`pip install -e .` done; Drive must contain `bg-remover-data/TRAIN/{im,gt}`,
`bg-remover-data/tar/_manifest.json` and
`bg-remover-data/train_composites_manifest.jsonl`. The repo must be
UP-TO-DATE (the env stage attempts an idempotent `git pull`):
`scripts/make_bokeh_copies.py` and the make_design glow fix were added in the
same piece of work as this cell — a stale clone stops with a clear error.

Status tracking is the SAME mechanism as the v6/v7 cells (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — stages: env, tar_fetch,
categories, bokeh, downloads, fonts, val_guard, design, export, drive_copy,
refresh_marker, (at the end) ALL.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# The TRAIN pool / fg sources may contain 100MP+ images; PIL's 179MP
# "decompression bomb" error threshold is removed (see the same line in the
# v4/v6/v7 cells).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS must come AFTER the PIL import)
from PIL import Image  # noqa: E402

# --- Constants (SAME Drive layout as the v4/v6/v7 cells) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
TAR_SUBDIR = "tar"
SEED = 42

# --- v8-specific constants ---
LOCAL_TRAIN_ROOT = Path("/content/v8_train_src")   # local TRAIN where the tars are extracted (im/ + gt/)
TAR_CACHE = Path("/content/tar_cache_v8")          # temporary local copy of the shards (deleted one by one)
BOKEH_OUT_DIR = Path("data/train_v8_bokeh")        # make_bokeh_copies.run() output (relative to WORKDIR)
BOKEH_COUNT = 9000                                 # bokeh hard-negative target (~9k, hair category)
EXPORT_DIR_BOKEH = "/content/birefnet_format_v8_bokeh"

RAW = Path("data/raw_train")
TOONOUT_HF_REPO = "joelseytre/toonout"
TOONOUT_DIR = Path("/content/downloads/toonout")   # normalized im/ gt/ go here
FONT_DIR = Path("/content/fonts")
TRANS460_PAIRS = RAW / "trans460_pairs"            # fg/alpha -> im/gt symlink bridge
DESIGN_OUT_DIR = Path("data/train_design_v8")      # FRESH dir — v8 renders must not resume over v7 files
EXPORT_DIR_DESIGN = "/content/birefnet_format_v8_design"
DESIGN_COUNT = 6000                                # SAME count as v7 -> SAME ids, corrected renders

# composite id contract: f"{source_name}_{_sanitize(raw_stem)}" (build_trainset)
# — pool -> prefix, for mapping VAL source ids back to raw fg stems (v7 cell copy).
FG_SOURCE_PREFIXES = {
    "trans460_pairs": "transparent_460_train",
    "him2k_merged": "him2k",
}

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ is not a package — we add the absolute path to sys.path so that
# make_bokeh_copies/make_design/export_birefnet/build_testset can be imported.
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import CATEGORIES  # noqa: E402  (package installed via pip install -e .)
import training.train_colab_lib as tcl  # noqa: E402


# ==========================================================================
# Status reporting — IDENTICAL to the v6/v7 cells' `report`.
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
# Drive FUSE Errno 5 guard — copy of the _listdir_retry pattern in the v6/v7 cells.
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
# repo git pull (idempotent) + HF env. Source: v7 cell stage0_env — since
# make_bokeh_copies and the make_design fix were added in the same piece of
# work as this cell, a stale clone is the most likely source of failure.
# ==========================================================================
def _git_pull_idempotent() -> None:
    """Updates the repo — `git pull --ff-only` is a no-op if already
    up-to-date (idempotent); on no network/conflict it prints a WARNING and
    continues (if make_bokeh_copies is missing, stage_bokeh will stop with a
    clear message anyway)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("WARNING: git pull failed — the repo may be stale; if the v8 scripts are "
                  "missing, we will stop below with a clear error.")
    except Exception as e:
        print(f"WARNING: could not run git pull ({e}) — continuing with the existing clone.")


def _purge_script_module_cache() -> None:
    """Colab keeps one kernel across cell interrupts/re-runs, and Python
    caches imported modules in sys.modules — so after a `git pull` an
    `import make_bokeh_copies` would silently reuse the PRE-pull module
    (lived through exactly this on 2026-07-21: an interrupted first run left
    the old, slow generator cached, and the re-run kept executing it while
    the pulled optimization sat unused on disk). The scripts/ modules and
    the training lib are evicted so the imports below always load the
    just-pulled code; a kernel restart is no longer required after a pull."""
    import importlib

    for name in ("make_bokeh_copies", "make_design", "make_textfx",
                 "make_v6_copies", "export_birefnet", "build_testset", "build_trainset"):
        sys.modules.pop(name, None)
    importlib.reload(tcl)


def _setup_hf_env() -> None:
    """Source: same function in the v4/v7 cells (for HF downloads)."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF_TOKEN obtained from Colab Secrets.")
    except Exception as e:
        print(f"Could not get HF_TOKEN (not in Secrets or access not granted): {e}")


def stage0_env() -> None:
    # Drive is mounted BEFORE EVERYTHING (including report() — STATUS_DIR is
    # on Drive!); drive.mount is idempotent. Source: same stage in the v6/v7 cells.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive could not be mounted: {DRIVE_ROOT} missing"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _purge_script_module_cache()
    _setup_hf_env()

    free_gb = shutil.disk_usage("/content").free / 1e9
    print(f"local free disk: {free_gb:.0f} GB (~45 GB needed: tar extraction + ~3GB fg "
          f"downloads + bokeh/design outputs)")
    report("env", "done", cwd=str(Path.cwd()), free_gb=round(free_gb, 1))


# ==========================================================================
# Stage "tar_fetch" — copy the tar shards from Drive to local disk + extract
# (bokeh generation source). Source: v6 cell stage_tar_fetch (copy).
# ==========================================================================
def stage_tar_fetch() -> int:
    report("tar_fetch", "running")
    tar_dir = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR
    manifest_path = tar_dir / "_manifest.json"
    assert manifest_path.exists(), (
        f"{manifest_path} missing — first run training/veri_tar_paketleme_hucresi.py (free CPU "
        f"Colab) to pack TRAIN into tar shards; those shards are the source for bokeh generation "
        f"(copying 68k+ small files one by one over Drive FUSE would take hours)."
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
# Stage "categories" — stem -> category map + VAL leak guard for the bokeh
# source pool. Source: v6 cell stage_categories (copy).
# ==========================================================================
def stage_categories() -> tuple[dict[str, str], set[str]]:
    report("categories", "running")
    drive_manifest = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / "train_composites_manifest.jsonl"
    assert drive_manifest.exists(), (
        f"{drive_manifest} missing — without the category map, the bokeh source selection "
        f"(hair category) cannot be done; the Phase 2 / v3 / v4 cells must have run."
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
# Stage "bokeh" — GENERATION: scripts/make_bokeh_copies.py (unit tested). On
# a signature/import mismatch we stop with a CLEAR error message (v6 stage_v6
# pattern); half-finished data is never produced silently.
# ==========================================================================
def stage_bokeh(category_by_stem: dict[str, str], exclude_stems: set[str]) -> dict[str, int]:
    report("bokeh", "running")

    try:
        import make_bokeh_copies as mbc  # scripts/ is on sys.path
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_bokeh_copies.py could not be imported ({e}) — is your repo "
            f"up-to-date? Try 'git -C {WORKDIR} pull' (the script was added in the same piece "
            f"of work as this cell)."
        ) from e

    try:
        counts = mbc.run(
            train_im_dir=LOCAL_TRAIN_ROOT / "im",
            train_gt_dir=LOCAL_TRAIN_ROOT / "gt",
            category_by_stem=category_by_stem,
            out_dir=BOKEH_OUT_DIR,
            seed=SEED,
            count=BOKEH_COUNT,
            categories={"hair"},
            exclude_stems=exclude_stems,
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_bokeh_copies.run() could not be called with the expected signature ({e}) — "
            f"this cell assumes the signature run(train_im_dir, train_gt_dir, category_by_stem, "
            f"out_dir, seed, count, categories, exclude_stems); check the current signature of "
            f"scripts/make_bokeh_copies.py and adapt the call."
        ) from e

    print("make_bokeh_copies.run() production:", counts)

    # Manifest guard (lesson from v3): do NOT proceed to export with an
    # empty/missing manifest; converted to manifest_full for export (the
    # v4/v6/v7 cell pattern).
    out_manifest = BOKEH_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} missing — make_bokeh_copies generation must have failed.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{out_manifest} is empty — not proceeding to export (lesson from v3).")

    full_manifest = BOKEH_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = BOKEH_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = BOKEH_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"file missing for manifest row: {r['id']} — generation may have been cut short.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")

    print(f"PRE-FLIGHT — {out_manifest}: total {len(rows)} bokeh pairs (all `hair`).")
    if len(rows) < 100:
        print(f"WARNING: bokeh count is very low ({len(rows)} < 100) — the hair source pool/"
              f"category map may be incomplete; inspect the logs.")

    report("bokeh", "done", counts=counts, total_pairs=len(rows))
    return counts


# ==========================================================================
# Stage "downloads" — ONLY the fg sources design needs, IDEMPOTENT.
# Source: v7 cell (copy; originally from the v3/v4 cells). BG-20k is NOT
# downloaded (the background is synthetic).
# ==========================================================================
def _load_source_defs() -> dict:
    with open("data/train_sources.json") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _download_trans460(source_defs: dict) -> int:
    """Source: v7 cell::_download_trans460 (copy) — design fg source:
    fg/ + alpha/ (transparent objects, cutouts with real alpha)."""
    from huggingface_hub import snapshot_download

    spec = source_defs["transparent_460_train"]
    trans_out = RAW / "trans460_train"
    existing = len(list((trans_out / "fg").iterdir())) if (trans_out / "fg").exists() else 0
    expected = spec.get("full_pair_count") or 0
    if expected and existing >= 0.9 * expected:
        print(f"trans460_train: {existing} images already on disk (>= 90% x {expected}); skipping download.")
        return existing

    trans_dir = snapshot_download(repo_id=spec["hf_repo"], repo_type="dataset", allow_patterns=["Train/*"])
    if trans_out.exists():
        shutil.rmtree(trans_out)
    shutil.copytree(Path(trans_dir) / "Train" / "fg", trans_out / "fg")
    shutil.copytree(Path(trans_dir) / "Train" / "alpha", trans_out / "alpha")
    total = len(list((trans_out / "fg").iterdir()))
    print(f"transparent_460_train: {total} images -> {trans_out}")
    return total


def _normalize_trans460_pairs() -> int:
    """Bridges Transparent-460's `fg/` + `alpha/` layout to the `im/` + `gt/`
    layout make_design expects, via STEM-matched SYMLINKS (v7 cell copy).
    Idempotent: existing links are not recreated."""
    src_fg = RAW / "trans460_train" / "fg"
    src_alpha = RAW / "trans460_train" / "alpha"
    if not (src_fg.is_dir() and src_alpha.is_dir()):
        print("trans460_pairs: source fg/alpha missing — skipping the bridge.")
        return 0
    out_im = TRANS460_PAIRS / "im"
    out_gt = TRANS460_PAIRS / "gt"
    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    alphas = {p.stem: p for p in src_alpha.iterdir()
              if p.is_file() and not p.name.startswith("._")}
    n = 0
    for img in sorted(src_fg.iterdir()):
        if not img.is_file() or img.name.startswith("._"):
            continue
        gt = alphas.get(img.stem)
        if gt is None:
            continue
        dst_i = out_im / img.name
        dst_g = out_gt / gt.name
        if not dst_i.exists():
            dst_i.symlink_to(img.resolve())
        if not dst_g.exists():
            dst_g.symlink_to(gt.resolve())
        n += 1
    print(f"trans460_pairs: {n} pairs of im/gt symlink bridge ready -> {TRANS460_PAIRS}")
    return n


def _ensure_gdown() -> None:
    """Source: v4/v7 cell::_ensure_gdown (copy)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)


def _gdown_extract(drive_id: str, out_dir: Path, label: str) -> bool:
    """Source: v4/v7 cell::_gdown_extract (copy) — returns False on failure
    (does not stop the pipeline), skips if out_dir is populated."""
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{label}: {out_dir} already populated; skipping download.")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    try:
        import gdown

        gdown.download(id=drive_id, output=str(zip_path), quiet=False)
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        print(f"{label}: downloaded and extracted -> {out_dir}")
        return True
    except Exception as e:
        print(f"WARNING: {label} could not be downloaded ({e}) — this source will be SKIPPED.")
        return False


def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
    """Source: v4/v7 cell::_walk_dirs (copy)."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
        jpgs = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg"))]
        pngs = [f for f in filenames if f.lower().endswith(".png")]
        out.append({
            "path": Path(dirpath),
            "jpg_count": len(jpgs),
            "png_count": len(pngs),
            "jpg_stems": {Path(f).stem for f in jpgs},
            "png_stems": {Path(f).stem for f in pngs},
            "subdirs": list(dirnames),
        })
    return out


def discover_him2k_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
    """Source: v4/v7 cell::discover_him2k_dirs (copy)."""
    if not raw_dir.exists():
        return None

    images_dir = None
    alphas_dir = None
    for dirpath, _dirnames, _filenames in os.walk(raw_dir):
        p = Path(dirpath)
        if p.name.lower() == "train" and p.parent.name.lower() == "images":
            images_dir = p
        if p.name.lower() == "train" and p.parent.name.lower() == "alphas":
            alphas_dir = p
    if images_dir and alphas_dir:
        return images_dir, alphas_dir

    dirs = _walk_dirs(raw_dir, max_depth=4)
    img_cands = [d for d in dirs if d["jpg_count"] >= 10]
    if not img_cands:
        return None
    img_best = max(img_cands, key=lambda d: d["jpg_count"])

    alpha_best = None
    best_score = -1
    for d in dirs:
        if d["path"] == img_best["path"]:
            continue
        score = len(d["subdirs"]) if d["subdirs"] else d["png_count"]
        if score > best_score and score > 0:
            best_score = score
            alpha_best = d["path"]
    if alpha_best is None:
        return None
    return img_best["path"], alpha_best


def merge_him2k(images_dir: Path, alphas_dir: Path, out_root: Path) -> int:
    """Source: v4/v7 cell::merge_him2k (copy) — max-merges the instance
    alphas and produces {im,gt} pairs (design general fg source)."""
    out_im = out_root / "im"
    out_gt = out_root / "gt"
    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    images = {p.stem: p for p in images_dir.iterdir()
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    count = 0
    for stem, img_path in sorted(images.items()):
        inst_dir = alphas_dir / stem
        merged = None
        if inst_dir.is_dir():
            insts = sorted(list(inst_dir.glob("*.png")) + list(inst_dir.glob("*.jpg")))
            for ip in insts:
                arr = np.asarray(Image.open(ip).convert("L"), dtype=np.uint8)
                merged = arr if merged is None else np.maximum(merged, arr)
        else:
            flat = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = alphas_dir / f"{stem}{ext}"
                if cand.exists():
                    flat = cand
                    break
            if flat is not None:
                merged = np.asarray(Image.open(flat).convert("L"), dtype=np.uint8)

        if merged is None:
            continue
        Image.fromarray(merged, mode="L").save(out_gt / f"{stem}.png")
        shutil.copy2(img_path, out_im / img_path.name)
        count += 1
    return count


def _ensure_him2k_merged(source_defs: dict) -> int:
    """Source: v4/v7 cell (copy) — idempotent download+merge; a failed
    download prints a WARNING and the pipeline continues with the other fg
    sources."""
    _ensure_gdown()
    ok = _gdown_extract(source_defs["him2k"]["drive_id"], RAW / "him2k_raw", "HIM2K")
    if not ok:
        return 0
    out_root = RAW / "him2k_merged"
    existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
    existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
    if existing_gt > 0 and existing_gt == existing_im:
        print(f"{out_root} already contains {existing_gt} pairs; skipping merge.")
        return existing_gt
    dirs = discover_him2k_dirs(RAW / "him2k_raw")
    if dirs is None:
        print("HIM2K images/alphas directory pair not found — general fg will be SKIPPED.")
        return 0
    n = merge_him2k(dirs[0], dirs[1], out_root)
    print(f"HIM2K merged: {n} pairs -> {out_root}")
    return n


def _download_toonout() -> int:
    """Source: v4/v7 cell::_download_toonout (copy, ToonOut tar structure fix
    INCLUDED): downloads ONLY the train split (data/train_*.tar archives; the
    test split is reserved for the illustration benchmark) and normalizes it
    into `/content/downloads/toonout/{im,gt}`. Idempotent."""
    from huggingface_hub import snapshot_download

    out_im = TOONOUT_DIR / "im"
    out_gt = TOONOUT_DIR / "gt"
    existing_im = len(list(out_im.iterdir())) if out_im.exists() else 0
    existing_gt = len(list(out_gt.iterdir())) if out_gt.exists() else 0
    if existing_im > 0 and existing_im == existing_gt:
        print(f"toonout: {TOONOUT_DIR} already contains {existing_im} pairs; skipping download.")
        return existing_im

    snap = Path(snapshot_download(repo_id=TOONOUT_HF_REPO, repo_type="dataset",
                                  allow_patterns=["data/train_*.tar"]))
    tars = sorted((snap / "data").glob("train_*.tar"))
    assert tars, (
        f"No data/train_*.tar found in the ToonOut snapshot: {snap} — the repo structure "
        f"may have changed (expected: data/train_generations_*.tar archives)."
    )
    extract_root = TOONOUT_DIR / "_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(extract_root, filter="data")

    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    copied = 0
    for gen_dir in sorted(p for p in extract_root.iterdir() if p.is_dir()):
        src_im, src_gt = gen_dir / "im", gen_dir / "gt"
        if not (src_im.is_dir() and src_gt.is_dir()):
            continue
        # macOS AppleDouble leftovers (`._*`) are not images — filter them out.
        gt_by_stem = {p.stem: p for p in src_gt.iterdir()
                      if p.is_file() and not p.name.startswith("._")}
        for img in sorted(p for p in src_im.iterdir()
                          if p.is_file() and not p.name.startswith("._")):
            gt = gt_by_stem.get(img.stem)
            if gt is None:
                continue  # an image without gt cannot be a source
            # name collisions are possible across generation folders -> prefix
            stem = f"{gen_dir.name}_{img.stem}"
            dst_i = out_im / f"{stem}{img.suffix}"
            dst_g = out_gt / f"{stem}{gt.suffix}"
            if dst_i.exists() and dst_g.exists():
                copied += 1
                continue
            shutil.copy2(img, dst_i)
            shutil.copy2(gt, dst_g)
            copied += 1
    shutil.rmtree(extract_root, ignore_errors=True)
    assert copied > 0, "No im/gt pairs could be extracted from the ToonOut train tars."
    print(f"toonout (train split): {copied} im/gt pairs -> {TOONOUT_DIR} (test split NOT TOUCHED).")
    return copied


def stage_downloads() -> dict:
    report("downloads", "running")
    RAW.mkdir(parents=True, exist_ok=True)
    source_defs = _load_source_defs()
    results: dict = {}

    try:
        results["trans460"] = _download_trans460(source_defs)
    except Exception as e:
        print(f"WARNING: transparent_460 could not be downloaded ({e}); the on-disk copy will be used if present.")
        results["trans460"] = -1
    results["trans460_pairs"] = _normalize_trans460_pairs()

    results["him2k_merged"] = _ensure_him2k_merged(source_defs)
    results["toonout"] = _download_toonout()

    # At least one fg source is REQUIRED (background/text/decor are synthetic, but not the subject).
    assert (results["trans460_pairs"] > 0 or results["him2k_merged"] > 0
            or results["toonout"] > 0), (
        "No fg source could be prepared (trans460_pairs / him2k_merged / toonout) — "
        "design generation cannot be subject-less; inspect the download logs."
    )
    report("downloads", "done", results=results)
    return results


# ==========================================================================
# Stage "fonts" — v4/v7 cell::stage_fonts (copy): ~20 OFL TTFs from the
# Google Fonts repo -> /content/fonts; DejaVu fallback if none come down.
# ==========================================================================
_GF_RAW = "https://raw.githubusercontent.com/google/fonts/main/"
GOOGLE_FONT_PATHS = [
    "ofl/anton/Anton-Regular.ttf",
    "ofl/bebasneue/BebasNeue-Regular.ttf",
    "ofl/lobster/Lobster-Regular.ttf",
    "ofl/pacifico/Pacifico-Regular.ttf",
    "ofl/permanentmarker/PermanentMarker-Regular.ttf",
    "ofl/bangers/Bangers-Regular.ttf",
    "ofl/righteous/Righteous-Regular.ttf",
    "ofl/satisfy/Satisfy-Regular.ttf",
    "ofl/abrilfatface/AbrilFatface-Regular.ttf",
    "ofl/alfaslabone/AlfaSlabOne-Regular.ttf",
    "ofl/archivoblack/ArchivoBlack-Regular.ttf",
    "ofl/shrikhand/Shrikhand-Regular.ttf",
    "ofl/staatliches/Staatliches-Regular.ttf",
    "ofl/monoton/Monoton-Regular.ttf",
    "ofl/pressstart2p/PressStart2P-Regular.ttf",
    "ofl/caveat/Caveat[wght].ttf",
    "ofl/dancingscript/DancingScript[wght].ttf",
    "ofl/oswald/Oswald[wght].ttf",
    "ofl/montserrat/Montserrat[wght].ttf",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/orbitron/Orbitron[wght].ttf",
]
DEJAVU_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVu*.ttf",  # standard Colab/Ubuntu path
    "/usr/share/fonts/TTF/DejaVu*.ttf",
]


def stage_fonts() -> int:
    report("fonts", "running")
    FONT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for rel in GOOGLE_FONT_PATHS:
        # The [wght] square brackets in the file name must be percent-encoded
        # in the URL; locally a plain bracket-free name (so it does not clash
        # with glob patterns).
        target = FONT_DIR / Path(rel).name.replace("[", "_").replace("]", "_")
        if target.exists() and target.stat().st_size > 0:
            ok += 1
            continue
        url = _GF_RAW + urllib.parse.quote(rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            assert data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), "not a TTF/OTF signature"
            target.write_bytes(data)
            ok += 1
        except Exception as e:
            failed.append((rel, str(e)))
            print(f"WARNING: font could not be downloaded ({rel}): {e}")

    if ok < 5:
        print(f"Only {ok} Google Fonts fonts could be downloaded — falling back to DejaVu.")
        import glob as _glob

        for pattern in DEJAVU_GLOBS:
            for p in _glob.glob(pattern):
                dst = FONT_DIR / Path(p).name
                if not dst.exists():
                    shutil.copy2(p, dst)

    total = len([p for p in FONT_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"}])
    print(f"/content/fonts: {total} fonts ready ({ok} Google Fonts, {len(failed)} failed).")
    if total == 0:
        raise RuntimeError(
            "No fonts could be downloaded and the DejaVu fallback was not found either — design "
            "text blocks cannot be generated. Check the network connection or place TTFs "
            "manually into /content/fonts."
        )
    report("fonts", "done", downloaded=ok, failed=len(failed), total=total)
    return total


# ==========================================================================
# Stage "val_guard" — VAL leak guard at fg SELECTION for design (v7 cell
# copy): val_stems.json -> tcl.derive_val_excluded_source_ids -> mapping back
# to raw fg stems through the composite id contract.
# ==========================================================================
def stage_val_guard() -> set[str]:
    report("val_guard", "running")
    val_json = STATUS_DIR / "val_stems.json"
    if not val_json.exists():
        print(f"NOTE: {val_json} missing (no training may have run yet) — skipping VAL "
              f"exclusion; design stems are overwritten in TRAIN anyway.")
        report("val_guard", "done", excluded=0)
        return set()

    from build_testset import _sanitize  # scripts/ is on sys.path (build_trainset id contract)

    val_stems = json.loads(val_json.read_text())["val_stems"]
    excluded_ids, unmatched = tcl.derive_val_excluded_source_ids(val_stems)
    if unmatched:
        print(f"NOTE: {len(unmatched)} val stems do not match the copy-suffix pattern "
              f"(e.g. {unmatched[:5]}) — expected for non-composite stems.")

    exclude_fg_stems: set[str] = set()
    per_pool: dict[str, int] = {}
    for pool_dirname, prefix in FG_SOURCE_PREFIXES.items():
        im_dir = RAW / pool_dirname / "im"
        if not im_dir.is_dir():
            continue
        n = 0
        for p in im_dir.iterdir():
            if not p.is_file() or p.name.startswith("._"):
                continue
            if f"{prefix}_{_sanitize(p.stem)}" in excluded_ids:
                exclude_fg_stems.add(p.stem)
                n += 1
        per_pool[pool_dirname] = n
    print(f"VAL leak guard: {len(val_stems)} val stems -> {len(exclude_fg_stems)} "
          f"raw fg stems will be excluded from the pool (per pool: {per_pool}).")
    report("val_guard", "done", excluded=len(exclude_fg_stems), per_pool=per_pool)
    return exclude_fg_stems


# ==========================================================================
# Stage "design" — REGENERATION with the FIXED generator: scripts/
# make_design.py (unit tested). SAME seed + SAME count as the v7 run -> the
# SAME design_NNNNN_c00 ids come out with corrected renders; the Drive merge
# then overwrites the old files in place.
# ==========================================================================
def stage_design_regen(exclude_fg_stems: set[str]) -> dict[str, int]:
    report("design", "running")

    if "design" not in CATEGORIES:
        raise RuntimeError(
            f"benchmark.testset.CATEGORIES does not know the 'design' category — your repo "
            f"clone looks stale. Run 'git -C {WORKDIR} pull' and re-run the cell."
        )

    try:
        import make_design as mdz  # scripts/ is on sys.path
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_design.py could not be imported ({e}) — is your repo up-to-date? "
            f"Try 'git -C {WORKDIR} pull'."
        ) from e

    # The whole point of the regeneration is the v8 glow fix — refuse to
    # regenerate with a stale generator (it would silently reproduce the old
    # canvas-wide glow tails).
    if not hasattr(mdz, "_zero_alpha_in_rects"):
        raise RuntimeError(
            "scripts/make_design.py does NOT contain the v8 glow fix "
            "(_zero_alpha_in_rects missing) — your repo clone is stale. Run "
            f"'git -C {WORKDIR} pull' and re-run the cell."
        )

    fg_dirs = [d for d in (TRANS460_PAIRS, RAW / "him2k_merged") if (d / "im").is_dir()]
    toon_dir = TOONOUT_DIR if (TOONOUT_DIR / "im").is_dir() else None

    counts = mdz.run(
        out_dir=DESIGN_OUT_DIR,
        bg_dir=None,  # background is synthetic — not used
        fg_dirs=fg_dirs,
        toonout_dir=toon_dir,
        font_dir=FONT_DIR,
        seed=SEED,
        count=DESIGN_COUNT,
        exclude_fg_stems=exclude_fg_stems,
    )
    print("make_design.run() production:", counts)

    out_manifest = DESIGN_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} missing — make_design generation must have failed.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"{out_manifest} is empty — not proceeding to export (lesson from v3).")

    full_manifest = DESIGN_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = DESIGN_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = DESIGN_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"file missing for manifest row: {r['id']} — generation may have been cut short.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")

    print(f"PRE-FLIGHT — {out_manifest}: {len(rows)} regenerated design pairs.")
    report("design", "done", counts=counts, total_pairs=len(rows))
    return counts


# ==========================================================================
# Stage "export" — v4/v6/v7 pattern: export_birefnet.export() into
# fresh/empty local directories, split_name="TRAIN".
# ==========================================================================
def stage_export_all() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ is on sys.path

    stats_bokeh = eb.export(
        manifest_path=str(BOKEH_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR_BOKEH,
        split_name="TRAIN",
    )
    stats_design = eb.export(
        manifest_path=str(DESIGN_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR_DESIGN,
        split_name="TRAIN",
    )
    print(json.dumps({"bokeh": stats_bokeh, "design": stats_design}, ensure_ascii=False, indent=2))
    report("export", "done", bokeh=stats_bokeh, design=stats_design)
    return {"bokeh": stats_bokeh, "design": stats_design}


# ==========================================================================
# Stage "drive_copy" — v6/v7 pattern with ONE v8 difference: the bokeh merge
# ADDS new stems (growth ~9k), the design merge OVERWRITES existing stems
# (growth 0 — the whole point; copytree with dirs_exist_ok=True replaces the
# file contents in place). im/gt are counted SEPARATELY (2026-07-12 lesson).
# ==========================================================================
def _merge_into_drive_train(src: Path, expect_overwrite: bool, manifest_local: Path) -> dict:
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Expected TRAIN data not found on Drive: {dst_train_im} / {dst_train_gt} — "
        f"this cell only updates an EXISTING dataset."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    assert len(src_im_files) == len(src_gt_files), f"im/gt counts do not match in {src}!"

    existing_dst_im_stems = {p.stem for p in _listdir_retry(dst_train_im)}
    existing_dst_gt_stems = {p.stem for p in _listdir_retry(dst_train_gt)}
    growth_im = len({p.stem for p in src_im_files} - existing_dst_im_stems)
    growth_gt = len({p.stem for p in src_gt_files} - existing_dst_gt_stems)
    n_overwrite = len({p.stem for p in src_im_files} & existing_dst_im_stems)

    if expect_overwrite and growth_im:
        print(f"NOTE: {growth_im} design stems are NEW on Drive (expected pure overwrite) — "
              f"fine on a first run against a dataset that never had them.")
    pre_im, pre_gt = len(existing_dst_im_stems), len(existing_dst_gt_stems)
    print(f"Drive TRAIN before merge: im={pre_im}, gt={pre_gt} — expected growth: "
          f"im +{growth_im}, gt +{growth_gt}; overwrites: {n_overwrite}")

    # ONLY the TRAIN/ subtree is copied — the PARTIAL stats.json at the src
    # root is NOT copied so it does not clobber the authoritative FULL
    # stats.json on Drive (v3 fix). tcl.copy_pairs instead of
    # shutil.copytree (2026-07-21 lesson, live in THIS run): single-threaded
    # copytree overwrites crawled on Drive FUSE (the 6k design overwrite sat
    # >1.5h with no output), while copy_pairs is threaded x16, size-checked
    # (skips up-to-date pairs -> an interrupted merge RESUMES instead of
    # re-copying everything) and prints progress.
    stems = sorted(p.stem for p in src_im_files)
    print(f"Copying (MERGE, TRAIN/ only, copy_pairs x16): {src / 'TRAIN'} -> {dst / 'TRAIN'}")
    n_copied = tcl.copy_pairs(
        stems, src / "TRAIN" / "im", src / "TRAIN" / "gt", dst_train_im, dst_train_gt
    )
    print(f"copy_pairs: {n_copied} pairs copied/repaired, {len(stems) - n_copied} already up to date.")

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

    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} new rows appended "
          f"({'0 expected for the design overwrite' if expect_overwrite else 'existing rows PRESERVED'}).")

    return {"added_im": growth_im, "added_gt": growth_gt, "overwritten": n_overwrite,
            "added_manifest_rows": n_appended, "total_im": post_im, "total_gt": post_gt}


def stage_drive_copy() -> None:
    report("drive_copy", "running")
    res_bokeh = _merge_into_drive_train(
        Path(EXPORT_DIR_BOKEH), expect_overwrite=False,
        manifest_local=BOKEH_OUT_DIR / "manifest_full.jsonl",
    )
    res_design = _merge_into_drive_train(
        Path(EXPORT_DIR_DESIGN), expect_overwrite=True,
        manifest_local=DESIGN_OUT_DIR / "manifest_full.jsonl",
    )
    print("\nINTEGRITY CHECK PASSED — v8 (bokeh + regenerated design) data on Drive.")
    report("drive_copy", "done", bokeh=res_bokeh, design=res_design)


# ==========================================================================
# Stage "refresh_marker" — records the overwritten design stems so that
# train_colab.ipynb cell (c) size-revalidates their LOCAL copies against
# Drive after the tar extraction (they may sit stale inside future repacks).
# Merged as a union with any existing marker (accumulates across versions).
# ==========================================================================
def stage_refresh_marker() -> None:
    report("refresh_marker", "running")
    manifest_full = DESIGN_OUT_DIR / "manifest_full.jsonl"
    stems = sorted(
        json.loads(line)["id"]
        for line in manifest_full.read_text().splitlines() if line.strip()
    )
    marker = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR / TAR_SUBDIR / "_refresh_stems.json"
    existing: list[str] = []
    if marker.exists():
        try:
            existing = json.loads(marker.read_text()).get("stems", [])
        except Exception:
            existing = []
    merged = sorted(set(existing) | set(stems))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(
        {"stems": merged, "written": _now(),
         "reason": "v8 design regeneration — local copies must be size-revalidated against Drive"},
        ensure_ascii=False,
    ))
    print(f"{marker}: {len(merged)} refresh stems recorded ({len(stems)} from this run).")
    report("refresh_marker", "done", stems=len(merged))


# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
def main() -> None:
    stage0_env()                                    # Drive mount + git pull — before everything
    stage_tar_fetch()                               # tar shards -> local TRAIN (bokeh source)
    category_by_stem, exclude_stems = stage_categories()
    stage_bokeh(category_by_stem, exclude_stems)    # make_bokeh_copies.run() + manifest guard
    stage_downloads()                               # trans460 + HIM2K + ToonOut (~3GB, design fg)
    stage_fonts()                                   # ~20 OFL Google Fonts (DejaVu fallback)
    exclude_fg_stems = stage_val_guard()
    stage_design_regen(exclude_fg_stems)            # make_design.run() with the FIXED generator
    stage_export_all()
    stage_drive_copy()
    stage_refresh_marker()
    report("ALL", "done")
    print(
        "\nNOTE: the tar shards were NOT REPACKED — on the next training run, "
        "train_colab.ipynb cell (c) will fill in the new ~9k bokeh pairs as a delta from Drive "
        "via copy_pairs, and the regenerated design pairs are size-revalidated through "
        "tar/_refresh_stems.json (both take a few minutes)."
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
