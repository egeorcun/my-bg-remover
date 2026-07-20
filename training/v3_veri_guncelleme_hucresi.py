"""V3 DATA UPDATE CELL — in a fresh (FREE, CPU is enough — NO GPU NEEDED)
Colab session, adds ONLY the NEW `_o00` (original background) copies to the
existing Drive dataset (`bg-remover-data/TRAIN`); it does NOT REGENERATE
v1/v2's full ~28k composite set (see the "v3" note in the
`scripts/make_composites.py` module docstring — the reason over-deletion was
persistent is that categories other than camouflage were trained only on
synthetic backgrounds; this cell closes the domain gap by adding one extra
copy per category that preserves the original background).

SOURCE / ATTRIBUTION: the env/manifest stages of this file (`report`,
`stage0_env_sanity`, `_walk_dirs`, `discover_cod10k`, `discover_him2k_dirs`,
`merge_him2k`, manifest construction) were COPIED VERBATIM from
`training/colab_devam_hucresi.py`; the raw-source download logic of the
"downloads" stage (`_download_hf_parquet_pairs` — including cumulative-counter
stems, P3M zip extraction, the Transparent-460 `snapshot_download`,
COD10K/HIM2K via gdown, the BG-20k background pool) was replicated from
`training/prepare_data_colab.ipynb` cells (c) [8-11] and (e) [15] (PROVEN
code that ran in the live Phase 2 run). To prevent drift, those files remain
the single source of truth — the only reason it is rewritten here is that
this cell, like `colab_devam_hucresi.py`, must be paste-and-run standalone
rather than imported as a module. The export/drive_copy stages were REWRITTEN
with v3-specific MERGE logic (see below).

PREREQUISITES (lesson from the live run — a FRESH VM has NO raw data, this
cell downloads all of it itself): only the repo must be cloned at
`/content/my-bg-remover` with `pip install -e .` done (for the imports at the
top). Mounting Drive (`drive.mount`) happens in this cell's OWN env stage,
BEFORE anything that touches DRIVE_ROOT (status reporting, reading
val_stems.json, the final merge). Raw sources (dis5k/camo/p3m/trans460_train/
cod10k/him2k + the BG-20k background pool) are downloaded idempotently in the
"downloads" stage; the manifest is rebuilt DETERMINISTICALLY from this raw
data, so the `id`s in `data/train/manifest.jsonl` come out EXACTLY identical
to the earlier v1/v2 runs (same source data + same `build_trainset.py` logic,
same ordering). On Drive, `bg-remover-data/TRAIN/{im,gt}` (the full composite
output of v1/v2) and `bg-remover-status/val_stems.json` (the VAL split) must
ALREADY exist.

DIFFERENCES (from v1/v2's `colab_devam_hucresi.py`):
1. After Stage 4, `val_stems.json` on Drive is read, the SOURCE ids that must
   not leak into VAL are derived (by stripping the `_v<NN>`/`_o<NN>` suffix —
   see `training.train_colab_lib.strip_composite_copy_suffix`) and passed to
   `make_composites.run()` as `exclude_source_ids`.
2. On a fresh VM, `data/train_composites/` (v1/v2's full ~28k composite
   output) does NOT exist — regenerating it takes hours. Instead, using
   `run()`'s `only_original_bg=True` flag, ONLY the `_o00` set is produced
   into a SEPARATE directory (`data/train_composites_o/`) (~14k or so, fast —
   a matter of minutes even on CPU, NO compose, only augment).
3. Because `export_birefnet.export()` runs against this SEPARATE (fresh,
   empty) local directory, only `_o00` files appear on disk (idempotent
   skip-existing already exists in `export_birefnet.py` — nothing extra is
   needed here, the source manifest already contains only `_o00` rows).
4. Copying to Drive is ONLY a `shutil.copytree(..., dirs_exist_ok=True)`
   MERGE of the `TRAIN/` subtree (existing `_v<NN>` files are NOT
   DELETED/OVERWRITTEN, only new `_o00` files are added; the PARTIAL —
   `_o00`-only — `stats.json` at the src root is NOT COPIED so it cannot
   clobber the authoritative FULL stats.json on Drive) + APPENDING only the
   NEW ids to the Drive copy of the composite manifest
   (`train_composites_manifest.jsonl`) (NOT a full overwrite — the
   continuation cell's full overwrite via `shutil.copy2` would be WRONG here,
   the file on Drive already contains all of v1/v2's `_v<NN>` rows).
   Integrity check: the INCREASE in the file count of Drive TRAIN must equal
   exactly the number of locally produced `_o00` files not yet on Drive.

Status tracking is the SAME mechanism as `colab_devam_hucresi.py`
(`report()` -> `bg-remover-status/log.txt` + `status.json`) — stages: env,
downloads, manifest, composites_o, export, drive_copy, (at the end) ALL.
"""

import io
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
# academic datasets, the limit is removed (see the same line in
# colab_devam_hucresi.py).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS must come AFTER the PIL import/assignment)
from PIL import Image  # noqa: E402

# --- Constants (SAME as colab_devam_hucresi.py — see that file's "Constants" section) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
SEED = 42
BG_POOL_SIZE = 5000

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"
VAL_STEMS_PATH = STATUS_DIR / "val_stems.json"

# scripts/ is not a package — we add the absolute path to sys.path to be able
# to import build_trainset/make_composites/export_birefnet (see colab_devam_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries, load_manifest  # noqa: E402  (package installed via pip install -e .)
import training.train_colab_lib as tcl  # noqa: E402  (same pip install -e . -- torch-free, testable logic)


# ==========================================================================
# Status reporting — EXACTLY IDENTICAL to `colab_devam_hucresi.py::report`
# (source: training/colab_devam_hucresi.py, line ~71).
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
# Stage "env" — Drive mount + environment sanity check (source:
# colab_devam_hucresi.py::stage0_env_sanity + prepare_data_colab.ipynb cell (a);
# lesson from the live run: without Drive mounted, ALL of
# report()/val_stems.json/the final merge silently ran without Drive and blew
# up at the very end — mount is now the FIRST thing done).
# ==========================================================================
RAW_DIR_CHECKS = {
    "dis5k": "data/raw_train/dis5k/im",
    "camo": "data/raw_train/camo/im",
    "p3m": "data/raw_train/p3m/im",
    "trans460_train": "data/raw_train/trans460_train/fg",
    "cod10k_raw": "data/raw_train/cod10k_raw",
    "him2k_raw": "data/raw_train/him2k_raw",
    "backgrounds": "data/backgrounds",
}


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _setup_hf_env() -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF_TOKEN obtained from Colab Secrets.")
    except Exception as e:
        print(f"Could not obtain HF_TOKEN (not in Secrets or access not granted): {e}")


def stage0_env_sanity() -> dict:
    # Drive is mounted BEFORE EVERYTHING ELSE (including report() — STATUS_DIR
    # lives on Drive!): in the live run, because Drive was not mounted,
    # val_stems.json was mistakenly considered "not found" and the final merge
    # would have blown up. drive.mount is idempotent (if already mounted it
    # says "already mounted" and does not raise). Source:
    # prepare_data_colab.ipynb cell (a).
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Could not mount Drive: {DRIVE_ROOT} does not exist"

    report("env", "running")
    os.chdir(WORKDIR)
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} files")
    for name in ("dis5k", "camo", "p3m", "trans460_train", "cod10k_raw", "him2k_raw"):
        if counts[name] == 0:
            print(f"NOTE: {name} is currently empty — it will be downloaded in the "
                  f"'downloads' stage (normal on a fresh VM, see the module "
                  f"docstring PREREQUISITES).")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage "downloads" — ALL raw sources + background pool, IDEMPOTENT.
# Lesson from the live run: a fresh VM has NO raw data; without this stage the
# manifest was built with 0 pairs and the pipeline blew up at export. The
# download logic was replicated from PROVEN code:
#   - HF parquet pairs (dis5k_tr/camo_tr, cumulative-counter stems +
#     integrity threshold): prepare_data_colab.ipynb cell (c)/9.
#   - P3M zip + Transparent-460 snapshot: same notebook cell (c)/10.
#   - COD10K/HIM2K gdown: same notebook cell (c)/11 (AM-2k is DELIBERATELY
#     skipped: the manifest never uses it — see build_trainset.SOURCE_SPECS +
#     cod10ktr/him2k; no point downloading ~GBs for nothing).
#   - BG-20k background pool: colab_devam_hucresi.py::stage1_bg_pool
#     (originating from prepare_data_colab.ipynb cell (e)/15).
# ==========================================================================
RAW = Path("data/raw_train")


def _load_source_defs() -> dict:
    with open("data/train_sources.json") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _sanitize_stem(name) -> str:
    """Source: prepare_data_colab.ipynb cell 9 (_sanitize_stem) — converts the
    file name in the parquet to a safe stem."""
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).stem)


def _download_hf_parquet_pairs(source_defs: dict, source_name: str, img_col: str,
                               mask_col: str, out_subdir: str) -> int:
    """Source: prepare_data_colab.ipynb cell 9 — reads ALL parquet shards of
    source_name and writes the (image, mask) pairs under RAW/out_subdir/{im,gt}/.
    Stem strategy (COLLISION PREVENTION): from the `image_name` column if it
    exists; otherwise a CUMULATIVE counter increasing across ALL shards — an
    index reset per shard would, from the 2nd shard onward, collide with the
    1st shard's stems and cause rows to be silently skipped. Idempotent:
    existing pairs are skipped."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    spec = source_defs[source_name]
    repo = spec["hf_repo"]
    out_im = RAW / out_subdir / "im"
    out_gt = RAW / out_subdir / "gt"
    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    def _bytes_of(cell_value):
        return cell_value["bytes"] if isinstance(cell_value, dict) else cell_value

    written = 0
    counter = 0  # cumulative row counter — NOT RESET at shard boundaries
    for pattern in spec["split_patterns"]:
        paths = fs.glob(f"datasets/{repo}/{pattern}")
        for p in sorted(paths):
            print(f"  reading: {p}")
            with fs.open(p, "rb") as fh:
                schema_names = pq.read_schema(fh).names
                fh.seek(0)
                has_name = "image_name" in schema_names
                columns = (["image_name"] if has_name else []) + [img_col, mask_col]
                table = pq.read_table(fh, columns=columns)
            for i in range(table.num_rows):
                if has_name:
                    stem = f"{source_name}_{_sanitize_stem(table['image_name'][i].as_py())}"
                else:
                    stem = f"{source_name}_{counter:06d}"
                counter += 1
                out_img_path = out_im / f"{stem}.jpg"
                out_gt_path = out_gt / f"{stem}.png"
                if out_img_path.exists() and out_gt_path.exists():
                    continue  # idempotent (sorted() -> stable shard order, deterministic stems)
                img_bytes = _bytes_of(table[img_col][i].as_py())
                mask_bytes = _bytes_of(table[mask_col][i].as_py())
                Image.open(io.BytesIO(img_bytes)).convert("RGB").save(out_img_path, quality=95)
                Image.open(io.BytesIO(mask_bytes)).convert("L").save(out_gt_path)
                written += 1

    total_pairs = len(list(out_im.glob("*")))
    expected = spec.get("full_pair_count")
    print(f"{source_name}: {written} new pairs written; {total_pairs} total on disk (expected ~{expected})")
    if expected and total_pairs < 0.9 * expected:
        raise RuntimeError(
            f"{source_name}: only {total_pairs}/{expected} pairs on disk (<90%) — "
            f"could be a stem collision, a missing parquet shard, or a changed repo schema."
        )
    return written


def _download_p3m(source_defs: dict) -> int:
    """Source: prepare_data_colab.ipynb cell 10 (P3M section). Idempotent: if
    >= 90% of the pairs are already on disk, the zip is never downloaded (fast
    skip); otherwise hf_hub_download (with its own cache) + per-file
    target.exists() skipping."""
    import zipfile

    from huggingface_hub import hf_hub_download

    spec = source_defs["p3m_10k_train"]
    p3m_out_im = RAW / "p3m" / "im"
    p3m_out_gt = RAW / "p3m" / "gt"
    existing = len(list(p3m_out_im.iterdir())) if p3m_out_im.exists() else 0
    expected = spec.get("full_pair_count") or 0
    if expected and existing >= 0.9 * expected:
        print(f"p3m: already {existing} pairs on disk (>= 90% x {expected}); skipping download.")
        return existing

    p3m_zip = hf_hub_download(repo_id=spec["hf_repo"], repo_type="dataset", filename="data/p3m10k.zip")
    p3m_out_im.mkdir(parents=True, exist_ok=True)
    p3m_out_gt.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p3m_zip) as zf:
        names = [n for n in zf.namelist() if "/train/blurred_image/" in n or "/train/mask/" in n]
        for n in names:
            if n.endswith("/"):
                continue
            target = (p3m_out_im if "/blurred_image/" in n else p3m_out_gt) / Path(n).name
            if target.exists():
                continue
            with zf.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
    total = len(list(p3m_out_im.iterdir()))
    print(f"p3m_10k_train: {total} images -> {p3m_out_im.parent}")
    return total


def _download_trans460(source_defs: dict) -> int:
    """Source: prepare_data_colab.ipynb cell 10 (Transparent-460 section).
    Idempotency ADDITION: if fg/ is already >= 90% full, the snapshot is never
    fetched (the original cell did rmtree+copytree on every run — no
    difference on a fresh VM, but it avoids needless work on a re-run)."""
    from huggingface_hub import snapshot_download

    spec = source_defs["transparent_460_train"]
    trans_out = RAW / "trans460_train"
    existing = len(list((trans_out / "fg").iterdir())) if (trans_out / "fg").exists() else 0
    expected = spec.get("full_pair_count") or 0
    if expected and existing >= 0.9 * expected:
        print(f"trans460_train: already {existing} images on disk (>= 90% x {expected}); skipping download.")
        return existing

    trans_dir = snapshot_download(repo_id=spec["hf_repo"], repo_type="dataset", allow_patterns=["Train/*"])
    if trans_out.exists():
        shutil.rmtree(trans_out)
    shutil.copytree(Path(trans_dir) / "Train" / "fg", trans_out / "fg")
    shutil.copytree(Path(trans_dir) / "Train" / "alpha", trans_out / "alpha")
    total = len(list((trans_out / "fg").iterdir()))
    print(f"transparent_460_train: {total} images -> {trans_out}")
    return total


def _gdown_extract(drive_id: str, out_dir: Path, label: str) -> bool:
    """Source: prepare_data_colab.ipynb cell 11 — downloads a zip from the
    Drive id and extracts it to out_dir; returns False on failure (does not
    stop the pipeline, the caller proceeds with a note). Idempotency ADDITION:
    if out_dir is already populated, the download is skipped."""
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


def _ensure_gdown() -> None:
    """Installs gdown if not already installed via pip (a dev dependency of
    the repo — `pip install -e .` does not bring it; the paste-run equivalent
    of the `!pip install gdown -q` line in prepare_data_colab.ipynb cell 8)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)


def _download_bg_pool(source_defs: dict) -> int:
    """Source: colab_devam_hucresi.py::stage1_bg_pool (originating from
    prepare_data_colab.ipynb cell (e)/15) — BG_POOL_SIZE backgrounds from
    BG-20k, idempotent via a cumulative counter."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    bg_dir = Path("data/backgrounds")
    bg_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(bg_dir.iterdir()))
    if existing >= BG_POOL_SIZE:
        print(f"data/backgrounds already contains {existing} images (>= {BG_POOL_SIZE}); skipping download.")
        return existing

    bg_spec = source_defs["bg_20k"]
    fs = HfFileSystem()
    pattern = bg_spec["split_patterns"][0]
    parts = sorted(fs.glob(f"datasets/{bg_spec['hf_repo']}/{pattern}"))

    written = existing  # CUMULATIVE counter — not reset at shard boundaries
    for part in parts:
        if written >= BG_POOL_SIZE:
            break
        with fs.open(part, "rb") as fh:
            table = pq.read_table(fh, columns=["image"])
        for i in range(table.num_rows):
            if written >= BG_POOL_SIZE:
                break
            out_path = bg_dir / f"bg20k_{written:06d}.jpg"
            if out_path.exists():
                written += 1
                continue
            img_bytes = table["image"][i].as_py()["bytes"]
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(out_path, format="JPEG", quality=88)
            written += 1

    print(f"data/backgrounds: {written} background images.")
    return written


def stage_downloads() -> dict:
    report("downloads", "running")
    RAW.mkdir(parents=True, exist_ok=True)
    source_defs = _load_source_defs()
    results: dict = {}

    # HF parquet pairs — column names were verified in Phase 2 (cell 9 note);
    # if one source fails the others are still attempted (the try/except
    # pattern from cell 9), a category left missing is skipped in the manifest
    # + the empty-manifest guard at the very end stops loudly if the total is
    # zero.
    try:
        results["dis5k"] = _download_hf_parquet_pairs(source_defs, "dis5k_tr", "image", "label", "dis5k")
    except Exception as e:
        print(f"WARNING: dis5k_tr could not be downloaded ({e}); data/raw_train/dis5k will be used if present.")
        results["dis5k"] = -1
    try:
        results["camo"] = _download_hf_parquet_pairs(source_defs, "camo_tr", "image", "mask", "camo")
    except Exception as e:
        print(f"WARNING: camo_tr could not be downloaded ({e}); data/raw_train/camo will be used if present.")
        results["camo"] = -1

    try:
        results["p3m"] = _download_p3m(source_defs)
    except Exception as e:
        print(f"WARNING: p3m could not be downloaded ({e}); the on-disk copy will be used if present.")
        results["p3m"] = -1
    try:
        results["trans460"] = _download_trans460(source_defs)
    except Exception as e:
        print(f"WARNING: transparent_460 could not be downloaded ({e}); the on-disk copy will be used if present.")
        results["trans460"] = -1

    # Google Drive sources (gdown) — cod10k matters for camouflage; him2k is
    # the general category (optional, but v1/v2 data included it, so it is
    # downloaded). AM-2k is DELIBERATELY skipped: the manifest does not use it
    # (see the stage comment).
    _ensure_gdown()
    results["cod10k"] = _gdown_extract(source_defs["cod10k_tr"]["drive_id"], RAW / "cod10k_raw", "COD10K-TR")
    results["him2k"] = _gdown_extract(source_defs["him2k"]["drive_id"], RAW / "him2k_raw", "HIM2K")

    results["backgrounds"] = _download_bg_pool(source_defs)

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    print("File counts after downloads:", counts)
    report("downloads", "done", results=results, counts=counts)
    return results


# ==========================================================================
# Stage "manifest" — COD10K/HIM2K discovery+merge + full manifest (source:
# colab_devam_hucresi.py::{discover_cod10k, stage2_discover_structure,
# discover_him2k_dirs, merge_him2k, stage3_merge_him2k, stage4_build_manifest}).
# Grouped under a single report("manifest", ...) pair (task item: report()
# stages are env/downloads/manifest/composites_o/export/drive_copy/ALL).
# ==========================================================================
def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
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


def discover_cod10k(raw_dir: Path) -> dict | None:
    if not raw_dir.exists():
        return None
    dirs = _walk_dirs(raw_dir, max_depth=4)
    img_candidates = [d for d in dirs if d["jpg_count"] >= 10]
    gt_candidates = [d for d in dirs if d["png_count"] >= 10]

    scored = []
    for ic in img_candidates:
        for gc in gt_candidates:
            if ic["path"] == gc["path"]:
                continue
            overlap = len(ic["jpg_stems"] & gc["png_stems"])
            if overlap == 0:
                continue
            name_bonus = 0
            if "image" in ic["path"].name.lower():
                name_bonus += 2
            if "gt" in gc["path"].name.lower():
                name_bonus += 2
            if "train" in str(ic["path"]).lower():
                name_bonus += 1
            scored.append({
                "img_dir": ic["path"], "gt_dir": gc["path"], "overlap": overlap,
                "score": (overlap, name_bonus),
            })
    if not scored:
        return None
    scored.sort(key=lambda s: s["score"], reverse=True)
    best = scored[0]
    ambiguous = len(scored) > 1 and scored[0]["score"] == scored[1]["score"]
    return {
        "img_dir": best["img_dir"], "gt_dir": best["gt_dir"], "overlap": best["overlap"],
        "ambiguous": ambiguous,
        "candidates": [(str(s["img_dir"]), str(s["gt_dir"]), s["overlap"]) for s in scored[:5]],
    }


def discover_him2k_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
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


def stage_manifest() -> dict:
    """COD10K discovery + HIM2K merge + full manifest construction — under a
    SINGLE `report("manifest", ...)` pair (see the module docstring)."""
    report("manifest", "running")
    import build_trainset as bt  # scripts/ is on sys.path

    # --- COD10K discovery (source: stage2_discover_structure) ---
    cod_raw_dir = Path("data/raw_train/cod10k_raw")
    cod10k_info = None
    if cod_raw_dir.exists():
        cod10k_info = discover_cod10k(cod_raw_dir)
        if cod10k_info:
            print(f"COD10K selected pair: img={cod10k_info['img_dir']}  gt={cod10k_info['gt_dir']}  "
                  f"overlapping stems={cod10k_info['overlap']}  ambiguous={cod10k_info['ambiguous']}")
        else:
            print("No overlapping img/gt directory pair found for COD10K.")
    else:
        print("data/raw_train/cod10k_raw does not exist — skipping COD10K.")

    # --- HIM2K merge (source: stage3_merge_him2k) ---
    him2k_raw_dir = Path("data/raw_train/him2k_raw")
    him2k_count = 0
    if him2k_raw_dir.exists():
        dirs = discover_him2k_dirs(him2k_raw_dir)
        if dirs is None:
            print("HIM2K images/alphas directory pair not found — skipping.")
        else:
            images_dir, alphas_dir = dirs
            out_root = Path("data/raw_train/him2k_merged")
            existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
            existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
            if existing_gt > 0 and existing_gt == existing_im:
                print(f"data/raw_train/him2k_merged already contains {existing_gt} pairs; skipping merge.")
                him2k_count = existing_gt
            else:
                him2k_count = merge_him2k(images_dir, alphas_dir, out_root)
                print(f"HIM2K merged: {him2k_count} pairs -> {out_root}")
    else:
        print("data/raw_train/him2k_raw does not exist — skipping HIM2K (the general category is optional).")

    # --- Full manifest (source: stage4_build_manifest) — DETERMINISTIC: same
    # raw data + same build_trainset.py logic -> ids EXACTLY identical to v1/v2. ---
    if bt.MANIFEST.exists():
        bt.MANIFEST.unlink()
    for d in (bt.OUT_IMG, bt.OUT_GT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts: dict = {}

    def _run(name: str, img_glob: str, gt_glob: str, category: str, **kw) -> int:
        rows = bt.sample_source(name, img_glob, gt_glob, category, n=None, copy=True, **kw)
        append_entries(str(bt.MANIFEST), rows)
        counts[name] = len(rows)
        print(f"{name} ({category}): {len(rows)} pairs")
        return len(rows)

    for name, spec in bt.SOURCE_SPECS.items():
        if spec["category"] == "disvd_tokens":
            continue
        _run(name, spec["img_glob"], spec["gt_glob"], spec["category"])

    rows = bt.sample_disvd_tokens("dis5ktr", bt.DIS5KTR_IMG_GLOB, bt.DIS5KTR_GT_GLOB, n=None, copy=True)
    append_entries(str(bt.MANIFEST), rows)
    dis_counts: dict = {}
    for r in rows:
        dis_counts[r["category"]] = dis_counts.get(r["category"], 0) + 1
    counts["dis5ktr"] = dis_counts
    for category, c in sorted(dis_counts.items()):
        print(f"dis5ktr ({category}): {c} pairs")

    if cod10k_info:
        root = Path(bt.ROOT).resolve()

        def _rel(p) -> str:
            rp = Path(p)
            if not rp.is_absolute():
                rp = root / rp
            return str(rp.resolve().relative_to(root))

        img_glob = _rel(cod10k_info["img_dir"]) + "/*"
        gt_glob = _rel(cod10k_info["gt_dir"]) + "/*"
        _run("cod10ktr", img_glob, gt_glob, "camouflage")
    else:
        counts["cod10ktr"] = 0
        print("cod10ktr: skipped (directory not found)")

    if him2k_count > 0:
        _run("him2k", "data/raw_train/him2k_merged/im/*", "data/raw_train/him2k_merged/gt/*", "general")
    else:
        counts["him2k"] = 0
        print("him2k: skipped (merge could not be performed)")

    # --- LOUD GUARD (lesson from the live run): if the manifest is built with
    # 0 pairs, NEVER continue — in that run, the export's FileNotFoundError was
    # only a SYMPTOM, the cause was an empty manifest. tcl.ensure_manifest_pairs
    # raises a RuntimeError with a clear message if the file is missing/empty
    # (torch-free, unit-tested). ---
    total_pairs = tcl.ensure_manifest_pairs(bt.MANIFEST)
    print(f"Manifest guard: {total_pairs} pairs with GT — continuing.")

    report("manifest", "done", counts=counts, total_pairs=total_pairs)
    return counts


# ==========================================================================
# Stage "composites_o" — NEW (v3-specific): produces only the _o00 (original
# background) copies, in a way that cannot leak into VAL (see the module
# docstring, items 1-2). The source-id derivation logic
# (`strip_composite_copy_suffix`/`derive_val_excluded_source_ids`) lives in
# `training.train_colab_lib` (torch-free, unit-tested) — see that module's
# "7) v3" section.
# ==========================================================================
def load_val_excluded_source_ids(val_stems_path: Path) -> tuple[set[str], list[str]]:
    """Reads `val_stems.json` from Drive (the `{"val_stems": [...]}` format
    written by `tcl.load_or_create_val_split`) and converts it via
    `tcl.derive_val_excluded_source_ids` into a `(source id set, unmatched
    stem list)` pair — the source ids are excluded from `_o00` production
    (VAL leak guard, see the task item "VAL leakage guard"); for the unmatched
    stems the guard is effectively BYPASSED (see the
    `tcl.strip_composite_copy_suffix` docstring), the caller should warn."""
    if not val_stems_path.exists():
        print(f"WARNING: {val_stems_path} not found — no sources are being excluded "
              f"(the VAL split may not have been made yet; in that case _o00 production "
              f"is applied to ALL categories, and the leak risk only applies in the "
              f"normal scenario where VAL_HOLDOUT ALREADY exists).")
        return set(), []
    payload = json.loads(val_stems_path.read_text())
    return tcl.derive_val_excluded_source_ids(payload.get("val_stems", []))


def stage_composites_o() -> dict:
    report("composites_o", "running")
    import make_composites as mc  # scripts/ is on sys.path

    excluded, unmatched = load_val_excluded_source_ids(VAL_STEMS_PATH)
    print(f"VAL leak guard: {len(excluded)} source ids excluded from _o00 production.")
    if unmatched:
        print("=" * 72)
        print(f"!!! WARNING — VAL LEAK GUARD PARTIALLY BYPASSED: {len(unmatched)} val stems "
              f"did NOT MATCH the _v<NN>/_o<NN> suffix pattern. The TRUE source ids of "
              f"these stems could NOT be excluded — the _o00 copies of those sources "
              f"will be produced into the training set and the same image will be seen "
              f"in both TRAIN and VAL (leak). First 10 unmatched stems: {unmatched[:10]}")
        print("=" * 72)
        report("composites_o", "warning", unmatched_val_stems=len(unmatched), sample=unmatched[:10])

    counts = mc.run(
        manifest_path=Path("data/train/manifest.jsonl"),
        backgrounds_dir=Path("data/backgrounds"),
        per_image=1,
        seed=SEED,
        out_dir=Path("data/train_composites_o"),
        exclude_source_ids=excluded,
        only_original_bg=True,
    )
    print("Per-category _o00 counts produced:", counts)

    # Integrity: expected total = (source rows outside NO_COMPOSE_CATEGORIES +
    # with gt_alpha + not excluded) x ORIGINAL_BG_COPIES (the formula is also
    # written to the report).
    source_rows = load_manifest("data/train/manifest.jsonl")
    eligible = [
        r for r in source_rows
        if r.get("gt_alpha") and r["category"] not in mc.NO_COMPOSE_CATEGORIES and r["id"] not in excluded
    ]
    expected_total = len(eligible) * mc.ORIGINAL_BG_COPIES

    out_manifest = Path("data/train_composites_o/manifest.jsonl")
    actual_total = len(load_manifest(str(out_manifest))) if out_manifest.exists() else 0
    print(f"composites_o integrity: expected={expected_total}, actual={actual_total} "
          f"(source rows x ORIGINAL_BG_COPIES={mc.ORIGINAL_BG_COPIES}).")
    assert actual_total == expected_total, (
        f"composites_o manifest total does not match the expectation: {actual_total} != {expected_total} "
        f"— the make_composites.run() logic or exclude_source_ids should be checked."
    )

    report("composites_o", "done", counts=counts, expected_total=expected_total, actual_total=actual_total)
    return counts


# ==========================================================================
# Stage "export" — NEW (v3-specific, but export_birefnet.export() is
# UNCHANGED): because it runs against a fresh/empty local directory, only
# _o00 files appear on disk (the source manifest already contains only _o00
# rows).
# ==========================================================================
def stage_export_o() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ is on sys.path

    stats = eb.export(
        manifest_path="data/train_composites_o/manifest.jsonl",
        out_dir="/content/birefnet_format_o",
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — NEW (v3-specific): MERGE into the existing Drive TRAIN
# (dirs_exist_ok=True, NO file is DELETED) + APPEND to the composite manifest
# (deduped, NO full overwrite — this is the biggest difference from the
# continuation cell). The merge logic lives in `tcl.merge_composite_manifest`
# (torch-free, unit-tested).
# ==========================================================================
def stage_drive_copy_o() -> None:
    report("drive_copy", "running")
    src = Path("/content/birefnet_format_o")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Expected v1/v2 TRAIN data not found on Drive: {dst_train_im} / {dst_train_gt} — "
        f"this cell is only for ADDING _o00 to an EXISTING dataset; to build a dataset "
        f"from scratch, colab_devam_hucresi.py must be used."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    existing_dst_im_stems = {p.stem for p in dst_train_im.iterdir()}
    new_stems = {p.stem for p in src_im_files} - existing_dst_im_stems
    expected_growth = len(new_stems)

    pre_im, pre_gt = len(list(dst_train_im.iterdir())), len(list(dst_train_gt.iterdir()))
    print(f"Drive TRAIN before merge: im={pre_im}, gt={pre_gt} — expected growth: {expected_growth}")

    # ONLY the TRAIN/ subtree is copied — the stats.json at the src root is
    # DELIBERATELY NOT COPIED: export_birefnet.export() wrote it with only the
    # PARTIAL statistics of the _o00 set; the stats.json on Drive is the
    # authoritative statistics of v1/v2's FULL dataset — copytree'ing the
    # entire src root would silently CLOBBER it (reviewer finding #1).
    print(f"Copying (MERGE, no deletion, TRAIN/ only): {src / 'TRAIN'} -> {dst / 'TRAIN'}")
    shutil.copytree(src / "TRAIN", dst / "TRAIN", dirs_exist_ok=True)

    post_im, post_gt = len(list(dst_train_im.iterdir())), len(list(dst_train_gt.iterdir()))
    print(f"Drive TRAIN after merge: im={post_im}, gt={post_gt}")

    assert post_im - pre_im == expected_growth, (
        f"im/ growth does not match the expectation: {post_im - pre_im} != {expected_growth}"
    )
    assert post_gt - pre_gt == expected_growth, (
        f"gt/ growth does not match the expectation: {post_gt - pre_gt} != {expected_growth}"
    )
    assert len(src_im_files) == len(src_gt_files), "im/gt counts do not match in the local _o00 export!"

    comp_manifest_local = Path("data/train_composites_o/manifest.jsonl")
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} new rows appended (the existing "
          f"v1/v2 rows on Drive were PRESERVED, not overwritten).")
    assert n_appended == expected_growth, (
        f"manifest append count ({n_appended}) is inconsistent with the file growth ({expected_growth}) — "
        f"the stem/id mapping should be checked."
    )

    print("\nINTEGRITY CHECK PASSED — _o00 data was MERGED into Drive.")
    report(
        "drive_copy", "done",
        added_files=expected_growth, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
def main() -> None:
    stage0_env_sanity()   # Drive mount happens HERE — before anything that touches DRIVE_ROOT
    stage_downloads()     # fresh VM: ALL raw sources + background pool (idempotent)
    stage_manifest()      # ends with the tcl.ensure_manifest_pairs guard (RuntimeError if empty)
    stage_composites_o()
    stage_export_o()
    stage_drive_copy_o()
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
