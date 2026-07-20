"""V4 DATA UPDATE CELL — in a fresh (FREE, CPU is enough — NO GPU NEEDED)
Colab session, adds ONLY the pairs of the NEW v4 categories (`text` =
logo/text preservation, `fx` = VFX glow around objects, `illustration` =
ToonOut illustrations) to the existing Drive dataset (`bg-remover-data/
TRAIN`); it does NOT REGENERATE the existing v1/v2/v3 data and does not
delete/overwrite ANY existing file.

SOURCE / ATTRIBUTION: this file's flow pattern (Drive mount -> download
bootstrap -> production -> TRAIN-only Drive merge -> integrity check) and the
`report`/`stage0_env_sanity`/`_download_bg_pool`/`_download_trans460`/
`_gdown_extract`/`_ensure_gdown`/`discover_him2k_dirs`/`merge_him2k`/
`_walk_dirs` functions were COPIED from
`training/v3_veri_guncelleme_hucresi.py` (that file CANNOT be imported as a
module because, by paste-run design, it runs `main()` on import — it remains
the single source of truth, update from there if you see drift). From the
download bootstrap, only what v4 REQUIRES was taken: the BG-20k background
pool + the transparent (Transparent-460) and general (HIM2K) foreground
sources for fx — dis5k/camo/p3m/cod10k are NOT USED in v4 production and are
not downloaded.

NEW IN V4:
1. **ToonOut** (HuggingFace `joelseytre/toonout`, a train/val/test split
   structure with im/gt/an subfolders): only the TRAIN split is downloaded
   and normalized as `/content/downloads/toonout/{im,gt}`. The TEST split is
   DELIBERATELY LEFT UNTOUCHED — that split will be reserved for the future
   illustration benchmark (if even a single file leaks from there into
   training, the benchmark is contaminated).
2. **Font bootstrap**: ~20 TTFs are downloaded to `/content/fonts` from the
   Google Fonts repository (github.com/google/fonts, OFL-licensed families);
   against network/URL rot, each font has its own try/except, and if none of
   them download we fall back to the system DejaVu fonts (preinstalled on
   Colab VMs).
3. **Production via `scripts/make_textfx.py`**: `run(out_dir, bg_dir,
   fg_dirs, toonout_dir, font_dir, seed, counts)` is called (counts:
   text=4000, fx=3500; the illustration count is AUTOMATIC from the ToonOut
   train size). That script is being written in a parallel effort — on an
   import/signature mismatch this cell stops with a CLEAR error message (the
   `stage_textfx` try/excepts below), it does not silently produce half a
   dataset.
4. **Drive merge with the v3 pattern**: only the `TRAIN/` subtree is MERGED
   via `shutil.copytree(..., dirs_exist_ok=True)` (no deletion/overwriting;
   the PARTIAL `stats.json` at the src root is NOT COPIED so it cannot
   clobber the authoritative FULL stats.json on Drive — the v3 reviewer
   finding #1 fix applies here too), and only the NEW ids are APPENDED to the
   Drive copy of the composite manifest (`train_composites_manifest.jsonl`)
   (`tcl.merge_composite_manifest`, deduped/idempotent). NO new stem goes to
   VAL — new stems are always written to TRAIN (existing rule:
   `val_stems.json` is not even READ in this cell, because the sources of the
   v4 categories do not intersect the existing VAL stems — they are all
   brand-new `text_`/`fx_`/ToonOut-sourced ids).

PREREQUISITES: the repo must be cloned at `/content/my-bg-remover` with `pip
install -e .` done; `bg-remover-data/TRAIN/{im,gt}` (the v1-v3 output) must
ALREADY exist on Drive. The repo must be UP TO DATE (the env stage attempts
an idempotent `git pull`): `scripts/make_textfx.py` and the text/fx support
in `benchmark.testset.CATEGORIES` were added in an effort SEPARATE from this
cell — if you run with a stale clone, `stage_textfx` stops with a clear
message.

Status tracking is the SAME mechanism as the v3 cell (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — stages: env, downloads,
fonts, textfx, export, drive_copy, (at the end) ALL.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import PIL.Image

# Transparent-460/HIM2K contain 100MP+ images; they can exceed PIL's 179MP
# "decompression bomb" error threshold (see the same line in
# v3_veri_guncelleme_hucresi.py).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS must come AFTER the PIL import/assignment)
from PIL import Image  # noqa: E402

# --- Constants (SAME as v3_veri_guncelleme_hucresi.py) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
SEED = 42
BG_POOL_SIZE = 5000

# --- v4-specific constants ---
TOONOUT_HF_REPO = "joelseytre/toonout"
TOONOUT_DIR = Path("/content/downloads/toonout")  # normalized im/ gt/ go here
FONT_DIR = Path("/content/fonts")
TEXTFX_OUT_DIR = Path("data/train_textfx")            # make_textfx.run() output (local, relative to WORKDIR)
EXPORT_DIR = "/content/birefnet_format_textfx"        # export_birefnet.export() output
TEXTFX_COUNTS = {"text": 4000, "fx": 3500}            # illustration is AUTOMATIC from the ToonOut size
V4_NEW_CATEGORIES = ("text", "fx", "illustration")

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ is not a package — we add the absolute path to sys.path to be able
# to import make_textfx/export_birefnet (see v3_veri_guncelleme_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import CATEGORIES, load_manifest  # noqa: E402  (package installed via pip install -e .)
import training.train_colab_lib as tcl  # noqa: E402  (torch-free, testable logic)


# ==========================================================================
# Status reporting — EXACTLY IDENTICAL to `v3_veri_guncelleme_hucresi.py::report`.
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
# Stage "env" — Drive mount (BEFORE everything, STATUS_DIR lives on Drive!) +
# repo git pull (idempotent) + environment sanity check. Source:
# v3_veri_guncelleme_hucresi.py::stage0_env_sanity; git pull is a v4-specific
# addition — since make_textfx.py was added in a parallel effort, a stale
# clone is the most common source of errors.
# ==========================================================================
RAW_DIR_CHECKS = {
    "trans460_train": "data/raw_train/trans460_train/fg",
    "him2k_raw": "data/raw_train/him2k_raw",
    "backgrounds": "data/backgrounds",
    "toonout": str(TOONOUT_DIR / "im"),
    "fonts": str(FONT_DIR),
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


def _git_pull_idempotent() -> None:
    """Updates the repo — `git pull --ff-only` is a no-op if already up to
    date (idempotent); if there is no network/there is a conflict, it prints a
    WARNING and continues (if make_textfx is missing, stage_textfx will stop
    with a clear message anyway)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("WARNING: git pull failed — the repo may be stale; if make_textfx.py "
                  "is missing, we will stop below with a clear error.")
    except Exception as e:
        print(f"WARNING: git pull could not be run ({e}) — continuing with the existing clone.")


def stage0_env_sanity() -> dict:
    # Drive is mounted BEFORE EVERYTHING ELSE (including report() — STATUS_DIR
    # lives on Drive!); drive.mount is idempotent. Source: the same stage in
    # the v3 cell.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Could not mount Drive: {DRIVE_ROOT} does not exist"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} files")
    for name in ("trans460_train", "him2k_raw", "backgrounds", "toonout", "fonts"):
        if counts[name] == 0:
            print(f"NOTE: {name} is currently empty — it will be downloaded in the "
                  f"'downloads'/'fonts' stage (normal on a fresh VM).")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage "downloads" — ONLY the sources v4 requires, IDEMPOTENT:
#   - BG-20k background pool (for the text/fx composites) — source:
#     v3_veri_guncelleme_hucresi.py::_download_bg_pool (copy).
#   - Transparent-460 (transparent foreground for fx) — source: same file,
#     _download_trans460 (copy).
#   - HIM2K (general foreground for fx; gdown + images/alphas merge) —
#     source: same file, _gdown_extract/discover_him2k_dirs/merge_him2k (copy).
#   - ToonOut (illustration) — v4-SPECIFIC, only the train split (do NOT
#     touch test).
# ==========================================================================
RAW = Path("data/raw_train")


def _load_source_defs() -> dict:
    with open("data/train_sources.json") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _download_bg_pool(source_defs: dict) -> int:
    """Source: v3_veri_guncelleme_hucresi.py::_download_bg_pool (originating
    from prepare_data_colab.ipynb cell (e)/15) — BG_POOL_SIZE backgrounds from
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


def _download_trans460(source_defs: dict) -> int:
    """Source: v3_veri_guncelleme_hucresi.py::_download_trans460 (copy) —
    fx foreground source: fg/ + alpha/ (transparent objects)."""
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


def _ensure_gdown() -> None:
    """Source: v3_veri_guncelleme_hucresi.py::_ensure_gdown (copy)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)


def _gdown_extract(drive_id: str, out_dir: Path, label: str) -> bool:
    """Source: v3_veri_guncelleme_hucresi.py::_gdown_extract (copy) — returns
    False on failure (does not stop the pipeline), skips if out_dir is
    populated."""
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
    """Source: v3_veri_guncelleme_hucresi.py::_walk_dirs (copy)."""
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
    """Source: v3_veri_guncelleme_hucresi.py::discover_him2k_dirs (copy)."""
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
    """Source: v3_veri_guncelleme_hucresi.py::merge_him2k (copy) —
    max-merges the instance alphas and produces {im,gt} pairs (fx general
    foreground)."""
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
    """Downloads HIM2K via gdown and merges images/alphas into {im,gt} —
    idempotent (skips if merged is already populated). The GENERAL leg of the
    fx foreground; if it fails to download, a WARNING is printed and we
    continue with trans460 only (only the existing fg directories are passed
    to make_textfx)."""
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
        print("HIM2K images/alphas directory pair not found — general foreground will be SKIPPED.")
        return 0
    n = merge_him2k(dirs[0], dirs[1], out_root)
    print(f"HIM2K merged: {n} pairs -> {out_root}")
    return n


def _download_toonout() -> int:
    """v4-SPECIFIC: downloads ONLY the train split of the HuggingFace
    `joelseytre/toonout` repo (allow_patterns=["train/*"] — the test split is
    DELIBERATELY never downloaded, it will be reserved for the illustration
    benchmark; val is unnecessary too) and normalizes it as
    `/content/downloads/toonout/{im,gt}` (from the im/gt/an subfolder
    structure, an/ is not used — make_textfx expects only im+gt).
    Idempotent: skips if the target im/ is already populated and the im/gt
    counts are equal."""
    from huggingface_hub import snapshot_download

    out_im = TOONOUT_DIR / "im"
    out_gt = TOONOUT_DIR / "gt"
    existing_im = len(list(out_im.iterdir())) if out_im.exists() else 0
    existing_gt = len(list(out_gt.iterdir())) if out_gt.exists() else 0
    if existing_im > 0 and existing_im == existing_gt:
        print(f"toonout: {TOONOUT_DIR} already contains {existing_im} pairs; skipping download.")
        return existing_im

    # Repo structure (verified as of 2026-07): the splits are NOT folders but
    # `data/{train,validation,test}_generations_*.tar` archives. Each tar
    # contains `<generation_name>/{im,gt,an}`. Only the train tars are
    # downloaded.
    import tarfile

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
                continue  # an image without gt cannot enter training
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
    print(f"toonout (train split): {copied} im/gt pairs -> {TOONOUT_DIR} (the test split was NOT TOUCHED).")
    assert copied > 0, "No im/gt pairs could be extracted from the ToonOut train split!"
    return copied


def stage_downloads() -> dict:
    report("downloads", "running")
    RAW.mkdir(parents=True, exist_ok=True)
    source_defs = _load_source_defs()
    results: dict = {}

    results["backgrounds"] = _download_bg_pool(source_defs)

    try:
        results["trans460"] = _download_trans460(source_defs)
    except Exception as e:
        print(f"WARNING: transparent_460 could not be downloaded ({e}); the on-disk copy will be used if present.")
        results["trans460"] = -1

    results["him2k_merged"] = _ensure_him2k_merged(source_defs)

    results["toonout"] = _download_toonout()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    print("File counts after downloads:", counts)
    report("downloads", "done", results=results, counts=counts)
    return results


# ==========================================================================
# Stage "fonts" — v4-SPECIFIC: downloads ~20 OFL-licensed TTFs from the
# Google Fonts repository (github.com/google/fonts) -> /content/fonts. Each
# font has its own try/except (URL rot must not stop the whole cell); if none
# download, we fall back to the DejaVu fonts preinstalled on the Colab VM.
# Producing the text category needs font VARIETY (text rendered with a single
# font does not generalize to the model).
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
        # in the URL; locally we use a plain bracket-free name (so it cannot
        # clash with glob patterns).
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
            "No fonts could be downloaded and the DejaVu fallback was not found either — the "
            "text category cannot be produced. Check the network connection or manually place "
            "TTFs into /content/fonts."
        )
    report("fonts", "done", downloaded=ok, failed=len(failed), total=total)
    return total


# ==========================================================================
# Stage "textfx" — PRODUCTION: scripts/make_textfx.py (being written in a
# PARALLEL effort — here only its documented signature is assumed:
# run(out_dir, bg_dir, fg_dirs, toonout_dir, font_dir, seed, counts)). On a
# signature/import mismatch we stop with a CLEAR error message, we do not
# silently produce half a dataset.
# ==========================================================================
def stage_textfx() -> dict:
    report("textfx", "running")

    # Repo freshness guard: the text/fx categories are added to
    # benchmark.testset.CATEGORIES by the make_textfx effort — on a stale
    # clone the manifest validation (append_entries/load_manifest) would blow
    # up with "unknown category"; we state the cause here, up front.
    missing_cats = {"text", "fx", "illustration"} - CATEGORIES
    if missing_cats:
        raise RuntimeError(
            f"benchmark.testset.CATEGORIES does not know these categories: {sorted(missing_cats)} — "
            f"your repo clone looks stale (the make_textfx effort adds them). "
            f"Run 'git -C {WORKDIR} pull' and re-run the cell."
        )

    try:
        import make_textfx as mtx  # scripts/ is on sys.path
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_textfx.py could not be imported ({e}) — this script is being written "
            f"in a parallel effort; is your repo up to date? Try 'git -C {WORKDIR} pull'. If the "
            f"script has not been merged yet, run this cell once make_textfx is ready."
        ) from e

    fg_dirs = [d for d in (RAW / "trans460_train", RAW / "him2k_merged") if d.is_dir()]
    assert fg_dirs, (
        "No foreground source for fx at all (both trans460_train and him2k_merged are missing) — "
        "inspect the 'downloads' stage logs."
    )

    # The illustration target is COMPUTED from the ToonOut pool:
    # make_textfx.run() counts a category NOT GIVEN in counts as 0 (it has no
    # "automatic" behavior — this is why illustration was never produced in
    # the first run). Each pair produces 3 copies (c00/c01 composite + c02
    # original), the whole pool is used.
    n_toonout = len(mtx._pairs_from_dir(TOONOUT_DIR))
    assert n_toonout > 0, f"No im/gt pairs under {TOONOUT_DIR} — check the 'downloads' logs."
    run_counts = dict(TEXTFX_COUNTS)
    run_counts["illustration"] = 3 * n_toonout

    try:
        counts = mtx.run(
            out_dir=TEXTFX_OUT_DIR,
            bg_dir=Path("data/backgrounds"),
            fg_dirs=fg_dirs,
            toonout_dir=TOONOUT_DIR,
            font_dir=FONT_DIR,
            seed=SEED,
            counts=run_counts,
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_textfx.run() could not be called with the expected signature ({e}) — this cell "
            f"assumes the run(out_dir, bg_dir, fg_dirs, toonout_dir, font_dir, seed, counts) "
            f"signature (the documented contract of the parallel effort). Check the current "
            f"signature of scripts/make_textfx.py and adapt the call."
        ) from e

    print("make_textfx.run() per-category production:", counts)

    # Manifest guard (the v3 lesson): do NOT PROCEED to export with an
    # empty/missing manifest. CAUTION: make_textfx's output manifest has
    # {"id","category"} rows — benchmark.testset.load_manifest (which requires
    # image/gt_alpha) is NOT USED HERE, and tcl.ensure_manifest_pairs cannot
    # be used either because it expects that schema.
    out_manifest = TEXTFX_OUT_DIR / "manifest.jsonl"
    if not out_manifest.exists():
        raise RuntimeError(f"{out_manifest} does not exist — make_textfx production must have failed.")
    rows = [json.loads(line) for line in out_manifest.read_text().splitlines() if line.strip()]
    total_pairs = len(rows)
    if total_pairs == 0:
        raise RuntimeError(f"{out_manifest} is empty — not proceeding to export (the v3 lesson).")

    # export_birefnet.export() requires the FULL testset schema (image +
    # gt_alpha paths) — we derive it from the {"id","category"} rows and write
    # it alongside. The path contract is the same as make_textfx._save_pair:
    # im/{id}.jpg + gt/{id}.png.
    full_manifest = TEXTFX_OUT_DIR / "manifest_full.jsonl"
    with open(full_manifest, "w") as f:
        for r in rows:
            im_p = TEXTFX_OUT_DIR / "im" / f"{r['id']}.jpg"
            gt_p = TEXTFX_OUT_DIR / "gt" / f"{r['id']}.png"
            if not (im_p.exists() and gt_p.exists()):
                raise RuntimeError(f"file for manifest row is missing: {r['id']} — production may have been cut short.")
            f.write(json.dumps({"id": r["id"], "image": str(im_p),
                                "category": r["category"], "gt_alpha": str(gt_p)},
                               ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"PRE-FLIGHT — {out_manifest}: {total_pairs} pairs total, by category:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")
    low = {c: by_cat.get(c, 0) for c in V4_NEW_CATEGORIES if by_cat.get(c, 0) < 100}
    if low:
        print(f"WARNING: these v4 categories are BELOW 100 samples: {low} — train_colab.ipynb's "
              f"v4 pre-flight guard will stop the GPU run in this situation.")

    report("textfx", "done", counts=counts, by_category=by_cat, total_pairs=total_pairs)
    return by_cat


# ==========================================================================
# Stage "export" — v3 pattern (export_birefnet.export() is UNCHANGED):
# because it runs against a fresh/empty local directory, only the new textfx
# files appear on disk (the source manifest already contains only
# text/fx/illustration rows). split_name="TRAIN": new stems ALWAYS go to
# TRAIN, NO new stem goes to VAL (existing rule).
# ==========================================================================
def stage_export_textfx() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ is on sys.path

    # manifest_full.jsonl: the manifest stage_textfx converted to the export
    # schema (image+gt_alpha) — the raw manifest.jsonl is {"id","category"},
    # so it cannot be given to the export DIRECTLY.
    stats = eb.export(
        manifest_path=str(TEXTFX_OUT_DIR / "manifest_full.jsonl"),
        out_dir=EXPORT_DIR,
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — v3 pattern: MERGE into the existing Drive TRAIN
# (dirs_exist_ok=True, NO file is DELETED/overwritten; the PARTIAL stats.json
# at the src root is NOT COPIED so it cannot clobber the authoritative FULL
# stats.json on Drive — v3 reviewer finding #1) + APPEND to the composite
# manifest (tcl.merge_composite_manifest, deduped — NO full overwrite).
# ==========================================================================
def stage_drive_copy_textfx() -> None:
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Expected v1-v3 TRAIN data not found on Drive: {dst_train_im} / {dst_train_gt} — "
        f"this cell is only for ADDING v4 (text/fx/illustration) to an EXISTING dataset; "
        f"to build a dataset from scratch, colab_devam_hucresi.py must be used."
    )

    def _listdir_retry(d: Path, attempts: int = 4, wait_s: int = 30) -> list[Path]:
        """Drive FUSE occasionally throws 'Errno 5 I/O error' on directories
        with 42k+ files (transient — also seen in the v3 run, retrying was
        enough). For that, it retries with a wait; on the last attempt it
        re-raises the error as-is."""
        import time
        for i in range(attempts):
            try:
                return list(d.iterdir())
            except OSError as e:
                if i == attempts - 1:
                    raise
                print(f"WARNING: {e} while listing {d} — waiting {wait_s}s and retrying "
                      f"({i + 1}/{attempts - 1}).")
                time.sleep(wait_s)
        raise AssertionError("unreachable")

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    # im and gt are counted SEPARATELY: a previous half-finished upload (if
    # the Drive flush did not complete while the VM was shutting down) can
    # leave pairs whose im arrived but whose gt did not — a single
    # `expected_growth` would mistakenly treat different growth in the two
    # directories as an error (this happened in the 2026-07-12 v4 run: 7200
    # broken pairs).
    existing_dst_im_stems = {p.stem for p in _listdir_retry(dst_train_im)}
    existing_dst_gt_stems = {p.stem for p in _listdir_retry(dst_train_gt)}
    growth_im = len({p.stem for p in src_im_files} - existing_dst_im_stems)
    growth_gt = len({p.stem for p in src_gt_files} - existing_dst_gt_stems)

    pre_im, pre_gt = len(existing_dst_im_stems), len(existing_dst_gt_stems)
    print(f"Drive TRAIN before merge: im={pre_im}, gt={pre_gt} — expected growth: im +{growth_im}, gt +{growth_gt}")

    # ONLY the TRAIN/ subtree is copied — the stats.json at the src root is
    # DELIBERATELY NOT COPIED (the v3 fix: the partial stats.json would
    # silently CLOBBER the authoritative FULL stats.json).
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
    assert len(src_im_files) == len(src_gt_files), "im/gt counts do not match in the local textfx export!"
    # The TRUE integrity condition: after the merge, every local stem's im AND gt are on Drive.
    assert post_im == post_gt, f"Drive TRAIN im/gt counts are not equal: {post_im} != {post_gt}"

    # manifest_full.jsonl: the load_manifest validation inside
    # merge_composite_manifest requires the FULL schema (image+gt_alpha), so
    # the raw manifest.jsonl cannot be given.
    comp_manifest_local = TEXTFX_OUT_DIR / "manifest_full.jsonl"
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} new rows appended (the existing "
          f"v1-v3 rows on Drive were PRESERVED, not overwritten).")
    # n_appended is the count of ids not yet in the manifest — on a repair run
    # the rows may already have been appended, so it can be 0; equality with
    # the file growth is NOT REQUIRED (the 2026-07-12 lesson). Sufficient
    # condition: if the deduped append is error-free and the file counts are
    # equal, integrity holds.

    print("\nINTEGRITY CHECK PASSED — v4 (text/fx/illustration) data was MERGED into Drive.")
    report(
        "drive_copy", "done",
        added_im=growth_im, added_gt=growth_gt, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )



# ==========================================================================
# Orchestration — runs at top level (when the cell is pasted and executed).
# ==========================================================================
def main() -> None:
    stage0_env_sanity()        # Drive mount + git pull happen HERE — before anything that touches Drive
    stage_downloads()          # ToonOut(train) + BG-20k + trans460 + HIM2K (idempotent)
    stage_fonts()              # ~20 OFL Google Fonts -> /content/fonts (DejaVu fallback)
    stage_textfx()             # make_textfx.run() + manifest guard + category pre-flight
    stage_export_textfx()
    stage_drive_copy_textfx()
    report("ALL", "done")
    # CRITICAL (the 2026-07-12 lesson): Drive writes are buffered
    # ASYNCHRONOUSLY — if the VM is shut down before this flush completes, the
    # files are SILENTLY lost (that is how the 7200 broken pairs came to be).
    # flush_and_unmount() FORCES the buffer to drain and blocks until it
    # finishes. It is called AFTER everything that writes to Drive (including
    # report).
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
