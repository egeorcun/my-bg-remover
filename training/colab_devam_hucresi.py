"""RESUME CELL — runs all remaining steps of training/prepare_data_colab.ipynb
(background pool download, COD10K/HIM2K structure discovery + HIM2K merge,
full manifest, composite generation, BiRefNet export, copy to Drive +
integrity check) end to end in a SINGLE cell.

USAGE: PASTE the ENTIRE contents of this file into a new cell in the live
Colab runtime (repo already checked out at /content/my-bg-remover, Drive
mounted, `pip install -e .` done, data/raw_train/{dis5k,camo,p3m,
trans460_train} + data/raw_train/{cod10k_raw,him2k_raw} already present) and
run it. No argparse, no `if __name__` block — the file runs directly at top
level without being imported.

Variables left in the kernel by earlier cells are NOT TRUSTED: this file
defines all of its state on its own, from scratch (idempotent where it is —
see each stage's docstring).

Status tracking: `report()` is called at the start/end of every stage; it
writes both to the console and to Drive
(`/content/drive/MyDrive/bg-remover-status/`) as `log.txt` (append) and
`status.json` (overwrite, accumulating `history`) — for monitoring progress
from outside (from outside this Colab session). If an unexpected error
occurs, the full traceback is reported with `stage="FATAL"` and the error is
RE-RAISED (not swallowed silently).
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

# Transparent-460/HIM2K contain 100MP+ images; the composite outputs can
# exceed PIL's 179MP "decompression bomb" error threshold. Since the data
# comes from trusted academic datasets, the limit is removed.
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np
from PIL import Image

# --- Constants ------------------------------------------------------------
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
SEED = 42
BG_POOL_SIZE = 5000

STATUS_DIR = Path(DRIVE_ROOT) / "bg-remover-status"
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ is not a package (no __init__.py above it) — we add the absolute path
# to sys.path so that build_trainset/make_composites/export_birefnet can be
# imported (we use an absolute path so it works even if cwd has not changed yet,
# independent of os.chdir).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries  # noqa: E402  (package installed via pip install -e .)


# ==========================================================================
# Status reporting — the controller watches these files FROM OUTSIDE, critical.
# ==========================================================================
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def report(stage: str, status: str, **extra) -> None:
    """Appends a line to log.txt + rewrites status.json (accumulating history).

    On every call, status.json's existing `history` is read (if present) and
    the new entry appended — so even if the script is interrupted and re-run,
    the history on Drive is not lost."""
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
# Stage 0 — environment sanity check
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
    """HF download timeout (from the stuck-download lesson) + HF_TOKEN from
    Colab Secrets (if present) — continues silently if not found (most sources
    work with anonymous access)."""
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    try:
        from google.colab import userdata

        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            print("HF_TOKEN obtained from Colab Secrets.")
    except Exception as e:
        print(f"Could not get HF_TOKEN (not in Secrets or access not granted): {e}")


def stage0_env_sanity() -> dict:
    report("env", "running")
    os.chdir(WORKDIR)
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} files")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage 1 — background pool (BG-20k)
# ==========================================================================
def stage1_bg_pool() -> int:
    report("bg_pool", "running")
    bg_dir = Path("data/backgrounds")
    bg_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(bg_dir.iterdir()))
    if existing >= BG_POOL_SIZE:
        print(f"data/backgrounds already contains {existing} images (>= {BG_POOL_SIZE}); skipping download.")
        report("bg_pool", "done", count=existing, skipped=True)
        return existing

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with open("data/train_sources.json") as f:
        source_defs = {s["name"]: s for s in json.load(f)["sources"]}
    bg_spec = source_defs["bg_20k"]

    fs = HfFileSystem()
    pattern = bg_spec["split_patterns"][0]  # "data/train-*-of-00022.parquet"
    parts = sorted(fs.glob(f"datasets/{bg_spec['hf_repo']}/{pattern}"))

    written = existing  # CUMULATIVE counter — not reset at part boundaries (see the note in notebook cell (c))
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
    report("bg_pool", "done", count=written)
    return written


# ==========================================================================
# Stage 2 — discovery of the real COD10K/HIM2K folder structure
# ==========================================================================
def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
    """For every directory under root (depth <= max_depth), returns jpg/png
    counts and stem sets — for pairing up img/gt directory pairs."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []  # do not descend further (but this directory itself is processed)
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
    """Discovers the real internal structure of the COD10K-v3 zip: pairs a
    directory containing many .jpg files with the .png directory that shares
    most of the same stems (stem overlap = the real correctness signal; name
    preference (Image/GT/Train) is used only as a tie-break among candidates
    with equal overlap)."""
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


def stage2_discover_structure() -> dict | None:
    report("discover_cod10k", "running")
    raw_dir = Path("data/raw_train/cod10k_raw")
    if not raw_dir.exists():
        print("data/raw_train/cod10k_raw missing — skipping COD10K.")
        report("discover_cod10k", "skipped", reason="directory missing")
        return None

    info = discover_cod10k(raw_dir)
    if info is None:
        print("No overlapping img/gt directory pair found for COD10K.")
        report("discover_cod10k", "skipped", reason="no match")
        return None

    print(f"COD10K selected pair: img={info['img_dir']}  gt={info['gt_dir']}  "
          f"overlapping stems={info['overlap']}  ambiguous={info['ambiguous']}")
    if info["ambiguous"]:
        print(f"WARNING: multiple candidates scored equally — best guess selected. Candidates: {info['candidates']}")
    report("discover_cod10k", "done", img_dir=str(info["img_dir"]), gt_dir=str(info["gt_dir"]),
           overlap=info["overlap"], ambiguous=info["ambiguous"], candidates=info["candidates"])
    return info


# ==========================================================================
# Stage 3 — HIM2K instance-matting merge
# ==========================================================================
def discover_him2k_dirs(raw_dir: Path) -> tuple[Path, Path] | None:
    """Finds the images/train and alphas/train directories. First tries by
    name (exact 'images/train' + 'alphas/train' path pattern); if not found,
    falls back to count-based guessing (the directory with the most .jpg
    files = images; a separate directory holding the most subdirectories =
    alphas, assuming instance folders)."""
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

    # Fallback: not found by name — count-based best guess.
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
    """For each image, if alphas_dir/<stem>/ is a directory (instance PNGs),
    merges them all with a pixel-wise max; if alphas_dir/<stem>.{png,jpg} is a
    flat file, uses it directly. Images are copied (no risk of broken
    symlinks when moving to Drive, see the _link note in
    scripts/build_trainset.py)."""
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


def stage3_merge_him2k() -> int:
    report("him2k_merge", "running")
    raw_dir = Path("data/raw_train/him2k_raw")
    if not raw_dir.exists():
        print("data/raw_train/him2k_raw missing — skipping HIM2K (the general category is optional).")
        report("him2k_merge", "skipped", reason="directory missing")
        return 0

    dirs = discover_him2k_dirs(raw_dir)
    if dirs is None:
        print("HIM2K images/alphas directory pair not found — skipping.")
        report("him2k_merge", "skipped", reason="images/alphas not found")
        return 0
    images_dir, alphas_dir = dirs
    print(f"HIM2K: images_dir={images_dir}  alphas_dir={alphas_dir}")

    out_root = Path("data/raw_train/him2k_merged")
    existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
    existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
    if existing_gt > 0 and existing_gt == existing_im:
        print(f"data/raw_train/him2k_merged already contains {existing_gt} pairs; skipping merge (idempotent).")
        report("him2k_merge", "done", count=existing_gt, skipped=True)
        return existing_gt

    count = merge_him2k(images_dir, alphas_dir, out_root)
    print(f"HIM2K merged: {count} pairs -> {out_root}")
    report("him2k_merge", "done", count=count)
    return count


# ==========================================================================
# Stage 4 — full manifest (using build_trainset.py logic, n=None + copy=True)
# ==========================================================================
def stage4_build_manifest(cod10k_info: dict | None, him2k_count: int) -> dict:
    report("manifest", "running")
    import build_trainset as bt  # scripts/ is on sys.path (added at the top of the file)

    # Clean start on every run — deterministic (spec item 10).
    if bt.MANIFEST.exists():
        bt.MANIFEST.unlink()
    for d in (bt.OUT_IMG, bt.OUT_GT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    counts: dict = {}

    def _run(name: str, img_glob: str, gt_glob: str, category: str, **kw) -> int:
        # sample_source(n=None, ...) returns ALL matching pairs (verified
        # against the source code: with the `n is not None` check, sampling
        # only kicks in when n is given) — no need for the huge-int trick.
        rows = bt.sample_source(name, img_glob, gt_glob, category, n=None, copy=True, **kw)
        append_entries(str(bt.MANIFEST), rows)
        counts[name] = len(rows)
        print(f"{name} ({category}): {len(rows)} pairs")
        return len(rows)

    # camotr / p3m / trans460tr — from the SINGLE source of truth SOURCE_SPECS (except disvd_tokens).
    for name, spec in bt.SOURCE_SPECS.items():
        if spec["category"] == "disvd_tokens":
            continue
        _run(name, spec["img_glob"], spec["gt_glob"], spec["category"])

    # dis5ktr — the category is assigned from the file-name token (thin/complex).
    rows = bt.sample_disvd_tokens("dis5ktr", bt.DIS5KTR_IMG_GLOB, bt.DIS5KTR_GT_GLOB, n=None, copy=True)
    append_entries(str(bt.MANIFEST), rows)
    dis_counts: dict = {}
    for r in rows:
        dis_counts[r["category"]] = dis_counts.get(r["category"], 0) + 1
    counts["dis5ktr"] = dis_counts
    for category, c in sorted(dis_counts.items()):
        print(f"dis5ktr ({category}): {c} pairs")

    # cod10ktr — from the real img/gt directories discovered in Stage 2.
    if cod10k_info:
        # Discovery may return relative paths; resolve against the root, then relativize.
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
        print("cod10ktr: skipped (directory not found in Stage 2)")

    # him2k — from him2k_merged/{im,gt} merged in Stage 3 (general category, optional).
    if him2k_count > 0:
        _run("him2k", "data/raw_train/him2k_merged/im/*", "data/raw_train/him2k_merged/gt/*", "general")
    else:
        counts["him2k"] = 0
        print("him2k: skipped (merge could not be done in Stage 3)")

    report("manifest", "done", counts=counts)
    return counts


# ==========================================================================
# Stage 5 — composite + augmentation generation (make_composites.run)
# ==========================================================================
def stage5_make_composites() -> dict:
    report("composites", "running")
    import make_composites as mc  # scripts/ is on sys.path

    # per_image=1 + the script's default CATEGORY_MULTIPLIER (transparent x10,
    # camouflage x2) — NO override, drift prevention (spec: script defaults are used).
    counts = mc.run(
        manifest_path=Path("data/train/manifest.jsonl"),
        backgrounds_dir=Path("data/backgrounds"),
        per_image=1,
        seed=SEED,
        out_dir=Path("data/train_composites"),
    )
    print("Composites generated per category:", counts)
    report("composites", "done", counts=counts)
    return counts


# ==========================================================================
# Stage 6 — export to BiRefNet format
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


# ==========================================================================
# Stage 7 — copy to Drive + integrity check
# ==========================================================================
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
    stage0_env_sanity()
    stage1_bg_pool()
    cod10k_info = stage2_discover_structure()
    him2k_count = stage3_merge_him2k()
    stage4_build_manifest(cod10k_info, him2k_count)
    stage5_make_composites()
    stats = stage6_export()
    stage7_drive_copy(stats)
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
