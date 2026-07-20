"""Build a categorized `data/train/manifest.jsonl` from GT-labeled training sources.

Usage:
    uv run python scripts/build_trainset.py                # all SOURCES + DIS5K-TR
    uv run python scripts/build_trainset.py source camotr   # add a single source

Difference from the test set (build_testset.py): files are NOT COPIED; a SYMBOLIC
LINK (symlink) pointing to the original raw file is created under
`data/train/{images,gt}/` — the disk budget (see Phase 2 plan Global Constraints)
does not allow copying. Format conversion (e.g. normalizing GT to single-channel
L PNG) is therefore NOT done at symlink time but during raw data acquisition
(fetch); this script only globs and links raw files that are already normalized.

Raw data acquisition (into data/raw_train/, outside git; DISK BUDGET: ≤300MB per
source, see Phase 2 plan REVISED disk rule — full materialization happens on Colab,
see data/train_sources.json):

- DIS5K-TR (category: thin/complex from the filename token, see classify_disvd): HF
  dataset "nobg/DIS5K", data/DIS_TR-00000-of-00006-*.parquet (only the first of 6
  shards; full DIS_TR is ~3000 pairs). Since even one shard is ~480MB, the FULL
  PARQUET WAS NOT DOWNLOADED: with pyarrow ParquetFile + huggingface_hub.HfFileSystem
  (fsspec, HTTP range request), ONLY row-group 0 (100 rows, ~120MB) was partially
  read, and the image/label bytes were written as files with PIL under
  data/raw_train/dis5k/{im,gt}/ (100 pairs; GT normalized to single channel with
  convert("L")). The full DIS_TR download will be done on Colab (T5 notebook) via
  the hf_repo in data/train_sources.json.
- CAMO-TR (category: camouflage): HF dataset "nobg/camo" (official CAMO, Le et al.
  2019; project page https://sites.github.com/view/ltnghia/research/camo; license
  CC-BY-NC-SA 4.0; train split 1000 pairs = 3 row-groups: 423+423+154). Only
  row-group 0's image_name/image/mask columns were partially read (EXCLUDING
  overlaid_mask_1/2 — column pruning avoided extra download); the first 100 rows
  were written under data/raw_train/camo/{im,gt}/ (~18MB total).
- COD10K-TR (category: camouflage): there is NO TRAIN mirror with pixel-level GT
  masks on HF (see scripts/build_testset.py docstring — Chranos/COD10K_train and
  Jrseee/COD10K are empty repos/missing LFS pointers; chandrabhuma/animal_cod10k(_train)
  has only images+Q&A, no pixel masks; SmallDoge/CoD-10K, also searched during this
  task, is an unrelated text dataset — a "CoD" name collision, a "Chain of Draft"
  style code/text corpus). The official source is Google Drive only (SINet/DengPingFan
  repo, https://github.com/DengPingFan/SINet): COD10K-train file id
  "1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt" (~3040 pairs). Since this environment has no
  gdown/Drive authentication, the LOCAL SAMPLE WAS SKIPPED (task instruction: "if
  unavailable, SKIP"); the record exists in data/train_sources.json (drive_id +
  official URL) — the full download will be done on Colab (T5) with gdown.
- P3M-10k TRAIN (category: hair): HF dataset "Rupant-ted/p3m-10k" is hosted as a
  SINGLE zip (data/p3m10k.zip, ~5.8GB); there are 9422 pairs under
  `P3M-10k/train/{blurred_image,mask}/` (verified first via list_repo_files for the
  zip's presence, then by reading the central directory through zipfile.ZipFile over
  huggingface_hub.HfFileSystem (fsspec) WITHOUT extracting the zip). The zip WAS NOT
  FULLY DOWNLOADED: the central directory was partially read via HTTP range request,
  then only the compressed byte ranges of 100 random pairs were fetched (again via
  range requests, in parallel on 12 threads) (~50MB total) and written with PIL (GT
  convert("L")) under data/raw_train/p3m/{im,gt}/.
- Transparent-460 TRAIN (category: transparent): HF dataset "Thinnaphat/transparent-460"
  has 410 pairs under `Train/{fg,alpha}/` (Phase 0 used only the 50 `Test/` pairs).
  The original files are very large (average ~4.2MB, some alpha PNGs 40-80MB) — to
  stay within the disk budget (≤300MB/source): 80 pairs were randomly selected from
  the pool of the 300/410 smallest pairs by size reported via
  `repo_info(files_metadata=True)`, streamed into memory with HfFileSystem (the
  hf_hub_download cache was NOT USED — disk savings), downscaled with PIL to a long
  side of 1280px (fg: JPEG q90, alpha: PNG, both to the SAME size) and written under
  data/raw_train/trans460_train/{fg,alpha}/ (~22MB total). The full TRAIN set (at
  original resolution) will be downloaded on Colab.
"""
import argparse
import random
import shutil
from pathlib import Path

from build_testset import _sanitize, classify_disvd  # noqa: E402  (same directory, scripts/)

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/train/images"
OUT_GT = ROOT / "data/train/gt"
MANIFEST = ROOT / "data/train/manifest.jsonl"


def _link(src: Path, dst: Path, copy: bool = False) -> None:
    """Creates a dst -> src symbolic link (default; NO copy, disk savings).

    With copy=True the actual file IS COPIED — for full data materialization on
    Colab (see training/prepare_data_colab.ipynb): a symlink can break during the
    move/zip to Drive by losing its target (Colab's temporary /content disk); a
    copy carries no such fragility.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src.resolve(), dst)
    else:
        dst.symlink_to(src.resolve())


def sample_source(name: str, img_glob: str, gt_glob: str, category: str,
                   n: int | None = None, copy: bool = False, *,
                   gt_stem_suffix: str | None = None) -> list[dict]:
    """Sample from a source (img_glob/gt_glob are paired by stem) and symlink under
    `data/train/{images,gt}/` (copy=True -> copy the actual file).
    n=None -> all matching pairs.

    gt_stem_suffix: in some sources the GT filename carries a fixed suffix on the
    image's stem (e.g. HIM2K: "foo.jpg" <-> "foo_matte.png") — the default pairing
    (identical stem) NEVER matches in that case (silently 0 pairs). If provided,
    this suffix is stripped from the GT stem before pairing; None (default) ->
    the existing behavior (exact same stem) stays unchanged."""
    imgs = sorted(ROOT.glob(img_glob))
    gts: dict[str, Path] = {}
    for p in ROOT.glob(gt_glob):
        stem = p.stem
        if gt_stem_suffix and stem.endswith(gt_stem_suffix):
            stem = stem[: -len(gt_stem_suffix)]
        gts[stem] = p
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    if n is not None and n < len(paired):
        paired = random.sample(paired, n)

    rows = []
    for img, gt in paired:
        rid = f"{name}_{_sanitize(img.stem)}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}{gt.suffix}"
        _link(img, dst_i, copy=copy)
        _link(gt, dst_g, copy=copy)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def sample_disvd_tokens(name: str, img_glob: str, gt_glob: str,
                         n: int | None = None, copy: bool = False) -> list[dict]:
    """Sample from the DIS5K pool; the category is assigned from the filename
    token (classify_disvd, reused from build_testset.py) — there is NO random
    distribution."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    random.shuffle(paired)
    if n is not None:
        paired = paired[:n]

    rows = []
    for img, gt in paired:
        sanitized_stem = _sanitize(img.stem)
        category = classify_disvd(sanitized_stem)
        rid = f"{name}_{category}_{sanitized_stem}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}{gt.suffix}"
        _link(img, dst_i, copy=copy)
        _link(gt, dst_g, copy=copy)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


# SINGLE SOURCE OF TRUTH: source name -> glob patterns + category rule. The sample
# size is DELIBERATELY NOT here (separate, in LOCAL_SAMPLE_N) — the Colab notebook
# (training/prepare_data_colab.ipynb) uses the same definitions with n=None (full
# set); this prevents the glob/category info from being hand-copied into the
# notebook and drifting over time. category "disvd_tokens" -> the category is
# assigned from the filename token (classify_disvd, handled by sample_disvd_tokens);
# the others are fixed categories (handled by sample_source).
#
# NOTE (matting sets research, see data/train_sources.json + Phase 2 T3 report):
# Distinctions-646 (Qiao et al. CVPR2020, HAttMatting) is distributed only on request
# by e-mail — there is NO HF/public download link, skipped. HIM2K (Sun et al.
# CVPR2022, InstMatt) and AM-2k (Li et al. IJCV2022, GFM) are distributed only via
# Google Drive/Baidu Wangpan (AM-2k additionally requires signing the official
# MIT-licensed "Dataset Release Agreement"); this environment has no gdown/Drive
# authentication (same constraint as COD10K-TR) — the LOCAL SAMPLE WAS SKIPPED, the
# records exist in data/train_sources.json (drive_id + official URL + license note)
# and will be downloaded on Colab (T5).
SOURCE_SPECS: dict[str, dict[str, str]] = {
    "camotr": {"img_glob": "data/raw_train/camo/im/*", "gt_glob": "data/raw_train/camo/gt/*",
               "category": "camouflage"},
    "p3m": {"img_glob": "data/raw_train/p3m/im/*", "gt_glob": "data/raw_train/p3m/gt/*",
            "category": "hair"},
    "trans460tr": {"img_glob": "data/raw_train/trans460_train/fg/*",
                   "gt_glob": "data/raw_train/trans460_train/alpha/*",
                   "category": "transparent"},
    "dis5ktr": {"img_glob": "data/raw_train/dis5k/im/*", "gt_glob": "data/raw_train/dis5k/gt/*",
                "category": "disvd_tokens"},
}

# Local validation sample sizes (disk budget, see module docstring) — only
# meaningful in local runs; Colab works with the full set (n=None).
LOCAL_SAMPLE_N: dict[str, int] = {"camotr": 100, "p3m": 100, "trans460tr": 80, "dis5ktr": 100}

# Backward-compatible view: (source_name, images_glob, gt_glob, category, count) —
# derived from SOURCE_SPECS (except disvd_tokens; that is handled by sample_disvd_tokens).
SOURCES: list[tuple[str, str, str, str, int]] = [
    (name, spec["img_glob"], spec["gt_glob"], spec["category"], LOCAL_SAMPLE_N[name])
    for name, spec in SOURCE_SPECS.items()
    if spec["category"] != "disvd_tokens"
]

# DIS5K-TR is sampled from a single pool; the category is assigned from the filename token.
DIS5KTR_IMG_GLOB = SOURCE_SPECS["dis5ktr"]["img_glob"]
DIS5KTR_GT_GLOB = SOURCE_SPECS["dis5ktr"]["gt_glob"]
DIS5KTR_N = LOCAL_SAMPLE_N["dis5ktr"]


def build(copy: bool = False) -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src, copy=copy)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]} ({src[3]}): {len(rows)} samples")

    rows = sample_disvd_tokens("dis5ktr", DIS5KTR_IMG_GLOB, DIS5KTR_GT_GLOB, DIS5KTR_N, copy=copy)
    append_entries(str(MANIFEST), rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"dis5ktr ({category}): {count} samples")


def add_source(name: str, copy: bool = False) -> None:
    """Sample a single source from SOURCES and append it to the manifest (incremental add)."""
    matches = [s for s in SOURCES if s[0] == name]
    if not matches:
        raise SystemExit(f"unknown source: {name} (SOURCES: {[s[0] for s in SOURCES]})")
    rows = sample_source(*matches[0], copy=copy)
    append_entries(str(MANIFEST), rows)
    print(f"{name} ({matches[0][3]}): {len(rows)} samples added")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["source"], default=None,
                         help="if given, adds a single source (see --name)")
    parser.add_argument("name", nargs="?", default=None, help="source name for the 'source' command")
    parser.add_argument("--copy", action="store_true",
                         help="copy the actual file instead of symlinking (for full data "
                              "materialization on Colab; locally the default stays symlink)")
    args = parser.parse_args()

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if args.command == "source":
        if not args.name:
            raise SystemExit("usage: build_trainset.py source <name> [--copy]")
        add_source(args.name, copy=args.copy)
    else:
        build(copy=args.copy)


if __name__ == "__main__":
    main()
