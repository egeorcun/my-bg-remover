"""Sample a categorized test set from GT-labeled source sets.

Usage:
    uv run python scripts/build_testset.py                          # sample the GT-labeled sets
    uv run python scripts/build_testset.py add data/testset/incoming product  # add images without GT

Raw data acquisition (into data/raw/, outside git):
- P3M-500-NP: HF dataset "Rupant-ted/p3m-10k" -> data/p3m10k.zip; only
  P3M-10k/validation/P3M-500-NP/{original_image,mask} was extracted from the zip and
  moved under data/raw/p3m10k/validation/P3M-500-NP/ (500 pairs).
- Transparent-460: HF dataset "Thinnaphat/transparent-460",
  snapshot_download(allow_patterns=["Test/*"]) -> data/raw/trans460/Test/{fg,alpha} (50 pairs).
- DIS-VD: HF dataset "nobg/DIS5K" -> data/DIS_VD-*.parquet; the (image, label) bytes
  in the parquet were written as files with PIL under data/raw/dis5k/DIS-VD/{im,gt}/
  (470 pairs, requires pyarrow). Raw filenames have the form
  '<groupIdx>#<Group>#<classIdx>#<Class>#<originalName>'
  (e.g. '1#Accessories#5#Jewelry#12836143775_...').
- CAMO (camouflage category): for Phase 2 Task 1 the COD10K test split was searched
  on HF (HfApi().list_datasets/list_repo_files: "Chranos/COD10K_train", "Jrseee/COD10K"
  are empty repos/missing LFS pointers; "chandrabhuma/animal_cod10k" contains the real
  COD10K-CAM-Test images (2026 samples, ids prefixed "COD10K-CAM-...") but only
  images+Q&A, NO pixel-level GT masks; the official source (SINet/DengPingFan) is
  Google Drive only and this environment has no gdown/kaggle credentials). Instead,
  the HF dataset "nobg/camo" was used: the official CAMO (Camouflaged Object, Le et
  al.) test split, image+mask parquet (250 pairs, ~61MB) -> written as files with PIL
  under data/raw/camo_test/{im,gt}/. CAMO is also a camouflage source accepted
  alongside COD10K-TR in this project's Phase 2 plan (Task 2); its use in place of
  COD10K for the "camouflage" category is documented in the report.

NOTE (final review fix): the DIS-VD rows were initially distributed RANDOMLY across
thin/complex/general (see git history). scripts/relabel_disvd.py fixed this as a
one-off: since the real DIS5K class is encoded inside the id (classify_disvd(),
below), each row's category was recomputed from the filename token; ids/filenames
did not change. sample_disvd_multi() now uses classify_disvd() FROM THE START, i.e.
there is NO random distribution in future rebuilds.
"""
import random
import re
import sys
from pathlib import Path

from PIL import Image

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/testset/images"
OUT_GT = ROOT / "data/testset/gt"
MANIFEST = ROOT / "data/testset/manifest.jsonl"

# (source_name, images_glob, gt_glob, category, count)
# NOTE: no working mirror was found for AIM-500 and AM-2k (Google Drive only,
# folder-based, and it also crawls the "train" tree with thousands of files -> not
# practical). Therefore DIS-VD (470 pairs) was sampled disjointly into three
# categories (thin/complex/general) to balance the GT-labeled total; see
# sample_disvd_multi().
SOURCES: list[tuple[str, str, str, str, int]] = [
    ("p3m", "data/raw/p3m10k/validation/P3M-500-NP/original_image/*.jpg",
     "data/raw/p3m10k/validation/P3M-500-NP/mask/*.png", "hair", 40),
    ("trans460", "data/raw/trans460/Test/fg/*", "data/raw/trans460/Test/alpha/*", "transparent", 25),
    ("camo", "data/raw/camo_test/im/*", "data/raw/camo_test/gt/*", "camouflage", 25),
]

# DIS-VD is sampled from a single pool; the category is assigned from the filename token (see classify_disvd).
DISVD_IMG_GLOB = "data/raw/dis5k/DIS-VD/im/*"
DISVD_GT_GLOB = "data/raw/dis5k/DIS-VD/gt/*"
DISVD_N = 65  # same size as the old thin(20)+complex(30)+general(15) total

# thin/complex classification from DIS5K class tokens.
# "thin" = classes dominated by thin, holey/woven geometry such as wires/mesh/skeletons.
_THIN_DISVD_CLASSES = {
    "racket", "cable", "wire", "fence", "gate", "antenna", "jewelry", "chandelier",
    "bicycle", "tricycle", "wheel", "ladder", "windmill", "drum", "drumkit", "scaffold",
    "net", "skeleton", "umbrella", "polevault", "handrail", "floorlamp", "musicstand",
    "stand", "spider", "shrimp", "streetlamp", "shoppingcart", "seadragon", "hangglider",
    "basketballhoop", "earphone",
}


def _sanitize(stem: str) -> str:
    """URL-safe id: turn characters outside [A-Za-z0-9._-] into '_', collapse runs."""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]", "_", stem))


def _copy_alpha(src: Path, dst: Path) -> None:
    """Normalize the GT alpha to single-channel (L) PNG and copy."""
    Image.open(src).convert("L").save(dst)


def _copy_image(src: Path, dst: Path) -> None:
    img = Image.open(src)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(dst)


def sample_source(name: str, img_glob: str, gt_glob: str, category: str, n: int) -> list[dict]:
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    rows = []
    for img, gt in random.sample(paired, min(n, len(paired))):
        rid = f"{name}_{_sanitize(img.stem)}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}.png"
        _copy_image(img, dst_i)
        _copy_alpha(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def parse_disvd_class(stem: str) -> str | None:
    """Defensively extract the class token from a DIS5K stem/id.

    Raw filenames have the form '<groupIdx>#<Group>#<classIdx>#<Class>#<originalName>';
    the same logic still parses after '#' -> '_' sanitization (see _sanitize) or after
    a 'disvd_<oldCategory>_' prefix has been added: among the underscore-separated
    tokens, the first two PURELY NUMERIC tokens are the group/class indices (the group
    name may span multiple tokens, like 'Non-motor_Vehicle' — that does not matter);
    the class name is the single token immediately after the second numeric token.
    Returns None if it cannot be parsed.
    """
    parts = stem.split("_")
    digit_idxs = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(digit_idxs) < 2:
        return None
    class_idx = digit_idxs[1]
    if class_idx + 1 >= len(parts):
        return None
    return parts[class_idx + 1]


def classify_disvd(stem: str) -> str:
    """Returns the true category (thin/complex) from a DIS5K stem/id.

    The default for unparseable or unlisted classes is 'complex'
    (see _THIN_DISVD_CLASSES; a safe default for unknown future classes).
    """
    cls = parse_disvd_class(stem)
    if cls is None:
        return "complex"
    return "thin" if cls.lower() in _THIN_DISVD_CLASSES else "complex"


def sample_disvd_multi(name: str, img_glob: str, gt_glob: str, n: int) -> list[dict]:
    """Draws n samples from the DIS-VD pool; the category is assigned from the
    filename token (classify_disvd) (there is NO random distribution)."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
    paired = [(i, gts[i.stem]) for i in imgs if i.stem in gts]
    random.shuffle(paired)

    rows = []
    for img, gt in paired[:n]:
        sanitized_stem = _sanitize(img.stem)
        category = classify_disvd(sanitized_stem)
        rid = f"{name}_{category}_{sanitized_stem}"
        dst_i = OUT_IMG / f"{rid}{img.suffix}"
        dst_g = OUT_GT / f"{rid}.png"
        _copy_image(img, dst_i)
        _copy_alpha(gt, dst_g)
        rows.append({"id": rid, "image": str(dst_i.relative_to(ROOT)),
                     "category": category, "gt_alpha": str(dst_g.relative_to(ROOT))})
    return rows


def add_unlabeled(folder: str, category: str) -> None:
    rows = []
    for img in sorted((ROOT / folder).glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rid = f"user_{category}_{_sanitize(img.stem)}"
        dst = OUT_IMG / f"{rid}{img.suffix}"
        _copy_image(img, dst)
        rows.append({"id": rid, "image": str(dst.relative_to(ROOT)),
                     "category": category, "gt_alpha": None})
    append_entries(str(MANIFEST), rows)
    print(f"{category}: {len(rows)} images without GT added")


def build() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]} ({src[3]}): {len(rows)} samples")

    rows = sample_disvd_multi("disvd", DISVD_IMG_GLOB, DISVD_GT_GLOB, DISVD_N)
    append_entries(str(MANIFEST), rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"disvd ({category}): {count} samples")


def add_source(name: str) -> None:
    """Sample a single source from SOURCES and append it to the manifest (without
    re-adding existing rows; build() is a from-scratch build, this is an INCREMENTAL add)."""
    matches = [s for s in SOURCES if s[0] == name]
    if not matches:
        raise SystemExit(f"unknown source: {name} (SOURCES: {[s[0] for s in SOURCES]})")
    rows = sample_source(*matches[0])
    append_entries(str(MANIFEST), rows)
    print(f"{name} ({matches[0][3]}): {len(rows)} samples added")


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) >= 4 and sys.argv[1] == "add":
        add_unlabeled(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "source":
        add_source(sys.argv[2])
    else:
        build()


if __name__ == "__main__":
    main()
