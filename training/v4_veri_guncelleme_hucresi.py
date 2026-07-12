"""V4 VERİ GÜNCELLEME HÜCRESİ — taze bir (ÜCRETSİZ, CPU yeterli — GPU GEREKMEZ)
Colab oturumunda, mevcut Drive veri setine (`bg-remover-data/TRAIN`) yalnız YENİ
v4 kategorilerinin (`text` = logo/yazı koruma, `fx` = obje etrafı VFX parıltı,
`illustration` = ToonOut illüstrasyonları) çiftlerini ekler; v1/v2/v3'ün mevcut
verisini YENİDEN ÜRETMEZ ve HİÇBİR mevcut dosyayı silmez/üzerine yazmaz.

KAYNAK / ATIF: bu dosyanın akış kalıbı (Drive mount → indirme bootstrap →
üretim → TRAIN-only Drive merge → bütünlük kontrolü) ve `report`/`stage0_env_
sanity`/`_download_bg_pool`/`_download_trans460`/`_gdown_extract`/`_ensure_
gdown`/`discover_him2k_dirs`/`merge_him2k`/`_walk_dirs` fonksiyonları
`training/v3_veri_guncelleme_hucresi.py`'den KOPYALANDI (o dosya paste-run
tasarımı gereği import edilirken `main()` çalıştırdığından modül olarak import
EDİLEMEZ — tek doğruluk kaynağı o dosyadır, drift görürseniz oradan güncelleyin).
İndirme bootstrap'inden yalnız v4'ün GEREKTİRDİKLERİ alındı: BG-20k arka plan
havuzu + fx için transparent (Transparent-460) ve general (HIM2K) foreground
kaynakları — dis5k/camo/p3m/cod10k v4 üretiminde KULLANILMAZ, indirilmez.

V4'E ÖZGÜ YENİLER:
1. **ToonOut** (HuggingFace `joelseytre/toonout`, im/gt/an alt klasörlü
   train/val/test split yapısı): yalnız TRAIN split'i indirilir ve
   `/content/downloads/toonout/{im,gt}` olarak normalize edilir. TEST split'ine
   BİLEREK DOKUNULMAZ — o split ileride illustration benchmark'ı için ayrılacak
   (buradan tek dosya bile eğitime sızarsa benchmark kirlenir).
2. **Font bootstrap**: Google Fonts deposundan (github.com/google/fonts, OFL
   lisanslı aileler) ~20 TTF `/content/fonts`'a indirilir; ağ/URL çürümesine
   karşı her font tek tek try/except'li, hiçbiri inmezse sistem DejaVu
   fontlarına düşülür (Colab VM'lerinde hazır bulunur).
3. **Üretim `scripts/make_textfx.py` ile**: `run(out_dir, bg_dir, fg_dirs,
   toonout_dir, font_dir, seed, counts)` çağrılır (counts: text=4000, fx=3500;
   illustration sayısı ToonOut train boyutundan OTOMATİK). Bu script paralel
   bir çalışmada yazılıyor — import/imza uyuşmazlığında bu hücre NET Türkçe
   hata mesajıyla durur (aşağıdaki `stage_textfx` try/except'leri), sessizce
   yarım veri üretmez.
4. **Drive merge v3 kalıbıyla**: yalnız `TRAIN/` alt ağacı `shutil.copytree(...,
   dirs_exist_ok=True)` ile MERGE edilir (silme/üzerine yazma yok; src kökündeki
   KISMİ `stats.json` Drive'daki otoriter TAM stats.json'u ezmesin diye
   KOPYALANMAZ — v3 reviewer bulgusu #1 fix'i burada da geçerli), kompozit
   manifest'in Drive kopyasına (`train_composites_manifest.jsonl`) yalnız YENİ
   id'ler APPEND edilir (`tcl.merge_composite_manifest`, dedupe'lu/idempotent).
   VAL'e HİÇBİR yeni stem gitmez — yeni stemler her zaman TRAIN'e yazılır
   (mevcut kural: `val_stems.json` bu hücrede OKUNMAZ bile, çünkü v4
   kategorilerinin kaynakları mevcut VAL stemleriyle kesişmez — hepsi yepyeni
   `text_`/`fx_`/ToonOut kaynaklı id'ler).

ÖN KOŞULLAR: repo `/content/my-bg-remover`'da klonlanmış ve `pip install -e .`
yapılmış olmalı; Drive'da `bg-remover-data/TRAIN/{im,gt}` (v1-v3 çıktısı) ZATEN
mevcut olmalı. Repo GÜNCEL olmalı (env aşaması idempotent `git pull` dener):
`scripts/make_textfx.py` ve `benchmark.testset.CATEGORIES`'in text/fx desteği
bu hücreden AYRI bir çalışmada eklendi — eski bir klonla koşarsanız
`stage_textfx` net bir mesajla durur.

Durum takibi v3 hücresiyle AYNI mekanizma (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — aşamalar: env, downloads,
fonts, textfx, export, drive_copy, (bitişte) ALL.
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

# Transparent-460/HIM2K'da 100MP+ görseller var; PIL'in 179MP "decompression
# bomb" hata eşiğini aşabiliyor (bkz. v3_veri_guncelleme_hucresi.py aynı satır).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS PIL importundan/atamasından SONRA gelmeli)
from PIL import Image  # noqa: E402

# --- Sabitler (v3_veri_guncelleme_hucresi.py ile AYNI) ---
WORKDIR = "/content/my-bg-remover"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_OUTPUT_SUBDIR = "bg-remover-data"
DRIVE_STATUS_SUBDIR = "bg-remover-status"
SEED = 42
BG_POOL_SIZE = 5000

# --- v4'e özgü sabitler ---
TOONOUT_HF_REPO = "joelseytre/toonout"
TOONOUT_DIR = Path("/content/downloads/toonout")  # normalize edilmiş im/ gt/ buraya
FONT_DIR = Path("/content/fonts")
TEXTFX_OUT_DIR = Path("data/train_textfx")            # make_textfx.run() çıktısı (yerel, WORKDIR'e göre)
EXPORT_DIR = "/content/birefnet_format_textfx"        # export_birefnet.export() çıktısı
TEXTFX_COUNTS = {"text": 4000, "fx": 3500}            # illustration ToonOut boyutundan OTOMATİK
V4_NEW_CATEGORIES = ("text", "fx", "illustration")

STATUS_DIR = Path(DRIVE_ROOT) / DRIVE_STATUS_SUBDIR
LOG_PATH = STATUS_DIR / "log.txt"
STATUS_PATH = STATUS_DIR / "status.json"

# scripts/ bir paket değil — make_textfx/export_birefnet'i import edebilmek
# için mutlak yolu sys.path'e ekliyoruz (bkz. v3_veri_guncelleme_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import CATEGORIES, load_manifest  # noqa: E402  (pip install -e . ile kurulu paket)
import training.train_colab_lib as tcl  # noqa: E402  (torch-free, test edilebilir mantık)


# ==========================================================================
# Durum raporlama — `v3_veri_guncelleme_hucresi.py::report`'la BİREBİR AYNI.
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
# Stage "env" — Drive bağlama (HERŞEYDEN önce, STATUS_DIR Drive'da!) + repo
# git pull (idempotent) + ortam sağlık kontrolü. Kaynak:
# v3_veri_guncelleme_hucresi.py::stage0_env_sanity; git pull v4'e özgü ek —
# make_textfx.py paralel çalışmada eklendiği için eski klon en sık hata kaynağı.
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
            print("HF_TOKEN Colab Secrets'tan alındı.")
    except Exception as e:
        print(f"HF_TOKEN alınamadı (Secrets'ta yok veya erişim izni verilmedi): {e}")


def _git_pull_idempotent() -> None:
    """Repo'yu günceller — `git pull --ff-only` zaten günceldeyse no-op
    (idempotent); ağ yoksa/çakışma varsa UYARI verip devam eder (make_textfx
    eksikse stage_textfx zaten net mesajla durduracak)."""
    try:
        r = subprocess.run(
            ["git", "-C", WORKDIR, "pull", "--ff-only"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"git pull: rc={r.returncode} {r.stdout.strip() or r.stderr.strip()}")
        if r.returncode != 0:
            print("UYARI: git pull başarısız — repo eski kalmış olabilir; make_textfx.py "
                  "eksikse aşağıda net hatayla durulacak.")
    except Exception as e:
        print(f"UYARI: git pull çalıştırılamadı ({e}) — mevcut klonla devam ediliyor.")


def stage0_env_sanity() -> dict:
    # Drive HERŞEYDEN ÖNCE bağlanır (report() dahil — STATUS_DIR Drive'da!);
    # drive.mount idempotenttir. Kaynak: v3 hücresi aynı aşama.
    from google.colab import drive

    drive.mount("/content/drive")
    assert Path(DRIVE_ROOT).is_dir(), f"Drive bağlanamadı: {DRIVE_ROOT} yok"

    report("env", "running")
    os.chdir(WORKDIR)
    _git_pull_idempotent()
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} dosya")
    for name in ("trans460_train", "him2k_raw", "backgrounds", "toonout", "fonts"):
        if counts[name] == 0:
            print(f"NOT: {name} şu an boş — 'downloads'/'fonts' aşamasında indirilecek "
                  f"(taze VM'de normal durum).")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage "downloads" — YALNIZ v4'ün gerektirdiği kaynaklar, İDEMPOTENT:
#   - BG-20k arka plan havuzu (text/fx kompozitleri için) — kaynak:
#     v3_veri_guncelleme_hucresi.py::_download_bg_pool (kopya).
#   - Transparent-460 (fx için transparent foreground) — kaynak: aynı dosya
#     _download_trans460 (kopya).
#   - HIM2K (fx için general foreground; gdown + images/alphas birleştirme) —
#     kaynak: aynı dosya _gdown_extract/discover_him2k_dirs/merge_him2k (kopya).
#   - ToonOut (illustration) — v4'e ÖZGÜ, yalnız train split'i (test'e DOKUNMA).
# ==========================================================================
RAW = Path("data/raw_train")


def _load_source_defs() -> dict:
    with open("data/train_sources.json") as f:
        return {s["name"]: s for s in json.load(f)["sources"]}


def _download_bg_pool(source_defs: dict) -> int:
    """Kaynak: v3_veri_guncelleme_hucresi.py::_download_bg_pool (kökeni
    prepare_data_colab.ipynb hücre (e)/15) — BG-20k'dan BG_POOL_SIZE arka plan,
    kümülatif sayaçla idempotent."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    bg_dir = Path("data/backgrounds")
    bg_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(bg_dir.iterdir()))
    if existing >= BG_POOL_SIZE:
        print(f"data/backgrounds zaten {existing} görsel içeriyor (>= {BG_POOL_SIZE}); indirme atlanıyor.")
        return existing

    bg_spec = source_defs["bg_20k"]
    fs = HfFileSystem()
    pattern = bg_spec["split_patterns"][0]
    parts = sorted(fs.glob(f"datasets/{bg_spec['hf_repo']}/{pattern}"))

    written = existing  # KÜMÜLATİF sayaç — parça sınırlarında sıfırlanmaz
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

    print(f"data/backgrounds: {written} arka plan görseli.")
    return written


def _download_trans460(source_defs: dict) -> int:
    """Kaynak: v3_veri_guncelleme_hucresi.py::_download_trans460 (kopya) —
    fx foreground kaynağı: fg/ + alpha/ (saydam objeler)."""
    from huggingface_hub import snapshot_download

    spec = source_defs["transparent_460_train"]
    trans_out = RAW / "trans460_train"
    existing = len(list((trans_out / "fg").iterdir())) if (trans_out / "fg").exists() else 0
    expected = spec.get("full_pair_count") or 0
    if expected and existing >= 0.9 * expected:
        print(f"trans460_train: diskte zaten {existing} görsel (>= %90 x {expected}); indirme atlanıyor.")
        return existing

    trans_dir = snapshot_download(repo_id=spec["hf_repo"], repo_type="dataset", allow_patterns=["Train/*"])
    if trans_out.exists():
        shutil.rmtree(trans_out)
    shutil.copytree(Path(trans_dir) / "Train" / "fg", trans_out / "fg")
    shutil.copytree(Path(trans_dir) / "Train" / "alpha", trans_out / "alpha")
    total = len(list((trans_out / "fg").iterdir()))
    print(f"transparent_460_train: {total} görsel -> {trans_out}")
    return total


def _ensure_gdown() -> None:
    """Kaynak: v3_veri_guncelleme_hucresi.py::_ensure_gdown (kopya)."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)


def _gdown_extract(drive_id: str, out_dir: Path, label: str) -> bool:
    """Kaynak: v3_veri_guncelleme_hucresi.py::_gdown_extract (kopya) —
    başarısızlıkta False döner (pipeline'ı durdurmaz), out_dir doluysa atlar."""
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{label}: {out_dir} zaten dolu; indirme atlanıyor.")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    try:
        import gdown

        gdown.download(id=drive_id, output=str(zip_path), quiet=False)
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        print(f"{label}: indirildi ve açıldı -> {out_dir}")
        return True
    except Exception as e:
        print(f"UYARI: {label} indirilemedi ({e}) — bu kaynak ATLANACAK.")
        return False


def _walk_dirs(root: Path, max_depth: int = 4) -> list[dict]:
    """Kaynak: v3_veri_guncelleme_hucresi.py::_walk_dirs (kopya)."""
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
    """Kaynak: v3_veri_guncelleme_hucresi.py::discover_him2k_dirs (kopya)."""
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
    """Kaynak: v3_veri_guncelleme_hucresi.py::merge_him2k (kopya) — instance
    alfalarını max-birleştirip {im,gt} çiftleri üretir (fx general foreground)."""
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
    """gdown ile HIM2K'yı indirip images/alphas'ı {im,gt} olarak birleştirir —
    idempotent (merged zaten doluysa atlar). fx foreground'un GENERAL ayağı;
    inmezse UYARI verilip yalnız trans460 ile devam edilir (make_textfx'e
    yalnız var olan fg dizinleri geçilir)."""
    _ensure_gdown()
    ok = _gdown_extract(source_defs["him2k"]["drive_id"], RAW / "him2k_raw", "HIM2K")
    if not ok:
        return 0
    out_root = RAW / "him2k_merged"
    existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
    existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
    if existing_gt > 0 and existing_gt == existing_im:
        print(f"{out_root} zaten {existing_gt} çift içeriyor; birleştirme atlanıyor.")
        return existing_gt
    dirs = discover_him2k_dirs(RAW / "him2k_raw")
    if dirs is None:
        print("HIM2K images/alphas dizin çifti bulunamadı — general foreground ATLANACAK.")
        return 0
    n = merge_him2k(dirs[0], dirs[1], out_root)
    print(f"HIM2K birleştirildi: {n} çift -> {out_root}")
    return n


def _download_toonout() -> int:
    """v4'e ÖZGÜ: HuggingFace `joelseytre/toonout` deposunun YALNIZ train
    split'ini indirir (allow_patterns=["train/*"] — test split'i BİLEREK hiç
    indirilmez, o illustration benchmark'ı için ayrılacak; val de gereksiz) ve
    `/content/downloads/toonout/{im,gt}` olarak normalize eder (im/gt/an alt
    klasör yapısından an/ kullanılmaz — make_textfx yalnız im+gt bekler).
    İdempotent: hedef im/ zaten doluysa ve im/gt sayıları eşitse atlar."""
    from huggingface_hub import snapshot_download

    out_im = TOONOUT_DIR / "im"
    out_gt = TOONOUT_DIR / "gt"
    existing_im = len(list(out_im.iterdir())) if out_im.exists() else 0
    existing_gt = len(list(out_gt.iterdir())) if out_gt.exists() else 0
    if existing_im > 0 and existing_im == existing_gt:
        print(f"toonout: {TOONOUT_DIR} zaten {existing_im} çift içeriyor; indirme atlanıyor.")
        return existing_im

    snap = Path(snapshot_download(repo_id=TOONOUT_HF_REPO, repo_type="dataset",
                                  allow_patterns=["train/*"]))
    train_root = snap / "train"
    assert train_root.is_dir(), (
        f"ToonOut snapshot'ında train/ bulunamadı: {snap} — repo yapısı değişmiş olabilir "
        f"(beklenen: im/gt/an alt klasörlü train/val/test split'leri)."
    )
    src_im, src_gt = train_root / "im", train_root / "gt"
    assert src_im.is_dir() and src_gt.is_dir(), (
        f"ToonOut train split'inde im/ ve gt/ bekleniyordu, bulunan alt dizinler: "
        f"{[p.name for p in train_root.iterdir() if p.is_dir()]}"
    )

    out_im.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)
    gt_by_stem = {p.stem: p for p in src_gt.iterdir() if p.is_file()}
    copied = 0
    for img in sorted(p for p in src_im.iterdir() if p.is_file()):
        gt = gt_by_stem.get(img.stem)
        if gt is None:
            continue  # gt'siz görsel eğitime giremez
        dst_i, dst_g = out_im / img.name, out_gt / gt.name
        if dst_i.exists() and dst_g.exists():
            copied += 1
            continue
        shutil.copy2(img, dst_i)
        shutil.copy2(gt, dst_g)
        copied += 1
    print(f"toonout (train split): {copied} im/gt çifti -> {TOONOUT_DIR} (test split'e DOKUNULMADI).")
    assert copied > 0, "ToonOut train split'inden hiç im/gt çifti çıkarılamadı!"
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
        print(f"UYARI: transparent_460 indirilemedi ({e}); mevcutsa diskteki kullanılacak.")
        results["trans460"] = -1

    results["him2k_merged"] = _ensure_him2k_merged(source_defs)

    results["toonout"] = _download_toonout()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    print("İndirme sonrası dosya sayıları:", counts)
    report("downloads", "done", results=results, counts=counts)
    return results


# ==========================================================================
# Stage "fonts" — v4'e ÖZGÜ: Google Fonts deposundan (github.com/google/fonts)
# ~20 OFL lisanslı TTF indirir -> /content/fonts. Her font tek tek try/except'li
# (URL çürümesi tüm hücreyi durdurmasın); hiçbiri inmezse Colab VM'sinde hazır
# bulunan DejaVu fontlarına düşülür. text kategorisi üretimi font ÇEŞİTLİLİĞİNE
# ihtiyaç duyar (tek fontla üretilen yazılar modele genellenmez).
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
    "/usr/share/fonts/truetype/dejavu/DejaVu*.ttf",  # Colab/Ubuntu standart yolu
    "/usr/share/fonts/TTF/DejaVu*.ttf",
]


def stage_fonts() -> int:
    report("fonts", "running")
    FONT_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for rel in GOOGLE_FONT_PATHS:
        # Dosya adındaki [wght] köşeli ayraçları URL'de yüzde-kodlanmalı;
        # yerelde ise ayraçsız sade bir ad kullanıyoruz (glob desenleriyle
        # çakışmasın diye).
        target = FONT_DIR / Path(rel).name.replace("[", "_").replace("]", "_")
        if target.exists() and target.stat().st_size > 0:
            ok += 1
            continue
        url = _GF_RAW + urllib.parse.quote(rel)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            assert data[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true"), "TTF/OTF imzası değil"
            target.write_bytes(data)
            ok += 1
        except Exception as e:
            failed.append((rel, str(e)))
            print(f"UYARI: font indirilemedi ({rel}): {e}")

    if ok < 5:
        print(f"Yalnız {ok} Google Fonts fontu inebildi — DejaVu fallback'ine düşülüyor.")
        import glob as _glob

        for pattern in DEJAVU_GLOBS:
            for p in _glob.glob(pattern):
                dst = FONT_DIR / Path(p).name
                if not dst.exists():
                    shutil.copy2(p, dst)

    total = len([p for p in FONT_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"}])
    print(f"/content/fonts: {total} font hazır ({ok} Google Fonts, {len(failed)} başarısız).")
    if total == 0:
        raise RuntimeError(
            "Hiç font indirilemedi ve DejaVu fallback'i de bulunamadı — text kategorisi "
            "üretilemez. Ağ bağlantısını kontrol edin veya /content/fonts'a elle TTF koyun."
        )
    report("fonts", "done", downloaded=ok, failed=len(failed), total=total)
    return total


# ==========================================================================
# Stage "textfx" — ÜRETİM: scripts/make_textfx.py (PARALEL çalışmada yazılıyor —
# burada yalnız belgelenen imzası varsayılır: run(out_dir, bg_dir, fg_dirs,
# toonout_dir, font_dir, seed, counts)). İmza/import uyuşmazlığında NET Türkçe
# hata mesajıyla durulur, sessizce yarım veri üretilmez.
# ==========================================================================
def stage_textfx() -> dict:
    report("textfx", "running")

    # Repo güncelliği guard'ı: text/fx kategorileri benchmark.testset.CATEGORIES'e
    # make_textfx çalışmasıyla ekleniyor — eski klonda manifest doğrulaması
    # (append_entries/load_manifest) "bilinmeyen kategori" ile patlardı; nedeni
    # burada, en başta söylüyoruz.
    missing_cats = {"text", "fx", "illustration"} - CATEGORIES
    if missing_cats:
        raise RuntimeError(
            f"benchmark.testset.CATEGORIES şu kategorileri tanımıyor: {sorted(missing_cats)} — "
            f"repo klonunuz eski görünüyor (make_textfx çalışması bunları ekliyor). "
            f"'git -C {WORKDIR} pull' çalıştırıp hücreyi yeniden koşun."
        )

    try:
        import make_textfx as mtx  # scripts/ sys.path'te
    except ImportError as e:
        raise RuntimeError(
            f"scripts/make_textfx.py import edilemedi ({e}) — bu script paralel bir çalışmada "
            f"yazılıyor; repo'nuz güncel mi? 'git -C {WORKDIR} pull' deneyin. Script henüz "
            f"merge edilmediyse bu hücreyi make_textfx hazır olunca koşun."
        ) from e

    fg_dirs = [d for d in (RAW / "trans460_train", RAW / "him2k_merged") if d.is_dir()]
    assert fg_dirs, (
        "fx için hiç foreground kaynağı yok (trans460_train ve him2k_merged ikisi de eksik) — "
        "'downloads' aşaması loglarını inceleyin."
    )

    try:
        counts = mtx.run(
            out_dir=TEXTFX_OUT_DIR,
            bg_dir=Path("data/backgrounds"),
            fg_dirs=fg_dirs,
            toonout_dir=TOONOUT_DIR,
            font_dir=FONT_DIR,
            seed=SEED,
            counts=TEXTFX_COUNTS,  # illustration BİLEREK yok — ToonOut boyutundan otomatik
        )
    except TypeError as e:
        raise RuntimeError(
            f"make_textfx.run() beklenen imzayla çağrılamadı ({e}) — bu hücre "
            f"run(out_dir, bg_dir, fg_dirs, toonout_dir, font_dir, seed, counts) imzasını "
            f"varsayar (paralel çalışmanın belgelenen sözleşmesi). scripts/make_textfx.py'nin "
            f"güncel imzasına bakıp çağrıyı uyarlayın."
        ) from e

    print("make_textfx.run() kategori bazlı üretim:", counts)

    # Manifest guard'ı (v3 dersi): boş/eksik manifest'le export'a GEÇME.
    out_manifest = TEXTFX_OUT_DIR / "manifest.jsonl"
    total_pairs = tcl.ensure_manifest_pairs(out_manifest)

    # PRE-FLIGHT: üretilen sayılar kategori bazında (manifest'ten, otoriter).
    rows = load_manifest(str(out_manifest))
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"PRE-FLIGHT — {out_manifest}: toplam {total_pairs} GT'li çift, kategori bazında:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")
    low = {c: by_cat.get(c, 0) for c in V4_NEW_CATEGORIES if by_cat.get(c, 0) < 100}
    if low:
        print(f"UYARI: şu v4 kategorileri 100 örneğin ALTINDA: {low} — train_colab.ipynb'nin "
              f"v4 ön-uçuş guard'ı bu durumda GPU koşusunu durduracaktır.")

    report("textfx", "done", counts=counts, by_category=by_cat, total_pairs=total_pairs)
    return by_cat


# ==========================================================================
# Stage "export" — v3 kalıbı (export_birefnet.export() DEĞİŞMEDİ): taze/boş
# bir yerel dizine karşı çalıştığından diskte yalnız yeni textfx dosyaları
# oluşur (kaynak manifest zaten yalnız text/fx/illustration satırları içeriyor).
# split_name="TRAIN": yeni stemler HER ZAMAN TRAIN'e gider, VAL'e HİÇBİR yeni
# stem gitmez (mevcut kural).
# ==========================================================================
def stage_export_textfx() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ sys.path'te

    stats = eb.export(
        manifest_path=str(TEXTFX_OUT_DIR / "manifest.jsonl"),
        out_dir=EXPORT_DIR,
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — v3 kalıbı: var olan Drive TRAIN'e MERGE (dirs_exist_ok=
# True, hiçbir dosya SİLİNMEZ/üzerine yazılmaz; src kökündeki KISMİ stats.json
# Drive'daki otoriter TAM stats.json'u ezmesin diye KOPYALANMAZ — v3 reviewer
# bulgusu #1) + kompozit manifest'e APPEND (tcl.merge_composite_manifest,
# dedupe'lu — tam üzerine yazma YOK).
# ==========================================================================
def stage_drive_copy_textfx() -> None:
    report("drive_copy", "running")
    src = Path(EXPORT_DIR)
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Drive'da beklenen v1-v3 TRAIN verisi bulunamadı: {dst_train_im} / {dst_train_gt} — "
        f"bu hücre yalnız MEVCUT bir veri setine v4 (text/fx/illustration) EKLEMEK içindir, "
        f"sıfırdan veri seti oluşturmak için colab_devam_hucresi.py kullanılmalı."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    existing_dst_im_stems = {p.stem for p in dst_train_im.iterdir()}
    new_stems = {p.stem for p in src_im_files} - existing_dst_im_stems
    expected_growth = len(new_stems)

    pre_im, pre_gt = len(list(dst_train_im.iterdir())), len(list(dst_train_gt.iterdir()))
    print(f"Merge öncesi Drive TRAIN: im={pre_im}, gt={pre_gt} — beklenen artış: {expected_growth}")

    # YALNIZ TRAIN/ alt ağacı kopyalanır — src kökündeki stats.json BİLİNÇLİ
    # OLARAK KOPYALANMAZ (v3 fix'i: kısmi stats.json otoriter TAM stats.json'u
    # sessizce EZERDİ).
    print(f"Kopyalanıyor (MERGE, silme yok, yalnız TRAIN/): {src / 'TRAIN'} -> {dst / 'TRAIN'}")
    shutil.copytree(src / "TRAIN", dst / "TRAIN", dirs_exist_ok=True)

    post_im, post_gt = len(list(dst_train_im.iterdir())), len(list(dst_train_gt.iterdir()))
    print(f"Merge sonrası Drive TRAIN: im={post_im}, gt={post_gt}")

    assert post_im - pre_im == expected_growth, (
        f"im/ büyümesi beklenenle uyuşmuyor: {post_im - pre_im} != {expected_growth}"
    )
    assert post_gt - pre_gt == expected_growth, (
        f"gt/ büyümesi beklenenle uyuşmuyor: {post_gt - pre_gt} != {expected_growth}"
    )
    assert len(src_im_files) == len(src_gt_files), "yerel textfx export'unda im/gt sayıları uyuşmuyor!"

    comp_manifest_local = TEXTFX_OUT_DIR / "manifest.jsonl"
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} yeni satır eklendi (Drive'daki mevcut "
          f"v1-v3 satırları KORUNDU, üzerine yazılmadı).")
    assert n_appended == expected_growth, (
        f"manifest ekleme sayısı ({n_appended}) dosya büyümesiyle ({expected_growth}) tutarsız — "
        f"stem/id eşlemesi kontrol edilmeli."
    )

    print("\nBÜTÜNLÜK KONTROLÜ BAŞARILI — v4 (text/fx/illustration) verisi Drive'a MERGE edildi.")
    report(
        "drive_copy", "done",
        added_files=expected_growth, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    stage0_env_sanity()        # Drive mount + git pull BURADA — Drive'a dokunan her şeyden önce
    stage_downloads()          # ToonOut(train) + BG-20k + trans460 + HIM2K (idempotent)
    stage_fonts()              # ~20 OFL Google Fonts -> /content/fonts (DejaVu fallback)
    stage_textfx()             # make_textfx.run() + manifest guard + kategori pre-flight
    stage_export_textfx()
    stage_drive_copy_textfx()
    report("ALL", "done")


try:
    main()
except Exception:
    tb = traceback.format_exc()
    report("FATAL", "error", traceback=tb)
    raise
