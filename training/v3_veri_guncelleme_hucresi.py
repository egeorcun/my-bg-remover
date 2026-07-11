"""V3 VERİ GÜNCELLEME HÜCRESİ — taze bir (ÜCRETSİZ, CPU yeterli — GPU GEREKMEZ)
Colab oturumunda, mevcut Drive veri setine (`bg-remover-data/TRAIN`) yalnız YENİ
`_o00` (orijinal arka plan) kopyalarını ekler; v1/v2'nin ~28k'lik tam kompozit
setini YENİDEN ÜRETMEZ (bkz. `scripts/make_composites.py` modül docstring'i "v3"
notu — over-deletion'ın kalıcı olma nedeni camouflage dışındaki kategorilerin
yalnız sentetik arka planlarda eğitilmesiydi; bu hücre her kategoriye orijinal
arka planı koruyan 1'er ek kopya ekleyerek domain gap'i kapatır).

KAYNAK / ATIF: bu dosyanın env/downloads/manifest aşamaları (`report`,
`stage0_env_sanity`, `stage1_bg_pool`, `_walk_dirs`, `discover_cod10k`,
`stage2_discover_structure`, `discover_him2k_dirs`, `merge_him2k`,
`stage3_merge_him2k`, `stage4_build_manifest`) `training/colab_devam_hucresi.py`
dosyasından BİREBİR KOPYALANDI (aynı kanıtlanmış indirme/keşif/manifest mantığı,
drift önleme için tek doğruluk kaynağı o dosyadır — burada tekrar yazılmasının
tek nedeni, bu hücrenin de `colab_devam_hucresi.py` gibi tek başına yapıştırılıp
çalıştırılabilir olması gerekmesi, bir modül olarak import edilmemesi). export/
drive_copy aşamaları v3'e özgü MERGE mantığıyla YENİDEN YAZILDI (aşağıya bakın).

ÖN KOŞULLAR (colab_devam_hucresi.py ile AYNI varsayım): repo `/content/
my-bg-remover`'da klonlanmış, Drive bağlı, `pip install -e .` yapılmış,
`data/raw_train/{dis5k,camo,p3m,trans460_train,cod10k_raw,him2k_raw}` bu VM'e
zaten indirilmiş (prepare_data_colab.ipynb / colab_devam_hucresi.py'nin yaptığı
gibi) — Stage 4 manifest'i bu ham verilerden DETERMİNİSTİK olarak yeniden
inşa eder, bu yüzden `data/train/manifest.jsonl`'deki `id`'ler önceki v1/v2
koşularıyla BİREBİR aynı çıkar (aynı `build_trainset.py` mantığı, aynı sıralama).
Drive'da `bg-remover-data/TRAIN/{im,gt}` (v1/v2'nin tam kompozit çıktısı) ve
`bg-remover-status/val_stems.json` (VAL bölünmesi) ZATEN mevcut olmalı.

FARK (v1/v2'nin `colab_devam_hucresi.py`'sinden):
1. Stage 4'ten sonra Drive'daki `val_stems.json` okunur, VAL'e sızmaması gereken
   KAYNAK id'ler türetilir (`_v<NN>`/`_o<NN>` son eki çıkarılarak — bkz.
   `training.train_colab_lib.strip_composite_copy_suffix`) ve
   `make_composites.run()`a `exclude_source_ids` olarak geçilir.
2. Taze bir VM'de `data/train_composites/` (v1/v2'nin ~28k'lik tam kompozit
   çıktısı) YOKTUR — onu yeniden üretmek saatler sürer. Bunun yerine `run()`un
   `only_original_bg=True` bayrağıyla YALNIZ `_o00` seti, AYRI bir dizine
   (`data/train_composites_o/`) üretilir (~14k civarı, hızlı — CPU'da bile
   dakikalar mertebesinde, compose YOK yalnız augment).
3. `export_birefnet.export()` bu AYRI (taze, boş) yerel dizine karşı çalıştığından
   diskte yalnız `_o00` dosyaları oluşur (idempotent skip-existing zaten
   `export_birefnet.py`'de var — burada ekstra bir şey gerekmez, kaynak manifest
   zaten yalnız `_o00` satırları içeriyor).
4. Drive'a kopyalama YALNIZ `TRAIN/` alt ağacının `shutil.copytree(...,
   dirs_exist_ok=True)` MERGE'i (var olan `_v<NN>` dosyaları SİLİNMEZ/ÜZERİNE
   YAZILMAZ, yalnız yeni `_o00` dosyaları eklenir; src kökündeki KISMİ — yalnız
   _o00'lu — `stats.json` Drive'daki otoriter TAM stats.json'u ezmesin diye
   KOPYALANMAZ) + kompozit manifest'in Drive kopyasına (`train_composites_
   manifest.jsonl`) yalnız YENİ id'lerin APPEND edilmesi (tam üzerine yazma
   DEĞİL — devam hücresinin `shutil.copy2` ile TAM üzerine yazması burada YANLIŞ
   olurdu, Drive'daki dosya zaten v1/v2'nin tüm `_v<NN>` satırlarını içeriyor).
   Bütünlük kontrolü: Drive TRAIN'deki dosya sayısının ARTIŞI, yerelde üretilen
   ama Drive'da henüz olmayan `_o00` dosya sayısına birebir eşit olmalı.

Durum takibi `colab_devam_hucresi.py` ile AYNI mekanizma (`report()` ->
`bg-remover-status/log.txt` + `status.json`) — aşamalar: env, downloads,
manifest, composites_o, export, drive_copy, (bitişte) ALL.
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

# Transparent-460/HIM2K'da 100MP+ görseller var; PIL'in 179MP "decompression
# bomb" hata eşiğini aşabiliyor. Veri güvenilir akademik setlerden geldiği için
# limit kaldırılıyor (bkz. colab_devam_hucresi.py aynı satır).
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np  # noqa: E402  (MAX_IMAGE_PIXELS PIL importundan/atamasından SONRA gelmeli)
from PIL import Image  # noqa: E402

# --- Sabitler (colab_devam_hucresi.py ile AYNI — bkz. o dosyanın "Sabitler" bölümü) ---
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

# scripts/ bir paket değil — build_trainset/make_composites/export_birefnet'i
# import edebilmek için mutlak yolu sys.path'e ekliyoruz (bkz. colab_devam_hucresi.py).
SCRIPTS_DIR = str(Path(WORKDIR) / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from benchmark.testset import append_entries, load_manifest  # noqa: E402  (pip install -e . ile kurulu paket)
import training.train_colab_lib as tcl  # noqa: E402  (aynı pip install -e . -- torch-free, test edilebilir mantık)


# ==========================================================================
# Durum raporlama — `colab_devam_hucresi.py::report`'la BİREBİR AYNI (kaynak:
# training/colab_devam_hucresi.py, satır ~71).
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
# Stage "env" — ortam sağlık kontrolü (kaynak: colab_devam_hucresi.py::stage0_env_sanity)
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
            print("HF_TOKEN Colab Secrets'tan alındı.")
    except Exception as e:
        print(f"HF_TOKEN alınamadı (Secrets'ta yok veya erişim izni verilmedi): {e}")


def stage0_env_sanity() -> dict:
    report("env", "running")
    os.chdir(WORKDIR)
    _setup_hf_env()

    counts = {name: _count_files(Path(rel)) for name, rel in RAW_DIR_CHECKS.items()}
    for name, c in counts.items():
        print(f"{name}: {c} dosya")
    for name in ("dis5k", "camo", "p3m", "trans460_train", "cod10k_raw", "him2k_raw"):
        if counts[name] == 0:
            print(f"UYARI: {name} boş görünüyor — Stage 'manifest' bu kategoriyi atlayacak "
                  f"(ön koşul: bu VM'e ham veri önceden indirilmiş olmalı, bkz. modül docstring'i).")

    report("env", "done", cwd=str(Path.cwd()), counts=counts)
    return counts


# ==========================================================================
# Stage "downloads" — arka plan havuzu (kaynak: colab_devam_hucresi.py::stage1_bg_pool)
# ==========================================================================
def stage1_bg_pool() -> int:
    report("downloads", "running")
    bg_dir = Path("data/backgrounds")
    bg_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(bg_dir.iterdir()))
    if existing >= BG_POOL_SIZE:
        print(f"data/backgrounds zaten {existing} görsel içeriyor (>= {BG_POOL_SIZE}); indirme atlanıyor.")
        report("downloads", "done", count=existing, skipped=True)
        return existing

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with open("data/train_sources.json") as f:
        source_defs = {s["name"]: s for s in json.load(f)["sources"]}
    bg_spec = source_defs["bg_20k"]

    fs = HfFileSystem()
    pattern = bg_spec["split_patterns"][0]
    parts = sorted(fs.glob(f"datasets/{bg_spec['hf_repo']}/{pattern}"))

    written = existing
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
    report("downloads", "done", count=written)
    return written


# ==========================================================================
# Stage "manifest" — COD10K/HIM2K keşif+birleştirme + tam manifest (kaynak:
# colab_devam_hucresi.py::{discover_cod10k, stage2_discover_structure,
# discover_him2k_dirs, merge_him2k, stage3_merge_him2k, stage4_build_manifest}).
# Tek bir report("manifest", ...) çifti altında toplanır (görev madde: report()
# aşamaları env/downloads/manifest/composites_o/export/drive_copy/ALL).
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
    """COD10K keşfi + HIM2K birleştirme + tam manifest inşası — TEK bir
    `report("manifest", ...)` çifti altında (bkz. modül docstring'i)."""
    report("manifest", "running")
    import build_trainset as bt  # scripts/ sys.path'te

    # --- COD10K keşfi (kaynak: stage2_discover_structure) ---
    cod_raw_dir = Path("data/raw_train/cod10k_raw")
    cod10k_info = None
    if cod_raw_dir.exists():
        cod10k_info = discover_cod10k(cod_raw_dir)
        if cod10k_info:
            print(f"COD10K seçilen çift: img={cod10k_info['img_dir']}  gt={cod10k_info['gt_dir']}  "
                  f"örtüşen stem={cod10k_info['overlap']}  belirsiz={cod10k_info['ambiguous']}")
        else:
            print("COD10K için örtüşen img/gt dizin çifti bulunamadı.")
    else:
        print("data/raw_train/cod10k_raw yok — COD10K atlanıyor.")

    # --- HIM2K birleştirme (kaynak: stage3_merge_him2k) ---
    him2k_raw_dir = Path("data/raw_train/him2k_raw")
    him2k_count = 0
    if him2k_raw_dir.exists():
        dirs = discover_him2k_dirs(him2k_raw_dir)
        if dirs is None:
            print("HIM2K images/alphas dizin çifti bulunamadı — atlanıyor.")
        else:
            images_dir, alphas_dir = dirs
            out_root = Path("data/raw_train/him2k_merged")
            existing_gt = len(list((out_root / "gt").iterdir())) if (out_root / "gt").exists() else 0
            existing_im = len(list((out_root / "im").iterdir())) if (out_root / "im").exists() else 0
            if existing_gt > 0 and existing_gt == existing_im:
                print(f"data/raw_train/him2k_merged zaten {existing_gt} çift içeriyor; birleştirme atlanıyor.")
                him2k_count = existing_gt
            else:
                him2k_count = merge_him2k(images_dir, alphas_dir, out_root)
                print(f"HIM2K birleştirildi: {him2k_count} çift -> {out_root}")
    else:
        print("data/raw_train/him2k_raw yok — HIM2K atlanıyor (general kategorisi opsiyonel).")

    # --- Tam manifest (kaynak: stage4_build_manifest) — DETERMİNİSTİK: aynı ham
    # veri + aynı build_trainset.py mantığı -> v1/v2 ile BİREBİR aynı id'ler. ---
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
        print(f"{name} ({category}): {len(rows)} çift")
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
        print(f"dis5ktr ({category}): {c} çift")

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
        print("cod10ktr: atlandı (dizin bulunamadı)")

    if him2k_count > 0:
        _run("him2k", "data/raw_train/him2k_merged/im/*", "data/raw_train/him2k_merged/gt/*", "general")
    else:
        counts["him2k"] = 0
        print("him2k: atlandı (birleştirme yapılamadı)")

    report("manifest", "done", counts=counts)
    return counts


# ==========================================================================
# Stage "composites_o" — YENİ (v3'e özgü): yalnız _o00 (orijinal arka plan)
# kopyalarını, VAL'e sızmayacak şekilde üretir (bkz. modül docstring'i madde 1-2).
# Kaynak id türetme mantığı (`strip_composite_copy_suffix`/
# `derive_val_excluded_source_ids`) `training.train_colab_lib`de (torch-free,
# birim testli) — bkz. o modülün "7) v3" bölümü.
# ==========================================================================
def load_val_excluded_source_ids(val_stems_path: Path) -> tuple[set[str], list[str]]:
    """Drive'daki `val_stems.json`ı (`tcl.load_or_create_val_split`in yazdığı
    `{"val_stems": [...]}` formatı) okuyup `tcl.derive_val_excluded_source_ids`
    ile `(kaynak id kümesi, eşleşmeyen stem listesi)` çiftine çevirir — kaynak
    id'ler `_o00` üretiminden hariç tutulur (VAL sızıntı koruması, bkz. görev
    "VAL leakage guard"); eşleşmeyen stem'ler için koruma BAYPAS edilmiş olur
    (bkz. `tcl.strip_composite_copy_suffix` docstring'i), çağıran uyarmalı."""
    if not val_stems_path.exists():
        print(f"UYARI: {val_stems_path} bulunamadı — hiçbir kaynak hariç tutulmuyor "
              f"(VAL bölünmesi henüz yapılmamış olabilir; bu durumda _o00 üretimi TÜM "
              f"kategorilere uygulanır, sızıntı riski yalnız VAL_HOLDOUT'un ZATEN "
              f"var olduğu normal senaryoda geçerlidir).")
        return set(), []
    payload = json.loads(val_stems_path.read_text())
    return tcl.derive_val_excluded_source_ids(payload.get("val_stems", []))


def stage_composites_o() -> dict:
    report("composites_o", "running")
    import make_composites as mc  # scripts/ sys.path'te

    excluded, unmatched = load_val_excluded_source_ids(VAL_STEMS_PATH)
    print(f"VAL sızıntı koruması: {len(excluded)} kaynak id _o00 üretiminden hariç tutuluyor.")
    if unmatched:
        print("=" * 72)
        print(f"!!! UYARI — VAL SIZINTI KORUMASI KISMEN BAYPAS: {len(unmatched)} val stem'i "
              f"_v<NN>/_o<NN> son ek deseniyle EŞLEŞMEDİ. Bu stem'lerin ASIL kaynak "
              f"id'leri hariç tutulaMADI — o kaynakların _o00 kopyaları eğitim setine "
              f"üretilecek ve aynı görsel hem TRAIN hem VAL'de görülecek (sızıntı). "
              f"İlk 10 eşleşmeyen stem: {unmatched[:10]}")
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
    print("Kategori bazlı üretilen _o00 sayısı:", counts)

    # Bütünlük: beklenen toplam = (NO_COMPOSE_CATEGORIES dışı + gt_alpha'lı +
    # hariç tutulmamış) kaynak satır sayısı x ORIGINAL_BG_COPIES (formül, rapora
    # da yazılır).
    source_rows = load_manifest("data/train/manifest.jsonl")
    eligible = [
        r for r in source_rows
        if r.get("gt_alpha") and r["category"] not in mc.NO_COMPOSE_CATEGORIES and r["id"] not in excluded
    ]
    expected_total = len(eligible) * mc.ORIGINAL_BG_COPIES

    out_manifest = Path("data/train_composites_o/manifest.jsonl")
    actual_total = len(load_manifest(str(out_manifest))) if out_manifest.exists() else 0
    print(f"composites_o bütünlük: beklenen={expected_total}, gerçek={actual_total} "
          f"(kaynak satır x ORIGINAL_BG_COPIES={mc.ORIGINAL_BG_COPIES}).")
    assert actual_total == expected_total, (
        f"composites_o manifest toplamı beklenenle uyuşmuyor: {actual_total} != {expected_total} "
        f"— make_composites.run() mantığı veya exclude_source_ids kontrol edilmeli."
    )

    report("composites_o", "done", counts=counts, expected_total=expected_total, actual_total=actual_total)
    return counts


# ==========================================================================
# Stage "export" — YENİ (v3'e özgü, ama export_birefnet.export() DEĞİŞMEDİ):
# taze/boş bir yerel dizine karşı çalıştığından diskte yalnız _o00 dosyaları
# oluşur (kaynak manifest zaten yalnız _o00 satırları içeriyor).
# ==========================================================================
def stage_export_o() -> dict:
    report("export", "running")
    import export_birefnet as eb  # scripts/ sys.path'te

    stats = eb.export(
        manifest_path="data/train_composites_o/manifest.jsonl",
        out_dir="/content/birefnet_format_o",
        split_name="TRAIN",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    report("export", "done", stats=stats)
    return stats


# ==========================================================================
# Stage "drive_copy" — YENİ (v3'e özgü): var olan Drive TRAIN'e MERGE (dirs_
# exist_ok=True, hiçbir dosya SİLİNMEZ) + kompozit manifest'e APPEND (dedupe'lu,
# tam üzerine yazma YOK — devam hücresinden en büyük fark budur). Merge mantığı
# `tcl.merge_composite_manifest`de (torch-free, birim testli).
# ==========================================================================
def stage_drive_copy_o() -> None:
    report("drive_copy", "running")
    src = Path("/content/birefnet_format_o")
    dst = Path(DRIVE_ROOT) / DRIVE_OUTPUT_SUBDIR
    dst_train_im = dst / "TRAIN" / "im"
    dst_train_gt = dst / "TRAIN" / "gt"
    assert dst_train_im.is_dir() and dst_train_gt.is_dir(), (
        f"Drive'da beklenen v1/v2 TRAIN verisi bulunamadı: {dst_train_im} / {dst_train_gt} — "
        f"bu hücre yalnız MEVCUT bir veri setine _o00 EKLEMEK içindir, sıfırdan veri seti "
        f"oluşturmak için colab_devam_hucresi.py kullanılmalı."
    )

    src_im_files = list((src / "TRAIN" / "im").iterdir())
    src_gt_files = list((src / "TRAIN" / "gt").iterdir())
    existing_dst_im_stems = {p.stem for p in dst_train_im.iterdir()}
    new_stems = {p.stem for p in src_im_files} - existing_dst_im_stems
    expected_growth = len(new_stems)

    pre_im, pre_gt = len(list(dst_train_im.iterdir())), len(list(dst_train_gt.iterdir()))
    print(f"Merge öncesi Drive TRAIN: im={pre_im}, gt={pre_gt} — beklenen artış: {expected_growth}")

    # YALNIZ TRAIN/ alt ağacı kopyalanır — src kökündeki stats.json BİLİNÇLİ
    # OLARAK KOPYALANMAZ: export_birefnet.export() onu yalnız _o00 setinin
    # KISMİ istatistikleriyle yazdı; Drive'daki stats.json ise v1/v2'nin TAM
    # veri setinin otoriter istatistikleri — tüm src kökünü copytree'lemek
    # onu sessizce EZERDİ (reviewer bulgusu #1).
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
    assert len(src_im_files) == len(src_gt_files), "yerel _o00 export'unda im/gt sayıları uyuşmuyor!"

    comp_manifest_local = Path("data/train_composites_o/manifest.jsonl")
    comp_manifest_drive = dst / "train_composites_manifest.jsonl"
    n_appended = tcl.merge_composite_manifest(comp_manifest_local, comp_manifest_drive)
    print(f"train_composites_manifest.jsonl: {n_appended} yeni satır eklendi (Drive'daki mevcut "
          f"v1/v2 satırları KORUNDU, üzerine yazılmadı).")
    assert n_appended == expected_growth, (
        f"manifest ekleme sayısı ({n_appended}) dosya büyümesiyle ({expected_growth}) tutarsız — "
        f"stem/id eşlemesi kontrol edilmeli."
    )

    print("\nBÜTÜNLÜK KONTROLÜ BAŞARILI — _o00 verisi Drive'a MERGE edildi.")
    report(
        "drive_copy", "done",
        added_files=expected_growth, added_manifest_rows=n_appended,
        total_im=post_im, total_gt=post_gt,
    )


# ==========================================================================
# Orkestrasyon — üst düzeyde koşar (hücre yapıştırılıp çalıştırıldığında).
# ==========================================================================
def main() -> None:
    stage0_env_sanity()
    stage1_bg_pool()
    stage_manifest()
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
