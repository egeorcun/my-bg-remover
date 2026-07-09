"""GT'li eğitim kaynaklarından kategorili `data/train/manifest.jsonl` üret.

Kullanım:
    uv run python scripts/build_trainset.py                # tüm SOURCES + DIS5K-TR
    uv run python scripts/build_trainset.py source camotr   # tek kaynağı ekle

Test setinden (build_testset.py) farkı: dosyalar KOPYALANMAZ, `data/train/{images,gt}/`
altına orijinal ham dosyaya işaret eden SEMBOLİK BAĞLANTI (symlink) oluşturulur — disk
bütçesi (bkz. Faz 2 planı Global Constraints) kopyalamaya izin vermez. Format dönüşümü
(örn. GT'yi tek kanallı L PNG'ye normalize etmek) bu yüzden symlink ANINDA değil, ham
veri edinimi (fetch) aşamasında yapılır; bu script yalnız zaten normalize edilmiş ham
dosyaları glob'lar ve bağlar.

Ham veri edinimi (data/raw_train/ altına, git dışı; DISK BÜTÇESİ: kaynak başına ≤300MB,
bkz. Faz 2 planı REVİZE disk kuralı — tam materyalizasyon Colab'da, bkz. data/train_sources.json):

- DIS5K-TR (kategori: dosya adı token'ından thin/complex, bkz. classify_disvd): HF
  dataset "nobg/DIS5K", data/DIS_TR-00000-of-00006-*.parquet (6 parçadan yalnız ilki;
  tam DIS_TR ~3000 çift). Parça bile ~480MB olduğundan TÜM PARQUET İNDİRİLMEDİ: pyarrow
  ParquetFile + huggingface_hub.HfFileSystem (fsspec, HTTP range request) ile YALNIZ
  0. row-group'u (100 satır, ~120MB) kısmi okundu, image/label byte'ları PIL ile
  data/raw_train/dis5k/{im,gt}/ altına dosya olarak yazıldı (100 çift; GT convert("L")
  ile tek kanala normalize edildi). Tam DIS_TR indirimi Colab'da (T5 notebook)
  data/train_sources.json'daki hf_repo üzerinden yapılacak.
- CAMO-TR (kategori: camouflage): HF dataset "nobg/camo" (resmi CAMO, Le et al. 2019;
  proje sayfası https://sites.github.com/view/ltnghia/research/camo; lisans
  CC-BY-NC-SA 4.0; train split 1000 çift = 3 row-group: 423+423+154). Yalnız 0.
  row-group'un YALNIZ image_name/image/mask kolonları (overlaid_mask_1/2 HARİÇ —
  column pruning ile ekstra indirme önlendi) kısmi okundu; ilk 100 satır
  data/raw_train/camo/{im,gt}/ altına yazıldı (~18MB toplam).
- COD10K-TR (kategori: camouflage): HF'de piksel-seviyeli GT maskeli TRAIN mirror'ı
  YOK (bkz. scripts/build_testset.py docstring — Chranos/COD10K_train ve Jrseee/COD10K
  boş/LFS işaretçisiz repo; chandrabhuma/animal_cod10k(_train) yalnız görsel+soru-cevap,
  piksel maskesi yok; bu görev sırasında ayrıca aranan SmallDoge/CoD-10K alakasız bir
  metin veri seti — "CoD" isim çakışması, "Chain of Draft" tipi bir kod/metin korpüsü).
  Resmi kaynak yalnız Google Drive üzerinden (SINet/DengPingFan reposu,
  https://github.com/DengPingFan/SINet): COD10K-train dosya id
  "1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt" (~3040 çift). Bu ortamda gdown/Drive kimlik
  doğrulaması olmadığından LOKAL ÖRNEKLEM ATLANDI (görev talimatı: "yoksa ATLA");
  kayıt data/train_sources.json'da (drive_id + resmi URL) mevcut — Colab'da (T5)
  gdown ile tam indirilecek.
- P3M-10k TRAIN (kategori: hair): HF dataset "Rupant-ted/p3m-10k" TEK bir zip
  (data/p3m10k.zip, ~5.8GB) olarak barındırılıyor; `P3M-10k/train/{blurred_image,mask}/`
  altında 9422 çift var (list_repo_files ile önce zip içeriği, sonra
  huggingface_hub.HfFileSystem (fsspec) ile zip'i AÇMADAN zipfile.ZipFile üzerinden
  merkezi dizin (central directory) okunarak doğrulandı). Zip TAMAMEN İNDİRİLMEDİ:
  merkezi dizin HTTP range request ile kısmi okundu, ardından 100 rastgele çiftin
  yalnız kendi sıkıştırılmış byte aralıkları (yine range request ile, 12 thread'de
  paralel) çekildi (~50MB toplam) ve data/raw_train/p3m/{im,gt}/ altına PIL ile (GT
  convert("L")) yazıldı.
- Transparent-460 TRAIN (kategori: transparent): HF dataset "Thinnaphat/transparent-460"
  `Train/{fg,alpha}/` altında 410 çift (Faz 0'da yalnız `Test/` 50 çift kullanılmıştı).
  Orijinal dosyalar çok büyük (ortalama ~4.2MB, bazı alpha PNG'leri 40-80MB) — disk
  bütçesini (≤300MB/kaynak) aşmamak için: `repo_info(files_metadata=True)` ile
  bildirilen boyuta göre en küçük 300/410 çiftlik havuzdan 80 çift rastgele seçildi,
  HfFileSystem ile belleğe akış (hf_hub_download önbelleği KULLANILMADI — disk
  tasarrufu) okunup PIL ile uzun kenarı 1280px'e küçültülüp (fg: JPEG q90, alpha: PNG,
  ikisi de AYNI boyuta) data/raw_train/trans460_train/{fg,alpha}/ altına yazıldı
  (toplam ~22MB). Tam TRAIN seti (orijinal çözünürlükte) Colab'da indirilecek.
"""
import argparse
import random
import shutil
from pathlib import Path

from build_testset import _sanitize, classify_disvd  # noqa: E402  (aynı dizin, scripts/)

from benchmark.testset import append_entries

random.seed(42)
ROOT = Path(__file__).resolve().parent.parent
OUT_IMG = ROOT / "data/train/images"
OUT_GT = ROOT / "data/train/gt"
MANIFEST = ROOT / "data/train/manifest.jsonl"


def _link(src: Path, dst: Path, copy: bool = False) -> None:
    """dst -> src sembolik bağlantısı oluşturur (varsayılan; kopya YOK, disk tasarrufu).

    copy=True ise gerçek dosya KOPYALANIR — Colab'da tam veri materyalizasyonu için
    (bkz. training/prepare_data_colab.ipynb): sembolik bağlantı Drive'a taşıma/zip
    sırasında hedefi (Colab'ın geçici /content diskini) kaybedip kırılabilir; kopya
    böyle bir kırılganlık taşımaz.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src.resolve(), dst)
    else:
        dst.symlink_to(src.resolve())


def sample_source(name: str, img_glob: str, gt_glob: str, category: str,
                   n: int | None = None, copy: bool = False) -> list[dict]:
    """Bir kaynaktan (img_glob/gt_glob stem'e göre eşleştirilir) örnekle ve
    `data/train/{images,gt}/` altına symlink'le (copy=True -> gerçek dosya kopyala).
    n=None -> tüm eşleşen çiftler."""
    imgs = sorted(ROOT.glob(img_glob))
    gts = {p.stem: p for p in ROOT.glob(gt_glob)}
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
    """DIS5K havuzundan örnekle; kategori dosya adı token'ından (classify_disvd,
    build_testset.py'den yeniden kullanılır) atanır — rastgele dağıtım YOKTUR."""
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


# TEK DOĞRULUK KAYNAĞI: kaynak adı -> glob desenleri + kategori kuralı. Örneklem
# boyutu BİLİNÇLİ OLARAK burada DEĞİL (LOCAL_SAMPLE_N'de ayrı) — Colab notebook'u
# (training/prepare_data_colab.ipynb) aynı tanımları n=None (tam set) ile kullanır;
# glob/kategori bilgisinin notebook'a elle kopyalanıp zamanla sapması (drift) böyle
# önlenir. category "disvd_tokens" -> kategori dosya adı token'ından atanır
# (classify_disvd, sample_disvd_tokens ile işlenir); diğerleri sabit kategori
# (sample_source ile işlenir).
#
# NOT (matting setleri araştırması, bkz. data/train_sources.json + Faz 2 T3 raporu):
# Distinctions-646 (Qiao et al. CVPR2020, HAttMatting) yalnız e-posta ile talep üzerine
# dağıtılıyor — HİÇBİR HF/genel indirme linki yok, atlandı. HIM2K (Sun et al. CVPR2022,
# InstMatt) ve AM-2k (Li et al. IJCV2022, GFM) yalnız Google Drive/Baidu Wangpan
# üzerinden (AM-2k için ayrıca MIT lisanslı resmi "Dataset Release Agreement" imzası
# gerekiyor) dağıtılıyor; bu ortamda gdown/Drive kimlik doğrulaması yok (COD10K-TR'deki
# aynı kısıt) — LOKAL ÖRNEKLEM ATLANDI, kayıtlar data/train_sources.json'da (drive_id +
# resmi URL + lisans notu) mevcut, Colab'da (T5) indirilecek.
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

# Lokal doğrulama örneklemi boyutları (disk bütçesi, bkz. modül docstring'i) — yalnız
# lokal koşuda anlamlı; Colab tam setle (n=None) çalışır.
LOCAL_SAMPLE_N: dict[str, int] = {"camotr": 100, "p3m": 100, "trans460tr": 80, "dis5ktr": 100}

# Geriye dönük uyumlu görünüm: (kaynak_ad, images_glob, gt_glob, kategori, adet) —
# SOURCE_SPECS'ten türetilir (disvd_tokens hariç; o sample_disvd_tokens ile işlenir).
SOURCES: list[tuple[str, str, str, str, int]] = [
    (name, spec["img_glob"], spec["gt_glob"], spec["category"], LOCAL_SAMPLE_N[name])
    for name, spec in SOURCE_SPECS.items()
    if spec["category"] != "disvd_tokens"
]

# DIS5K-TR tek havuzdan örneklenir; kategori dosya adı token'ından atanır.
DIS5KTR_IMG_GLOB = SOURCE_SPECS["dis5ktr"]["img_glob"]
DIS5KTR_GT_GLOB = SOURCE_SPECS["dis5ktr"]["gt_glob"]
DIS5KTR_N = LOCAL_SAMPLE_N["dis5ktr"]


def build(copy: bool = False) -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        rows = sample_source(*src, copy=copy)
        append_entries(str(MANIFEST), rows)
        print(f"{src[0]} ({src[3]}): {len(rows)} örnek")

    rows = sample_disvd_tokens("dis5ktr", DIS5KTR_IMG_GLOB, DIS5KTR_GT_GLOB, DIS5KTR_N, copy=copy)
    append_entries(str(MANIFEST), rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for category, count in sorted(counts.items()):
        print(f"dis5ktr ({category}): {count} örnek")


def add_source(name: str, copy: bool = False) -> None:
    """SOURCES'taki tek bir kaynağı örnekle ve manifest'e ekle (artımlı ekleme)."""
    matches = [s for s in SOURCES if s[0] == name]
    if not matches:
        raise SystemExit(f"bilinmeyen kaynak: {name} (SOURCES: {[s[0] for s in SOURCES]})")
    rows = sample_source(*matches[0], copy=copy)
    append_entries(str(MANIFEST), rows)
    print(f"{name} ({matches[0][3]}): {len(rows)} örnek eklendi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=["source"], default=None,
                         help="belirtilirse tek bir kaynağı ekler (bkz. --name)")
    parser.add_argument("name", nargs="?", default=None, help="'source' komutu için kaynak adı")
    parser.add_argument("--copy", action="store_true",
                         help="symlink yerine gerçek dosya kopyala (Colab'da tam veri "
                              "materyalizasyonu için; lokalde varsayılan symlink kalır)")
    args = parser.parse_args()

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_GT.mkdir(parents=True, exist_ok=True)
    if args.command == "source":
        if not args.name:
            raise SystemExit("kullanım: build_trainset.py source <ad> [--copy]")
        add_source(args.name, copy=args.copy)
    else:
        build(copy=args.copy)


if __name__ == "__main__":
    main()
