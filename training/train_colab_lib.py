"""Faz 3 (`training/train_colab.ipynb`) için saf-Python, bağımlılıksız yardımcı
mantık. Bilinçli olarak `torch`/`PIL` içe aktarmaz — böylece hem Colab'da (BiRefNet
eğitim döngüsü içinde) hem de bu depoda GPU/torch olmadan `pytest` ile test
edilebilir (bkz. `tests/test_train_colab_lib.py`). Notebook bu dosyayı repo'yu
klonladıktan sonra `sys.path`'e ekleyip import eder (`scripts/` için kullanılan
"TEK DOĞRULUK KAYNAĞI" deseniyle aynı — bkz. `training/prepare_data_colab.ipynb`
hücre (d) notu) — mantık notebook içine kopyalanıp tekrar yazılmaz, drift riski
böylece ortadan kalkar.

Altı bağımsız endişe kapsar:
1. Kategori ağırlıklı örnekleme (`compute_sample_weights` / `compute_expected_shares`)
   — `torch.utils.data.WeightedRandomSampler`'a beslenecek ağırlıkları hesaplar.
2. Checkpoint keşfi/budama (`find_latest_checkpoint` / `prune_old_checkpoints`)
   — Colab oturum kopması sonrası otomatik devam + Drive disk kotasını sınırlama.
3. Deterministik + KALICI TRAIN/VAL bölünmesi (`deterministic_val_split` /
   `load_or_create_val_split`) + sabit hızlı-değerlendirme alt kümesi
   (`fixed_eval_subset`).
4. BiRefNet resmi `train.py`/`config.py` mantığının iki küçük parçasının yeniden
   üretimi (`should_apply_finetune_reweight`, `effective_lr`) — bkz. modül
   içindeki fonksiyon docstring'lerinde satır bazlı referanslar.
5. BiRefNet `config.py` metin yaması (`apply_config_patches`) — İDEMPOTENT
   (aynı VM'de notebook'un yeniden koşturulması patlamamalı).
6. Drive→yerel disk veri kopyalama (`copy_pairs`) — hem im hem gt dosya boyutu
   doğrulamalı (yarım kalmış/kesilmiş kopyalar onarılır).
7. v3 — VAL sızıntı hariç tutma + kompozit manifest merge + boş-manifest
   koruması (`strip_composite_copy_suffix` / `derive_val_excluded_source_ids` /
   `merge_composite_manifest` / `ensure_manifest_pairs`) — `training/
   v3_veri_guncelleme_hucresi.py`'nin `_o00` üretimi öncesi VAL kümesini hariç
   tutması, Drive'daki kompozit manifest'i (üzerine yazmadan) güncellemesi ve
   boş bir manifest'le export'a geçmeyi erken/yüksek sesle engellemesi için
   (bkz. o dosyanın modül docstring'i).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from benchmark.testset import append_entries, load_manifest

CKPT_FILENAME_RE = re.compile(r"^epoch_(\d+)\.pth$")


# ============================================================================
# 1) Kategori ağırlıklı örnekleme
# ============================================================================
def load_stem_categories(manifest_path: str | Path) -> dict[str, str]:
    """Kompozit manifest'ini (`benchmark.testset` formatı: id/image/category/gt_alpha
    JSONL satırları — bkz. `scripts/export_birefnet.py` docstring'i, export sırasında
    stem = row['id']) okuyup `{stem: category}` sözlüğü döndürür.

    Notebook bu dosyayı `Drive'daki bg-remover-data/train_composites_manifest.jsonl`
    kopyasından okur (bkz. `training/colab_devam_hucresi.py` `stage7_drive_copy`)."""
    result: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row["id"]] = row["category"]
    return result


SAMPLER_PRESET_V1: dict[str, float] = {"transparent": 0.20, "camouflage": 0.20}
"""v1 fine-tune koşusunda (`epoch_1.pth`) fiilen kullanılan hedef —
`docs/reports/2026-07-faz2-veri.md` §5 madde 3. Yalnız transparent+camouflage'ı
sabitlediği için, geri kalan %60'lık pay hair/complex/thin/general arasında HAM
sayılarıyla orantılı bölüşülüyordu; hair'in ham hacmi (~9422) complex (~2190) ve
thin'in (~810) çok üzerinde olduğundan bu %60'ın büyük kısmını hair alıyor,
complex/thin'e neredeyse hiç pay kalmıyordu — v1 karşılaştırma raporundaki
"catastrophic forgetting"in (complex MAE 0.156 vs 0.024 baseline, thin 0.090 vs
0.018, hair 0.013 vs 0.0045) kök nedeni."""

SAMPLER_PRESET_V2: dict[str, float] = {
    "camouflage": 0.18,
    "transparent": 0.20,
    "hair": 0.20,
    "complex": 0.20,
    "thin": 0.12,
    "general": 0.10,
}
"""v2 rebalancing hedefi (toplam TAM %100 — `compute_sample_weights` yalnız
`sum > 1.0`'da ValueError fırlatır; toplam tam 1.0 iken manifest'te kategorisi
bulunamayan `_other` stem'lere SIFIR ağırlık düşer, yani hiç örneklenmezler —
bilinçli tercih: kategorisi bilinmeyen veri eğitim karışımını bulandırmasın;
notebook hücre (e) zaten `n_unknown` sayısını konsola yazdırıyor). camouflage
v1'deki %20'den %18'e hafifçe DÜŞÜRÜLDÜ (ham payı zaten ~%28-36 — v1'de sampler
dışı bırakılsa bile doğal olarak büyük pay alırdı; v1 kazanımını korumaya %18
yeter). transparent %20'de TUTULDU: ideogram skorlaması somutlaştırdı —
transparent, bgr-v1'in ideogram'a KAYBETTİĞİ tek kategori (MAE 0.0437 vs
0.0343, en yakın kovalama hedefi), payını kısmak yanlış olurdu. hair/complex/
thin'e AÇIKÇA hedef verildi (v1'de hedefsizdi) — hair %20 (mutlak hatası zaten
küçük, 0.013 MAE — toparlama hedefi mütevazı), complex %20, thin %12 — v1'de
çöken kategorileri toparlamak için; general %10 kürasyonlu genel-amaçlı
görseller. Bkz. `.superpowers/sdd/v2-hazirlik-report.md`."""

SAMPLER_PRESET_V3: dict[str, float] = {
    "camouflage": 0.16,
    "transparent": 0.24,
    "hair": 0.18,
    "complex": 0.20,
    "thin": 0.12,
    "general": 0.10,
}
"""v3 rebalancing hedefi (toplam TAM %100) — v2'nin gerçek benchmark sonuçlarından
sonraki ayar (bkz. `results/baseline/metrics.json`, `.superpowers/sdd/
v3-hazirlik-report.md`). İki somut bulguya cevap verir:

1. **Domain gap / over-deletion kalıcılığı**: gerçek-fotoğraf benchmark'ında
   over-deletion'ın v1→v2 arası düzelmemesinin kök nedeni, camouflage DIŞINDAKİ
   TÜM kategorilerin yalnız SENTETİK kompozit arka planlarda eğitilmesiydi —
   sampler payını değiştirmek bunu çözmez, veriye orijinal arka plan örnekleri
   (`scripts/make_composites.py` `_o00` kopyaları — bkz. o dosyanın v3 notu)
   eklemek gerekiyordu. Sampler tarafında yapılabilecek TEK şey, bu yeni verinin
   epoch içinde yeterince görülmesini sağlamak.
2. **transparent v1→v2 arası KÖTÜLEŞTİ** (MAE 0.0437→0.0481) — ideogram'ın
   0.0343'lük hedefinden UZAKLAŞTIK (v2'nin %18'e düşürdüğü pay yanlış yönde
   hareket etmiş olabilir). v3 transparent'ı %24'e (v2'nin %18'inden +6 puan)
   YÜKSELTEREK ideogram'ı kovalamayı önceliklendiriyor — bu preset'in en büyük
   tek payı.
   camouflage v2'de zaten güçlü marj bırakıyor (bgr-v2 MAE 0.0310, en yakın
   genel-amaçlı rakip birefnet-hr 0.0752 — %59 daha iyi, ideogram ise camo'da
   ÇOK daha kötü: 0.1179) — bu marj sayesinde camo payı v2'nin %18'inden %16'ya
   biraz daha düşürülüp kazanılan 2 puan transparent'a aktarılabildi. hair
   %20'den %18'e (mutlak hatası zaten küçük, 0.0156 MAE), complex/thin/general
   v2'deki değerlerinde (%20/%12/%10) KORUNDU (v1'in çöken kategorileri — bkz.
   SAMPLER_PRESET_V2 docstring'i — toparlanmaya devam ediyor, henüz payı
   azaltacak kanıt yok). Bkz. `.superpowers/sdd/v3-hazirlik-report.md`."""

SAMPLER_PRESET_V4: dict[str, float] = {
    "camouflage": 0.12,
    "transparent": 0.18,
    "hair": 0.08,
    "complex": 0.19,
    "thin": 0.13,
    "general": 0.04,
    "text": 0.10,
    "fx": 0.08,
    "illustration": 0.08,
}
"""v4 rebalancing hedefi (toplam TAM %100) — v3'ün gerçek benchmark sonuçlarından
sonraki ayar. Kullanıcı v3 benchmark'ı sonrası odağı iki eksene kaydırdı:
complex+thin'in toparlanmaya devam etmesi ve YENİ yeteneklerin kazanılması —
logo/yazı koruma (`text`), obje etrafı VFX parıltı (`fx`) ve illüstrasyon
(`illustration`); üç yeni kategorinin verisini `training/
v4_veri_guncelleme_hucresi.py` üretir (`scripts/make_textfx.py` + ToonOut).

1. **transparent %18'de TUTULDU**: Ideogram'ın 0.0343'lük hedefine yalnız
   0.0043 kaldı — kovalamaca sürüyor, payı kısmak v2 dersinin (payı düşürünce
   MAE kötüleşti, bkz. SAMPLER_PRESET_V3 docstring'i madde 2) tekrarı olurdu;
   ama v3'teki %24'lük tek-en-büyük-pay da artık gerekmiyor, %18 koruma için
   yeterli.
2. **camouflage %12'ye DÜŞTÜ**: v3 MAE 0.0304 vs Ideogram 0.1179 — marj
   DEVASA (Ideogram'ın yaklaşık dörtte biri). v2→v3'te %18→%16'ya inen pay,
   bu marj sayesinde %12'ye kadar güvenle indirilebildi; serbest kalan puanlar
   yeni kategorilere aktarıldı.
3. **hair %8'e DÜŞTÜ**: 0.0067 MAE ile rmbg'nin 0.0045'ine zaten yakın —
   pay azaltılabilir (v3'te %18'di; mutlak hata küçük, koruma için %8 yeter).
4. **complex %19 / thin %13**: v3'teki %20/%12'ye yakın tutuldu (odak
   kategoriler — v1'in çöken kategorileri hâlâ öncelikli, thin +1 puanla
   hafifçe güçlendirildi). general %10'dan %4'e indi (kürasyonlu genel-amaçlı
   görseller; yeni kategorilere yer açmak için en az riskli kesinti).
5. **text %10 / fx %8 / illustration %8**: yeni yetenekler — toplam %26'lık
   pay, modelin bu üç beceriyi sıfırdan öğrenmesine yetecek epoch-içi
   görünürlük sağlar."""

SAMPLER_PRESETS: dict[str, dict[str, float]] = {
    "v1": SAMPLER_PRESET_V1,
    "v2": SAMPLER_PRESET_V2,
    "v3": SAMPLER_PRESET_V3,
    "v4": SAMPLER_PRESET_V4,
}
"""Notebook `SAMPLER_PRESET` parametresinin ("v1"/"v2"/"v3"/"v4") çözümlendiği
tablo — bkz. `training/train_colab.ipynb` parametre hücresi ve hücre (e)."""


def resolve_sampler_num_samples(dataset_len: int, num_samples: int | None = None) -> int:
    """`WeightedRandomSampler(weights, num_samples=...)`'a verilecek değeri çözer
    (torch'a bağımlı olmadan test edilebilmesi için sampler NESNESİ değil, yalnız
    bu SAYIYI hesaplayan saf fonksiyon — bkz. modül başı docstring "torch/PIL
    içe aktarmaz" ilkesi).

    `num_samples=None` (varsayılan): v1/v2 davranışıyla BİREBİR aynı —
    `dataset_len` (o anki `len(train_dataset)`) döner, yani epoch uzunluğu veri
    setiyle birlikte büyür/küçülür.

    `num_samples` verilirse (v3): epoch uzunluğu veri setinin gerçek boyutundan
    BAĞIMSIZ, SABİT bu değere kilitlenir. v3'te veri setine `scripts/
    make_composites.py`'nin `_o00` kopyalarıyla ~14k yeni çift eklendiğinde
    (bkz. o dosyanın v3 notu), `num_samples=None` bırakılsaydı epoch başına
    iterasyon sayısı (ve dolayısıyla Colab birim maliyeti) da otomatik büyürdü;
    notebook bunun yerine `EPOCH_NUM_SAMPLES=27715` (v2'nin epoch büyüklüğüyle
    PARİTE) geçer — epoch maliyeti ~48 birimde sabit kalır. `WeightedRandomSampler`
    zaten `replacement=True` ile çalıştığından `num_samples < dataset_len` veri
    KAYBI değildir — yalnızca epoch'un ne kadar örnek çektiğini kısaltır, yeni
    eklenen `_o00` örnekleri sampler ağırlıkları üzerinden (kategori paylarına
    göre) yine olasılıksal olarak seçilebilir kalır.

    `num_samples <= 0` -> `ValueError` (WeightedRandomSampler'ın kendisi de bunu
    reddeder, ama net bir mesajla erken yakalanır)."""
    if num_samples is None:
        return dataset_len
    if num_samples <= 0:
        raise ValueError(f"num_samples > 0 olmalı: {num_samples}")
    return num_samples


def compute_sample_weights(
    stems: list[str],
    stem_category: dict[str, str],
    target_share: dict[str, float] | None = None,
    default_category: str = "_other",
) -> list[float]:
    """`stems` (MyData.image_paths ile AYNI SIRADA olmalı — WeightedRandomSampler
    ağırlıkları dataset indeksleriyle hizalanmak zorunda) için, `target_share`'de
    adı geçen kategorilerin epoch-içi BEKLENEN payını `target_share`'e sabitleyen,
    geri kalan kategorilerin KENDİ ARALARINDAKİ göreli oranını (ham sayılarıyla
    orantılı) koruyan ağırlıklar üretir.

    `target_share=None` (varsayılan) ise `SAMPLER_PRESET_V1` kullanılır — v1
    fine-tune koşusunun (epoch_1.pth) davranışıyla BİREBİR aynı (geriye dönük
    uyumluluk: mevcut çağıranlar hiçbir şey değiştirmeden aynı sonucu almaya
    devam eder). v2 rebalancing için `SAMPLER_PRESET_V2` (veya
    `SAMPLER_PRESETS["v2"]`) açıkça geçilmeli — bkz. modül başı `SAMPLER_PRESETS`
    ve v2-hazırlik raporu (v1'de transparent+camouflage payı birleşik >%50'ye
    çıkıp complex/thin/hair'de "catastrophic forgetting"e yol açmıştı).

    Algoritma: hedefi olan bir kategori c için örnek başına ağırlık =
    target_share[c] / count(c) (kategori toplamda tam olarak target_share[c]
    payını alır, kategori-içi örnekler eşit ağırlıklı). Hedefsiz kategoriler için
    örnek başına ağırlık = (1 - sum(target_share)) / toplam_hedefsiz_sayı — TÜM
    hedefsiz örnekler için AYNI sabit değer, bu da onların birbirlerine göre
    payının (ağırlıksız haldeki gibi) ham sayılarıyla orantılı kalmasını sağlar.

    Bu, fiziksel oversampling'e (kompozit dosyalarını fazladan üretmek) göre EN AZ
    MÜDAHALECİ mekanizma: `scripts/make_composites.py`'nin `CATEGORY_MULTIPLIER`
    çarpanlarına (transparent×10, camouflage×2 — bkz. o dosyanın docstring'i)
    dokunmadan, yalnız `DataLoader`'ın sampler'ını değiştirerek çalışır; resmi
    `train.py`nin `prepare_dataloader`'ı (`shuffle=is_train, sampler=None`)
    üzerine YALNIZ bir `sampler=` argümanı eklenir (bkz. notebook eğitim hücresi).
    """
    if target_share is None:
        target_share = SAMPLER_PRESET_V1
    categories = [stem_category.get(s, default_category) for s in stems]
    counts = Counter(categories)

    targeted = {c: share for c, share in target_share.items() if counts.get(c, 0) > 0}
    sum_targeted = sum(targeted.values())
    if sum_targeted > 1.0 + 1e-9:  # tam 1.0'a İZİN VAR (bkz. SAMPLER_PRESET_V2); epsilon fp toplama gürültüsü için
        raise ValueError(f"target_share toplamı > 1.0 olamaz (mevcut kategorilerde): {targeted}")
    remaining_mass = max(0.0, 1.0 - sum_targeted)  # sum==1.0 -> hedefsiz (_other) örneklere 0 ağırlık (hiç örneklenmezler)

    other_categories = [c for c in counts if c not in targeted]
    n_other_total = sum(counts[c] for c in other_categories)

    per_category_weight: dict[str, float] = {}
    for c, share in targeted.items():
        per_category_weight[c] = share / counts[c]
    other_weight = (remaining_mass / n_other_total) if n_other_total > 0 else 0.0
    for c in other_categories:
        per_category_weight[c] = other_weight

    return [per_category_weight[c] for c in categories]


def compute_expected_shares(
    weights: list[float], stems: list[str], stem_category: dict[str, str], default_category: str = "_other"
) -> dict[str, float]:
    """Tanılama: verilen ağırlıklarla (normalize edilmemiş de olsa) her kategorinin
    epoch-içi BEKLENEN örnekleme payını hesaplar (`sum(weights in cat) / sum(all weights)`).
    Notebook bu fonksiyonu, sampler kurulduktan hemen sonra hedefin (≥%20) gerçekten
    tutturulduğunu konsola yazdırmak için çağırır."""
    total = sum(weights)
    if total <= 0:
        return {}
    sums: dict[str, float] = {}
    for w, s in zip(weights, stems):
        c = stem_category.get(s, default_category)
        sums[c] = sums.get(c, 0.0) + w
    return {c: v / total for c, v in sums.items()}


# ============================================================================
# 2) Checkpoint keşfi / budama (resume + Drive disk kotası)
# ============================================================================
def find_latest_checkpoint(ckpt_dir: str | Path, pattern: re.Pattern = CKPT_FILENAME_RE) -> tuple[str, int] | None:
    """`ckpt_dir` altında `epoch_<N>.pth` desenine uyan dosyaları tarar, en büyük
    N'e sahip olanı `(yol, epoch)` olarak döndürür; hiç yoksa `None` (ilk koşu —
    `BiRefNet.from_pretrained(HF_MODEL_ID)` ile sıfırdan başlanır)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best: tuple[str, int] | None = None
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if best is None or epoch > best[1]:
            best = (str(p), epoch)
    return best


def prune_old_checkpoints(
    ckpt_dir: str | Path, keep_last_n: int, pattern: re.Pattern = CKPT_FILENAME_RE
) -> list[str]:
    """`ckpt_dir`'de yalnız en son `keep_last_n` epoch'un checkpoint'ini bırakır,
    gerisini SİLER; silinen dosya yollarını döndürür. Hem lokal Colab diskinde hem
    de Drive'da çağrılır (100 epoch × ~güncel BiRefNet checkpoint boyutu Drive
    kotasını hızla doldurur — bkz. notebook parametre hücresi `KEEP_LAST_N_CHECKPOINTS`)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    entries: list[tuple[int, Path]] = []
    for p in ckpt_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0], reverse=True)
    removed = []
    for _, p in entries[keep_last_n:]:
        p.unlink()
        removed.append(str(p))
    return removed


# ============================================================================
# 3) Deterministik TRAIN/VAL bölünmesi + sabit hızlı-değerlendirme alt kümesi
# ============================================================================
def deterministic_val_split(all_stems: list[str], seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    """`all_stems`'i (girdi sırası ÖNEMSİZ — önce sıralanır, sonra tohumlu
    karıştırılır, böylece dosya sistemi listeleme sırasından bağımsız aynı sonuç
    üretir) deterministik olarak (train_stems, val_stems) ikilisine böler.
    Yeniden koşularda (idempotentlik, görev madde 6) AYNI val kümesini üretir —
    fiziksel taşıma YOK, yalnız notebook bu listeye göre hangi dosyaları
    TRAIN/ vs VAL/ alt dizinine kopyalayacağına karar verir."""
    import random

    stems_sorted = sorted(all_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def load_or_create_val_split(
    all_stems: list[str], seed: int, val_fraction: float, persist_path: str | Path
) -> tuple[list[str], list[str]]:
    """`deterministic_val_split`'in KALICI hali: ilk koşuda bölünmeyi yapıp
    val listesini `persist_path`'e (JSON) yazar; sonraki koşularda dosyadan okur.

    Neden gerekli: Drive'daki veri seti sonradan BÜYÜYEBİLİR (Faz 2 pipeline'ı
    idempotent — yeni bir koşu yeni çiftler ekleyebilir). Salt-deterministik
    bölünme, girdi listesi değişince FARKLI bir val kümesi üretirdi — önceki
    epoch'larda eğitimde görülmüş görseller val'e sızardı. Kalıcı listeyle val
    kümesi İLK koşuda ne seçildiyse o kalır; SONRADAN eklenen stem'lerin TAMAMI
    TRAIN'e gider (bilinçli, basit tercih: val kümesinin epoch'lar arası
    karşılaştırılabilirliği, oransal val büyütmekten daha değerli — val payı
    zamanla %2'nin biraz altına düşebilir, hızlı-değerlendirme zaten sabit
    `n=24`'lük bir alt küme kullandığı için pratik etkisi yok).

    Dosyada kayıtlı olup artık diskte OLMAYAN stem'ler sessizce val'den düşürülür
    (veri silinmişse bölünme yine tutarlı kalır)."""
    persist_path = Path(persist_path)
    if persist_path.exists():
        saved = json.loads(persist_path.read_text())
        saved_val = set(saved["val_stems"])
        all_set = set(all_stems)
        val = sorted(saved_val & all_set)
        train = sorted(all_set - saved_val)
        return train, val

    train, val = deterministic_val_split(all_stems, seed=seed, val_fraction=val_fraction)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    persist_path.write_text(
        json.dumps({"seed": seed, "val_fraction": val_fraction, "val_stems": val}, ensure_ascii=False, indent=1)
    )
    return train, val


def fixed_eval_subset(val_stems: list[str], seed: int, n: int) -> list[str]:
    """VAL kümesinden (2% — yüzlerce görsel olabilir) HER epoch aynı, sabit
    `n` (varsayılan 24) görsellik bir alt küme seçer — periyodik hızlı-değerlendirmenin
    epoch'lar arasında karşılaştırılabilir olması için (her seferinde farklı
    rastgele görsellerle ölçülen MAE'nin gürültüsü epoch-to-epoch trendini
    gizleyebilir)."""
    import random

    stems_sorted = sorted(val_stems)
    rng = random.Random(seed)
    shuffled = stems_sorted[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[: min(n, len(shuffled))])


# ============================================================================
# 4) BiRefNet resmi train.py/config.py mantığının küçük parçaları
# ============================================================================
def should_apply_finetune_reweight(epoch: int, total_epochs: int, finetune_last_epochs: int) -> bool:
    """BiRefNet resmi `train.py::Trainer.train_epoch` içindeki koşul:
    `if epoch > args.epochs + config.finetune_last_epochs:` (kaynak:
    ZhengPeng7/BiRefNet `train.py`, GitHub `main` dalı, fonksiyon `train_epoch`,
    ~satır 195 civarı — bu depoya `curl` ile çekilip incelendi, bkz. Faz 3 raporu).
    `finetune_last_epochs` NEGATİF bir sayıdır (`config.py`'de `Matting` görevi
    için `-10` — son 10 epoch'ta pixel loss ağırlıkları kademeli değiştirilir,
    "belgelenen fine-tune hilesi"). `total_epochs`, o Colab OTURUMUNUN DEĞİL,
    eğitimin NİHAİ HEDEF epoch sayısıdır (`EPOCHS` parametresi — resume'lerde
    HER OTURUMDA AYNI değer verilmeli, aksi halde bu eşik oturumdan oturuma kayar).

    Resmi koşula EK iki koruma (kısa koşular için — resmi kod EPOCHS>=150
    varsaydığından bu durumu hiç ele almıyor):
    - `finetune_last_epochs == 0` -> hep False (`config.py` yorumu: "choose 0 to skip").
    - `total_epochs <= |finetune_last_epochs|` (ör. EPOCHS=6, ft=-10) -> hep False:
      pencere başlangıcı (total+ft+1) epoch 1'in ÖNCESİNE düşerdi ve decay üssü
      daha ilk epoch'ta n>1 olurdu (ör. 0.9^5) — kısa keşif koşularında loss
      ağırlıklarını daha eğitim başlamadan bozmak anlamsız, hile tamamen atlanır.
      Bu koruma sayesinde hile uygulandığında üs her zaman n>=1'den başlar
      (epoch > total+ft >= 1 -> n = epoch-(total+ft) >= 1)."""
    if finetune_last_epochs == 0:
        return False
    if finetune_last_epochs < 0 and total_epochs <= -finetune_last_epochs:
        return False
    return epoch > total_epochs + finetune_last_epochs


def effective_lr(task: str, batch_size: int, accum_steps: int, base_lr_override: float | None = None) -> float:
    """BiRefNet resmi `config.py`'deki formülün ADAPTE edilmiş hali:
    `self.lr = (1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4)`
    (kaynak: `config.py`, GitHub `main`, `Config.__init__`). Resmi kodda gradient
    accumulation YOK (`train.py`'de `accelerator.gradient_accumulation_steps=1`
    SABİT kodlanmış, `accelerator.accumulate(...)` context'i YORUM SATIRINDA
    bırakılmış — kullanılmıyor); bu notebook gradient accumulation EKLEDİĞİ için
    formüldeki `batch_size`'ı, optimizer adımı başına GERÇEK (efektif) batch'e
    (`batch_size * accum_steps`) genişletiyoruz — resmi formülün "efektif batch
    büyüdükçe lr'yi karekök oranında büyüt" mantığını, artık iki eksende (fiziksel
    batch + accumulation) büyüyen efektif batch'e doğru şekilde uygulamak için.
    `base_lr_override` doluysa (parametre hücresinde `LR` elle ayarlanmışsa) bu
    hesap tamamen atlanır."""
    if base_lr_override is not None:
        return float(base_lr_override)
    base = 1e-4 if "DIS5K" in task else 1e-5
    effective_batch = batch_size * accum_steps
    return base * math.sqrt(effective_batch / 4)


# ============================================================================
# 5) BiRefNet config.py metin yaması (İDEMPOTENT)
# ============================================================================
_TASK_LIST = ["DIS5K", "COD", "HRSOD", "General", "General-2K", "Matting"]
_TASK_LINE_RE = re.compile(
    r"self\.task = (\['DIS5K', 'COD', 'HRSOD', 'General', 'General-2K', 'Matting'\])\[\d+\]"
)
_HOME_LINE_RE = re.compile(r"self\.sys_home_dir = \[os\.path\.expanduser\('~'\), '[^']*'\]\[1\]")
_BS_LINE_RE = re.compile(r"self\.batch_size = \d+")


def apply_config_patches(src: str, task: str, sys_home_dir: str, batch_size: int) -> str:
    """BiRefNet `config.py` kaynağına üç yamayı uygular: (1) `self.task` seçili
    indeksi, (2) `self.sys_home_dir` ikinci elemanı (data_root_dir'ın kökü),
    (3) `self.batch_size`. Satır desenleri GitHub `main` dalındaki `Config.__init__`
    ile doğrulandı (bkz. Faz 3 raporu).

    İDEMPOTENT ve yeniden-parametrelenebilir: regex, satırın HEM orijinal
    (yamasız) HEM önceden yamalanmış halini eşler — aynı VM'de notebook'un
    (aynı ya da FARKLI parametre değerleriyle) yeniden koşturulması hata vermez,
    `apply(apply(src)) == apply(src)`. Desen hiç eşleşmezse (repo `main` dalı
    değişmişse) SESSİZCE geçmek yerine net bir ValueError fırlatılır."""
    if task not in _TASK_LIST:
        raise ValueError(f"bilinmeyen görev: {task!r} (geçerli: {_TASK_LIST})")
    idx = _TASK_LIST.index(task)

    patched, n = _TASK_LINE_RE.subn(rf"self.task = \1[{idx}]", src, count=1)
    if n == 0:
        raise ValueError(
            "config.py'de beklenen `self.task = [...][N]` satırı bulunamadı — "
            "BiRefNet main dalı değişmiş olabilir, config.py'yi elle kontrol edin."
        )
    patched, n = _HOME_LINE_RE.subn(
        f"self.sys_home_dir = [os.path.expanduser('~'), '{sys_home_dir}'][1]", patched, count=1
    )
    if n == 0:
        raise ValueError("config.py'de beklenen `self.sys_home_dir = [...]` satırı bulunamadı.")
    patched, n = _BS_LINE_RE.subn(f"self.batch_size = {batch_size}", patched, count=1)
    if n == 0:
        raise ValueError("config.py'de beklenen `self.batch_size = N` satırı bulunamadı.")
    return patched


# ============================================================================
# 6) Drive -> yerel disk veri kopyalama (boyut doğrulamalı, idempotent)
# ============================================================================
def copy_pairs(
    stems: list[str],
    src_im_dir: str | Path,
    src_gt_dir: str | Path,
    dst_im_dir: str | Path,
    dst_gt_dir: str | Path,
    im_ext: str = ".jpg",
    gt_ext: str = ".png",
    max_workers: int = 16,
) -> int:
    """`stems` listesindeki (im, gt) çiftlerini kaynaktan hedefe kopyalar;
    kopyalanan çift sayısını döndürür. İdempotent: hedefte HEM im HEM gt varsa
    VE her ikisinin de dosya boyutu kaynakla birebir eşleşiyorsa atlanır —
    yalnız im boyutuna bakmak yetmez, yarım kalmış bir Colab kopyalamasında
    gt dosyası kesik (truncated) kalmış olabilir; o durumda çift YENİDEN
    kopyalanır (onarım).

    Drive FUSE bağlantısı üzerinden onbinlerce küçük dosyayı TEK İŞ PARÇACIĞIYLA
    kopyalamak saatler sürüyor (canlı bir Colab oturumunda ölçüldü); bu yüzden
    her çift bağımsız bir iş birimi olarak `ThreadPoolExecutor` ile `max_workers`
    kadar iş parçacığına dağıtılır (I/O-bound kopyalama — GIL burada engel
    değil). Her çiftin hedef dosyaları kendine özgü olduğundan iş parçacıkları
    arasında paylaşılan durum YOK (yarış koşulu riski yok); sonuç (kopyalanan
    sayı, atlanan/onarılan çiftler) sıralamadan BAĞIMSIZ, seri koşumla birebir
    aynıdır. Tek tek çiftlerdeki hatalar ANINDA fırlatılmaz — TÜMÜ toplanır,
    geri kalan tüm çiftler işlenir (kısmi ilerleme kaybolmaz), sonda İLK hata
    toplam hata sayısıyla birlikte yeniden fırlatılır. Her 2000 tamamlanan
    çiftte bir ilerleme (hız + ETA) konsola yazdırılır."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src_im_dir, src_gt_dir = Path(src_im_dir), Path(src_gt_dir)
    dst_im_dir, dst_gt_dir = Path(dst_im_dir), Path(dst_gt_dir)

    def _copy_one(stem: str) -> bool:
        src_im, src_gt = src_im_dir / f"{stem}{im_ext}", src_gt_dir / f"{stem}{gt_ext}"
        dst_im, dst_gt = dst_im_dir / f"{stem}{im_ext}", dst_gt_dir / f"{stem}{gt_ext}"
        if (
            dst_im.exists()
            and dst_gt.exists()
            and dst_im.stat().st_size == src_im.stat().st_size
            and dst_gt.stat().st_size == src_gt.stat().st_size
        ):
            return False
        shutil.copy2(src_im, dst_im)
        shutil.copy2(src_gt, dst_gt)
        return True

    total = len(stems)
    copied = 0
    completed = 0
    errors: list[tuple[str, BaseException]] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stem = {executor.submit(_copy_one, stem): stem for stem in stems}
        for future in as_completed(future_to_stem):
            stem = future_to_stem[future]
            completed += 1
            try:
                if future.result():
                    copied += 1
            except Exception as exc:  # per-item hata: topla, işlemeye devam et
                errors.append((stem, exc))
            if completed % 2000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"copy_pairs: {completed}/{total} tamamlandı "
                    f"({rate:.1f} çift/sn, ETA {eta:.0f}sn)"
                )

    if errors:
        first_stem, first_exc = errors[0]
        raise RuntimeError(
            f"copy_pairs: {len(errors)}/{total} çift kopyalanamadı "
            f"(ilk hata, stem={first_stem!r}: {first_exc!r})"
        ) from first_exc

    return copied


# ============================================================================
# 7) v3 — VAL sızıntı hariç tutma + kompozit manifest merge
# ============================================================================
_COMPOSITE_COPY_SUFFIX_RE = re.compile(r"_[vo]\d{2}$")


def strip_composite_copy_suffix(stem: str) -> str:
    """`<kaynak_id>_v<NN>` veya `<kaynak_id>_o<NN>` -> `<kaynak_id>` (bkz.
    `scripts/make_composites.py` isimlendirme sözleşmesi: `_v<NN>` compose'lu
    kopyalar, `_o<NN>` orijinal-arka-plan kopyaları). Eşleşme yoksa (beklenmeyen
    bir stem) `stem` OLDUĞU GİBİ döner.

    DİKKAT — eşleşmeme SIZINTI RİSKİDİR, zararsız değil: eşleşmeyen bir VAL
    stem'i hariç-tutma kümesine SON EKLİ (yanlış) haliyle girer; bu string
    kaynak manifest'teki hiçbir `id` ile eşleşmez, dolayısıyla ASIL kaynak id
    hariç tutulMAZ ve o kaynağın `_o00` kopyası eğitim setine ÜRETİLİR — koruma
    o kaynak için fiilen BAYPAS edilmiş olur (aynı görsel hem TRAIN'de `_o00`
    olarak hem VAL'de başka bir kopyasıyla görülür). Bu yüzden çağıranlar
    eşleşmeyen stem'leri MUTLAKA teşhis etmeli — `derive_val_excluded_source_
    ids` bunları ayrıca döndürür ve `training/v3_veri_guncelleme_hucresi.py`
    boş-olmayan bir eşleşmeme listesinde yüksek sesli uyarı basar."""
    return _COMPOSITE_COPY_SUFFIX_RE.sub("", stem)


def derive_val_excluded_source_ids(val_stems: list[str]) -> tuple[set[str], list[str]]:
    """VAL kümesindeki (kompozit) stem'lerden KAYNAK satır id'lerini türetir —
    bu id'ler `scripts/make_composites.py`'nin `_o00` üretiminden hariç
    tutulmalı (VAL sızıntı koruması): VAL_HOLDOUT zaten belirli `_v<NN>`/`_o<NN>`
    kopyalarını içeriyor olsa da, AYNI kaynak görsele ait BAŞKA bir `_o00`
    kopyasını eğitim setine eklemek, o görselin (farklı bir varyantla da olsa)
    hem TRAIN hem VAL'de görülmesi anlamına gelir — model o KAYNAK görseli
    ezberleyebilir. `training.train_colab_lib.load_or_create_val_split`in
    yazdığı `val_stems.json`daki `"val_stems"` listesi doğrudan bu fonksiyona
    verilir (bkz. `training/v3_veri_guncelleme_hucresi.py`).

    Dönüş: `(excluded_source_ids, unmatched_stems)`. `unmatched_stems`, son ek
    deseniyle (`_[vo]\\d{2}$`) EŞLEŞMEYEN val stem'leri — bunlar hariç-tutma
    kümesine olduğu gibi (son ekli/yanlış biçimde) girdiğinden kaynak
    manifest'teki hiçbir id ile eşleşmez ve koruma o kaynaklar için fiilen
    BAYPAS edilir (ayrıntı: `strip_composite_copy_suffix` docstring'i).
    Çağıran, `unmatched_stems` boş değilse bunu YÜKSEK SESLE raporlamalı
    (bkz. v3 hücresindeki `stage_composites_o` uyarısı)."""
    excluded: set[str] = set()
    unmatched: list[str] = []
    for s in val_stems:
        stripped = strip_composite_copy_suffix(s)
        if stripped == s:
            unmatched.append(s)
        excluded.add(stripped)
    return excluded, unmatched


def merge_composite_manifest(local_manifest_path: str | Path, drive_manifest_path: str | Path) -> int:
    """`local_manifest_path`teki satırları `drive_manifest_path`e APPEND eder —
    yalnız `drive_manifest_path`de henüz OLMAYAN `id`'ler (dedupe; idempotent:
    aynı çağrı iki kez yapılırsa ikinci çağrı 0 satır ekler). `drive_manifest_
    path` (v1/v2'nin TÜM `_v<NN>` satırlarını zaten içeren, büyük — ~28k+ satır
    — Drive kopyası) ASLA baştan okunup YENİDEN YAZILMAZ, yalnız açılıp eklenir
    (`benchmark.testset.append_entries`) — `training/v3_veri_guncelleme_
    hucresi.py`'nin `shutil.copy2` ile TAM üzerine yazan `colab_devam_hucresi.py`
    deseninden BİLİNÇLİ SAPMASI budur (o dosyada yerel kompozit manifest zaten
    TAM/güncel olduğundan üzerine yazmak güvenliydi; burada yerel manifest
    yalnız YENİ `_o00` satırlarını içeriyor). `local_manifest_path` yoksa
    (hiç `_o00` üretilmemişse) sessizce `0` döner.

    `drive_manifest_path` mevcutsa satırları TEK TEK okunup yalnız `id`
    alanları kümeye eklenir (tam `load_manifest` + `_validate` çağrılmaz —
    büyük dosyada yalnız id kümesi için gereksiz doğrulama/bellek maliyetinden
    kaçınmak için); `local_manifest_path` ise (küçük, ~14k satır) tam
    `load_manifest` ile okunur (tekrarlanan id koruması dahil)."""
    local_manifest_path = Path(local_manifest_path)
    drive_manifest_path = Path(drive_manifest_path)
    if not local_manifest_path.exists():
        return 0

    local_rows = load_manifest(str(local_manifest_path))
    existing_ids: set[str] = set()
    if drive_manifest_path.exists():
        with open(drive_manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    new_rows = [r for r in local_rows if r["id"] not in existing_ids]
    if new_rows:
        append_entries(str(drive_manifest_path), new_rows)
    return len(new_rows)


def ensure_manifest_pairs(manifest_path: str | Path, min_pairs: int = 1) -> int:
    """`manifest_path`teki GT'li (gt_alpha != null) satır sayısını döndürür;
    dosya yoksa veya sayı `min_pairs`'ın altındaysa NET bir `RuntimeError`
    fırlatır — boş/eksik bir manifest'le pipeline'ın devam edip çok daha
    aşağıda anlaşılmaz bir hatayla (ör. export'un FileNotFoundError'ı) çökmesini
    önler (canlı v3 koşusu dersi: ham veri hiç inmemişken manifest 0 çiftle
    kuruldu, hata ancak export aşamasında — SEMPTOM olarak — göründü; bu guard
    NEDENİ, manifest kurulumundan hemen sonra, yüksek sesle yakalar). Bkz.
    `training/v3_veri_guncelleme_hucresi.py` "manifest" aşaması sonu."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise RuntimeError(
            f"manifest dosyası yok: {manifest_path} — ham veri indirme/manifest kurulumu "
            f"başarısız olmuş olmalı; önceki aşamaların loglarını inceleyin."
        )
    n = sum(1 for r in load_manifest(str(manifest_path)) if r.get("gt_alpha"))
    if n < min_pairs:
        raise RuntimeError(
            f"manifest'te yalnız {n} GT'li çift var (< {min_pairs}): {manifest_path} — "
            f"ham veri kaynakları inmemiş/boş olabilir; export'a GEÇİLMEYECEK "
            f"(boş manifest'le devam etmek aşağıda anlaşılmaz hatalara yol açar)."
        )
    return n
