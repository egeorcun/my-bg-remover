"""Faz 3 (`training/train_colab.ipynb`) için saf-Python, bağımlılıksız yardımcı
mantık. Bilinçli olarak `torch`/`PIL` içe aktarmaz — böylece hem Colab'da (BiRefNet
eğitim döngüsü içinde) hem de bu depoda GPU/torch olmadan `pytest` ile test
edilebilir (bkz. `tests/test_train_colab_lib.py`). Notebook bu dosyayı repo'yu
klonladıktan sonra `sys.path`'e ekleyip import eder (`scripts/` için kullanılan
"TEK DOĞRULUK KAYNAĞI" deseniyle aynı — bkz. `training/prepare_data_colab.ipynb`
hücre (d) notu) — mantık notebook içine kopyalanıp tekrar yazılmaz, drift riski
böylece ortadan kalkar.

Dört bağımsız endişe kapsar:
1. Kategori ağırlıklı örnekleme (`compute_sample_weights` / `compute_expected_shares`)
   — `torch.utils.data.WeightedRandomSampler`'a beslenecek ağırlıkları hesaplar.
2. Checkpoint keşfi/budama (`find_latest_checkpoint` / `prune_old_checkpoints`)
   — Colab oturum kopması sonrası otomatik devam + Drive disk kotasını sınırlama.
3. Deterministik TRAIN/VAL bölünmesi + sabit hızlı-değerlendirme alt kümesi
   (`deterministic_val_split` / `fixed_eval_subset`).
4. BiRefNet resmi `train.py`/`config.py` mantığının iki küçük parçasının yeniden
   üretimi (`should_apply_finetune_reweight`, `effective_lr`) — bkz. modül
   içindeki fonksiyon docstring'lerinde satır bazlı referanslar.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

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


def compute_sample_weights(
    stems: list[str],
    stem_category: dict[str, str],
    target_share: dict[str, float],
    default_category: str = "_other",
) -> list[float]:
    """`stems` (MyData.image_paths ile AYNI SIRADA olmalı — WeightedRandomSampler
    ağırlıkları dataset indeksleriyle hizalanmak zorunda) için, `target_share`'de
    adı geçen kategorilerin epoch-içi BEKLENEN payını `target_share`'e sabitleyen,
    geri kalan kategorilerin KENDİ ARALARINDAKİ göreli oranını (ham sayılarıyla
    orantılı) koruyan ağırlıklar üretir.

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
    categories = [stem_category.get(s, default_category) for s in stems]
    counts = Counter(categories)

    targeted = {c: share for c, share in target_share.items() if counts.get(c, 0) > 0}
    sum_targeted = sum(targeted.values())
    if sum_targeted >= 1.0:
        raise ValueError(f"target_share toplamı >= 1.0 olamaz (mevcut kategorilerde): {targeted}")
    remaining_mass = 1.0 - sum_targeted

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
    """BiRefNet resmi `train.py::Trainer.train_epoch` içindeki koşulun BİREBİR
    aynısı: `if epoch > args.epochs + config.finetune_last_epochs:` (kaynak:
    ZhengPeng7/BiRefNet `train.py`, GitHub `main` dalı, fonksiyon `train_epoch`,
    ~satır 195 civarı — bu depoya `curl` ile çekilip incelendi, bkz. Faz 3 raporu).
    `finetune_last_epochs` NEGATİF bir sayıdır (`config.py`'de `Matting` görevi
    için `-10` — son 10 epoch'ta pixel loss ağırlıkları kademeli değiştirilir,
    "belgelenen fine-tune hilesi"). `total_epochs`, o Colab OTURUMUNUN DEĞİL,
    eğitimin NİHAİ HEDEF epoch sayısıdır (`EPOCHS` parametresi — resume'lerde
    HER OTURUMDA AYNI değer verilmeli, aksi halde bu eşik oturumdan oturuma kayar)."""
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
