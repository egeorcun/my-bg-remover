# Baseline Gap Analizi (Ideogram vs açık modeller)

**Tarih:** 2026-07-09 (kategori düzeltmesi sonrası revize) · **Koşu:** `results/baseline/` (130 görsel, BiRefNet_HR + RMBG-2.0, MPS fp32) · **Referans:** Ideogram (fal.ai, 130 çıktı)

> **Revizyon notu:** İlk sürümdeki thin/complex/general etiketleri DIS-VD örneklerine rastgele atanmıştı (final review bulgusu). Kategoriler DIS5K dosya adı sınıf token'larından yeniden etiketlendi (33/65 satır değişti, `general` kategorisi kalktı); bu rapor düzeltilmiş verilerle yazıldı. İlk sürümün "thin'de BiRefNet önde" ve "Router'ı doğrular" iddiaları düzeltilmiş veride TUTMADI ve geri çekildi.

## Sayısal sonuçlar (GT'li kategoriler, düşük = iyi)

Genel (130 görsel):

| Model | SAD | MAE | MSE | Grad | Conn |
|---|---|---|---|---|---|
| BiRefNet_HR | 361.7 | 0.0287 | 0.0216 | **2.461** | 366.7 |
| RMBG-2.0 | **342.5** | **0.0260** | **0.0194** | 2.754 | **350.7** |

Kategori bazında MAE (parantez: Grad):

| Kategori (n) | BiRefNet_HR | RMBG-2.0 | Önde |
|---|---|---|---|
| hair (40) | 0.0048 (0.108) | **0.0045 (0.105)** | Berabere sayılır |
| thin (36) | 0.0196 **(3.62)** | **0.0180** (4.04) | Alanda RMBG, kenarda BiRefNet |
| complex (29) | 0.0385 (2.44) | **0.0241** (2.84) | RMBG, açık farkla |
| transparent (25) | **0.0687 (4.58)** | 0.0741 (5.05) | İkisi de çok kötü |

Okuma: **RMBG-2.0 alan doğruluğunda (özneyi bulma) tüm kategorilerde önde veya eşit; saydamlıkta ikisi de çöküyor** (diğer kategorilerin 3-15 katı hata). BiRefNet_HR'nin tutarlı tek avantajı kenar keskinliği (Grad, tüm kategorilerde). Fine-tune stratejimizin doğrulaması: BiRefNet mimarisi + RMBG kalitesinde veri = iki gücü birleştirmek (RMBG-2.0 zaten bunun kanıtı — aynı mimari, daha iyi veri).

## Kategori bazlı gözlemler (galeri + örnek incelemesi)

### transparent — ANA GAP
- Ders kitabı örneği `trans460_pexels-pixabay-34487` (yusufçuk kanadı, MAE 0.312): RMBG-2.0 saydam kanat zarını **komple opak beyaz blob** olarak işaretliyor — damar yapısı ve kısmi saydamlık tamamen kayıp. Ideogram aynı görselde zarı hücre düzeyinde yarı saydam alpha ile koruyor.
- Kök neden: açık modeller binary-ish maskeye eğilimli; "pikselin kendisi kısmen arka plan" durumlarında ara-alpha üretmiyorlar.

### complex — RMBG açık farkla önde (0.0241 vs 0.0385)
- BiRefNet_HR karmaşık sahnelerde parça atlama / yanlış obje dahil etme yapıyor (dişçi koltuğu 0.078, balon halatları 0.058). RMBG'nin küratörlü verisi "özne hangisi" sorusunda daha isabetli. Fine-tune veri karışımında çok nesneli sahnelere ağırlık gerektiğinin kanıtı.

### thin — alan hatasında RMBG önde, kenar keskinliğinde BiRefNet
- En kötüler delikli/örgülü yapılar (raket ağı 0.077): ağ delikleri opak dolduruluyor. Bu aslında saydamlıkla akraba bir hata türü (arka planın "içinden göründüğü" bölgeler).

### hair — gap küçük
- İki model de ~0.005 MAE; saç bu modeller için artık zayıf nokta değil. Ancak galeri kompozitlerinde Ideogram'ın kenar renk temizliği hâlâ daha temiz — bu fark metriklere girmiyor (GT alpha karşılaştırması renk sızmasını ölçmez), Faz 1 decontamination tam bunu hedefliyor.

### product / illustration — GT yok, veri bekliyor
- `data/testset/incoming/` hâlâ boş; kullanıcı görselleri eklenince galeri karşılaştırmasıyla güncellenecek. Tasarım spec'inin 2. başarı kriteri (galeri incelemesinde Ideogram'dan ayırt edilemezlik) bu kategorilerde henüz test edilmedi.

## Sonuç: Faz 2 veri öncelikleri

1. **Saydamlık (açık ara en yüksek öncelik):** Transparent-460 train split + cam/zar/duman sentetik kompozitleri; transparent kategorisine yüksek örnekleme ağırlığı. Hedef: binarizasyonu kırıp ara-alpha üretmeyi öğretmek. Delikli/örgülü thin yapılar da aynı aileden — birlikte ele alınmalı.
2. **Karmaşık sahne / özne seçimi:** DIS5K-TR tamamı + çok nesneli kompozitler — BiRefNet tabanının complex'te RMBG'ye yetişmesi için (0.0385 → 0.024 hedefi).
3. Saç: mevcut karışım yeterli; ağırlık artırmaya gerek yok.

## Sonuç: Faz 1 post-processing beklentileri

- **Decontamination (kesin kazanç):** GT metriklerinde görünmez ama kompozit kalitesinde Ideogram'la "temizlik" farkının ana kaynağı.
- **Edge Refiner:** hair/thin kenarlarında Grad iyileşmesi beklenir; transparent'ta tek başına yetmez (sorun kenar değil, alan içi alpha).
- **Router (düzeltilmiş değerlendirme):** Düzeltilmiş veride kategori şampiyonluğu ayrışması YOK — RMBG-2.0 GT'li tüm kategorilerde önde veya eşit (transparent'ta ikisi de kullanılamaz durumda; BiRefNet'in oradaki küçük farkı pratik değer taşımıyor). **Baseline verisi Router'a gerekçe sunmuyor**; Router kararı product/illustration verisi ve Faz 1 ablation'ları geldiğinde yeniden değerlendirilecek. Şimdilik en güçlü tek-model baseline: RMBG-2.0 (kıyas için), fine-tune tabanı: BiRefNet_HR-matting (lisans kararı gereği).

## Ham çıktılar

- Metrikler: `results/baseline/metrics.json` · Galeri (RGBA kompozitli): `results/baseline/gallery.html`
- Manifest (git'e sabitlendi): `data/testset/manifest.jsonl` · Loglar: `results/baseline_run.log`, `results/ideogram_fetch.log`
