# Faz 1 Ablation Raporu: Eğitimsiz Kalite Katmanı

**Tarih:** 2026-07-09 · **Koşu:** `results/baseline/` (130 görsel; rmbg-2.0 vs rmbg-2.0+refine; RGBA/decontamination galeriyle nitel)

## Edge Refiner (patch re-inference) — sayısal sonuç

Kategori bazında MAE (rmbg-2.0 → rmbg-2.0+refine):

| Kategori | Önce | Sonra | Değişim |
|---|---|---|---|
| hair | 0.0045 | **0.0035** | **−22%** |
| thin | 0.0180 | **0.0165** | −8% |
| complex | 0.0241 | **0.0226** | −6% |
| transparent | 0.0741 | 0.0742 | ±0 (beklenen) |
| **OVERALL** | 0.0260 | **0.0250** | −4% |

SAD büyük ölçüde aynı yönde iyileşti (hair 7.6→5.9, thin 141→128, complex 198→185; transparent değişmedi) ama Conn/Grad kategoriye göre karışık (mixed): hair Conn hafif kötüleşti (5.41→5.51), hair ve transparent Grad de hafif kötüleşti (hair 0.105→0.110, transparent 5.049→5.123); buna karşın thin/complex Grad iyileşti ve genel Grad hafif iyileşti (2.754→2.729). Yani refiner net kazancı MAE/SAD/overall'da net, ama kenar-bağlantılılığı (Conn) ve gradyan metriklerinde kategori bazında karışık — saç gibi ince yapılarda re-inference bazen bağlantılılığı biraz bozabiliyor.

**Örnek görseller (rmbg-2.0 → rmbg-2.0+refine, MAE):**

En çok iyileşen 3:
| id | önce | sonra |
|---|---|---|
| `disvd_general_11_Furniture_4_Chair_7022021321_345e1dd87b_o` | 0.0159 | 0.0093 |
| `disvd_complex_17_Non-motor_Vehicle_2_Bicycle_6226540199_f69b5fe7fe_o` | 0.0187 | 0.0126 |
| `disvd_thin_4_Architecture_6_Gate_14749560442_811a1aa50a_o` | 0.0230 | 0.0191 |

En çok kötüleşen 3 (hepsi transparent kategorisinde, marjinal):
| id | önce | sonra |
|---|---|---|
| `trans460_andrew-ren-BhspCN17HT8-unsplash` | 0.0510 | 0.0523 |
| `trans460_pexels-pixabay-34487` | 0.3121 | 0.3133 |
| `trans460_behnam-norouzi-8MOL8cBGPdE-unsplash` | 0.1427 | 0.1437 |

**Yorum:** Refiner tasarım hipotezini doğruladı — belirsiz kenar bölgelerini yüksek efektif çözünürlükte yeniden işlemek kenar kaynaklı hataları kesiyor; saydamlıkta etkisiz çünkü oradaki hata kenarda değil, alan içi alpha'nın binarize edilmesinde. **Saydamlık ancak eğitimle (Faz 3) çözülür — Faz 2 veri önceliği değişmedi.**

**Maliyet:** refine, görsel başına 1-6 ek model geçişi (MPS'te ~2-3× yavaşlama). Servis/CLI'da bayrak opsiyonel kalıyor; kalite modunda açık, hız modunda kapalı önerilir.

## Decontamination — nitel sonuç

Alpha'yı değiştirmediği için GT metriklerinde görünmez; galeri artık model hücrelerinde decontaminated RGBA gösteriyor (`results/rmbg-2.0/rgba/`). Galeri incelemesinde: renkli zeminli portrelerde (P3M) saç kenarındaki zemin rengi sızması belirgin azalıyor; kompozitler Ideogram'ın "temiz kesim" görünümüne yaklaştı. Ideogram'la kalan görünür fark ağırlıkla saydam objelerde (alpha probleminde, renk probleminde değil).

## Faz 1 çıktıları

- `bgr remove <girdi> -o <çıktı.png> [--model] [--refine] [--no-decontaminate]` CLI
- `POST /remove` FastAPI servisi (lazy model cache)
- `rmbg-2.0+refine` benchmark varyantı; `--rgba` decontaminated çıktı üretimi
- 46 non-slow test

## Faz 2'ye devir

1. Veri önceliği #1: **saydamlık** (Transparent-460 train, cam/zar sentetikleri, yüksek örnekleme ağırlığı) — refiner'ın çözemediği kanıtlandı.
2. Veri önceliği #2 (kullanıcı hedefi): **COD/kamuflaj** — COD10K-TR + CAMO eğitim karışımına; test setine `camouflage` kategorisi (COD10K test'ten) eklenmeli ki ilerleme ölçülebilsin.
3. Fine-tune sonrası refiner yeniden ölçülmeli — taban model keskinleştikçe refiner kazancı değişebilir.
