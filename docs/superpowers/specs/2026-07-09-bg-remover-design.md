# Design: Ideogram Seviyesinde Background Remover

**Tarih:** 2026-07-09 · **Durum:** Onaylandı (brainstorming akışıyla bölüm bölüm)

## 1. Amaç ve başarı kriteri

`fal-ai/ideogram/remove-background` kalitesinde, genel amaçlı (insan, ürün, hayvan, illüstrasyon) bir background remover. Kullanım: kişisel/araştırma (non-commercial model ağırlıkları ve research-only dataset'ler kullanılabilir). Şimdilik tamamen lokal (M4 Max 48GB); deploy kararı model kanıtlanınca.

**Başarı kriteri:**
1. GT alpha'lı test setlerinde SAD/MAE/Grad/Conn metriklerinde RMBG-2.0 baseline'ını geçmek.
2. Zor kategorilerde (saç/kürk, cam/saydamlık, ince yapılar) yan yana galeri incelemesinde Ideogram'dan ayırt edilemez veya daha iyi olmak.

**Temel içgörü (araştırmadan):** Ticari modellerin üstünlüğü mimari değil, veri + post-processing. RMBG-2.0 birebir BiRefNet mimarisidir; farkı 15K+ elle etiketlenmiş veri ve kenar post-processing'i yaratır. Bu yüzden strateji: BiRefNet tabanı + en iyi açık ağırlıklardan fine-tune + eğitimsiz kalite katmanı.

## 2. Mimari

Dört modüllü Python paketi `bgr/`. Her modül tek başına test edilebilir; ortak sözleşme: `PIL.Image -> alpha matte (float32 HxW, 0..1)`.

| Modül | Görev | V1 kapsamı |
|---|---|---|
| **Router** | Görsel tipini algıla (insan/saç, ürün, saydam, illüstrasyon), modeli seç | Kural bazlı basit heuristik/hafif sınıflandırıcı (kullanıcı isteğiyle V1'de) |
| **Segmenter** | Alpha matte üret | Değiştirilebilir arayüz: RMBG-2.0 (genel), BiRefNet_HR-matting (saç/detay), ToonOut-BiRefNet (illüstrasyon); fine-tune checkpoint'i hazır olunca tek satırla devreye girer |
| **Edge Refiner** | Düşük güvenli bölgeleri (alpha 0.1–0.9) ikinci geçişle rafine et (BEN2 CGM fikri) | Aç/kapa bayraklı; benchmark iki modda ölçer |
| **Decontaminator** | Kenar renk sızmasını temizle (FBA/closed-form foreground estimation) | Her zaman açık; eğitim gerektirmez |

Sarmalayıcılar: `bgr remove input.jpg -o out.png` CLI + küçük FastAPI endpoint. Çıktı: doğru premultiplied-alpha PNG.

SAM3 ana pipeline'da **yok** (prompt'lu segmentasyon farklı problem; binary maske üretir, alpha değil). Olası kullanım: 2. eğitim iterasyonunda veri motoru (ZIM tarzı kaba maske → alpha pseudo-label).

## 3. Veri ve eğitim

- **Dataset'ler:** DIS5K, P3M-10k, AM-2k, AIM-500, Distinctions-646, HIM2K, Transparent-460, PPM-100, BG-20k (arka plan havuzu), ToonOut (illüstrasyon); opsiyonel gated Adobe-1K.
- **Compositing:** alpha'lı foreground'lar BG-20k üzerine; augmentasyon (renk/ışık jitter, JPEG artifact, ölçek, blur).
- **Birleşik format:** image + alpha + kategori etiketi; kategori bazlı örnekleme ağırlıkları.
- **Eğitim:** BiRefNet resmi training kodu; başlangıç ağırlığı RMBG-2.0. Colab Pro A100 (aylık plan, ₺165,60; ~100 birim ≈ 8-11 saat A100). 1024px, multi-stage supervision, gradient accumulation (bs≥2; bs=1'de bilinen checkpoint bug'ı — BiRefNet issue #140). Her epoch Google Drive'a checkpoint + tam resume.
- **Lokal debug:** MPS'te 512px/batch 1/birkaç yüz step, `PYTORCH_ENABLE_MPS_FALLBACK=1`, PyTorch ≥2.4.
- **İterasyon döngüsü:** eğit → benchmark → zayıf kategorilere veri ekle/ağırlıkla → tekrar.

## 4. Değerlendirme

- ~150-200 görsellik kategorili test seti (saç/kürk, cam/saydam, ince yapı, ürün, karmaşık sahne, illüstrasyon). Kaynak: AIM-500, P3M-500-NP, Transparent-460 test, DIS-VD + gerçek dünya görselleri.
- Ideogram referans çıktıları fal.ai API'den (~$2, $0.01/görsel).
- GT'li setlerde SAD/MAE/Grad/Conn; yan yana HTML galeri (giriş / bizim / Ideogram / fark haritası).
- Her modülün katkısı ablation ile ayrı ölçülür (refiner açık/kapalı, decontamination açık/kapalı).

## 5. Fazlar

0. **Kurulum + baseline benchmark** — venv (uv), model inference (MPS), test seti, Ideogram referansları, harness, gap analizi raporu.
1. **Lokal pipeline** — Segmenter arayüzü + Router + Decontaminator + Refiner + CLI/FastAPI.
2. **Veri pipeline'ı** — indirme, compositing, birleşik format.
3. **Fine-tune** — Colab Pro A100, RMBG-2.0'dan; iterasyonlu.
4. **Paketleme** — checkpoint entegrasyonu, ONNX export (+ opsiyonel CoreML/ANE, M4 Max'te ~3-5 sn/görsel).

## 6. Ortam notları (2026-07-09 tespiti)

- HF cache: `ZhengPeng7/BiRefNet_HR` tam indirilmiş (424MB); `briaai/RMBG-2.0` sadece metadata — gated, HF login + lisans onayı gerekli.
- Sistem Python 3.12'de torch yok → `uv` ile izole venv.
- Proje dizini: `~/Documents/Projects/my-bg-remover/` — yapı: `bgr/`, `benchmark/`, `data/` (git dışı), `training/`, `serving/`, `docs/`.

## 7. Hata yönetimi ve sınırlar

- Router yanlış sınıflandırırsa: her zaman genel model fallback'i; CLI'da `--model` ile manuel override.
- MPS'te op fallback/OOM: `PYTORCH_ENABLE_MPS_FALLBACK=1`, gerektiğinde çözünürlük düşürme; inference'ta fp16.
- Colab oturum kopması: her epoch checkpoint + resume; eğitim scripti idempotent.
- Gated dataset'ler (BG-20k, Adobe-1K anlaşma ister): bloklamadan ilerle, alternatif arka plan havuzuyla (ör. telifsiz koleksiyon) başla, anlaşma gelince değiştir.

## 8. Riskler

| Risk | Etki | Önlem |
|---|---|---|
| A100 kuyruğu/birim tükenmesi | Eğitim gecikir | Checkpoint/resume; ek birim; L4 yedek (düşük çözünürlüklü ön deneme) |
| Fine-tune RMBG-2.0'ı geçemez | Kalite hedefi | Data-centric iterasyon; post-processing katmanı zaten eğitimsiz kazanç sağlar |
| Compositing domain gap | Gerçek görsellerde düşüş | Gerçek dünya test seti ile ölçüm; augmentasyon; gerekirse SAM3 pseudo-label |

## Kaynaklar

- BiRefNet: https://github.com/ZhengPeng7/BiRefNet
- RMBG-2.0: https://huggingface.co/briaai/RMBG-2.0
- BEN2 (CGM): https://github.com/PramaLLC/BEN2
- ToonOut: https://arxiv.org/abs/2509.06839
- Dataset hub: https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting
- FBA Matting: https://github.com/MarcoForte/FBA_Matting
- Ideogram referansı: https://fal.ai/models/fal-ai/ideogram/remove-background
