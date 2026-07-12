"""v4 eğitimi için 3 yeni kategorinin (text / fx / illustration) veri üreticisi.

Colab'da (CPU yeterli, GPU GEREKMEZ) koşacak şekilde tasarlandı — `run()` import
edilebilir (bkz. `training/v3_veri_guncelleme_hucresi.py`'nin scripts/ import
kalıbı), CLI ile de koşulabilir.

Kategoriler:
- **text (~4.000):** PIL ile sentetik yazı/logo render'ı. Rastgele font
  (`--font-dir` içindeki .ttf/.otf/.ttc glob'u; yoksa PIL varsayılan fontu),
  1-3 rastgele "marka-vari" kelime, punto (kanvas kısa kenarının %5-%40'ı),
  renk, konum, rotasyon (±30°) ve efektler: kontur (stroke), gölge
  (offset+blur), glow (alpha'nın MaxFilter+GaussianBlur ile genişletilmiş,
  0.35-0.8 çarpanlı YARI SAYDAM kopyası). Bazı örneklerde yazının arkasına
  basit vektör rozet (daire / rounded-rect / yıldız) eklenir (logo görünümü).
  GT alpha = render'ın KENDİ alpha'sı (gölge+glow dahil) — binarize EDİLMEZ.
  ARA-ALPHA GARANTİSİ: her örnekte en az bir yumuşak efekt (glow veya blur'lu
  gölge) zorunlu kılınır; ikisi de rastgele seçilmediyse glow'a düşülür —
  gt'de 0/255 dışı ara değerler her örnekte bulunur (anti-aliasing tek başına
  garanti sayılmaz). Arka plan: `--bg-dir` (BG-20k) havuzundan rastgele
  kırpılmış gerçek fotoğraf, %20 olasılıkla (`FLAT_BG_PROB`) düz/gradyan renk.
- **fx (~3.500):** `--fg-dirs` köklerindeki (her kök `im/` + `gt/` alt
  dizinleri, stem eşleşmeli) mevcut alpha'lı foreground'ların etrafına
  prosedürel VFX parıltısı: glow halkası (nesne alpha'sının MaxFilter +
  GaussianBlur ile DIŞA doğru bulanığı — HER örnekte uygulanır, ara-alpha
  garantisinin fx ayağı), gaussian çekirdekli parçacık parıltıları (bir kısmı
  4 kollu yıldız), lens-flare-vari ince ışık çizgileri. Parıltı renkleri
  parlak (beyaz/altın/cyan `_FX_PALETTE` + jitter), eleman alpha'ları yarı
  saydam ve birleşik fx alpha'sı `FX_ALPHA_MAX`=0.9 ile kırpılır. Yeni
  alpha = max(fg_alpha, fx_alpha); yeni RGB = fg'nin gerçek arka plana
  kompoziti üzerine parıltı enerjisinin SCREEN blend'i (out = 1-(1-base)(1-E),
  E = eleman bazında screen ile biriktirilen premultiplied renk) — model
  "obje + etrafındaki parıltılar birlikte foreground" ilişkisini görür.
- **illustration (~3.600):** ToonOut'un HAZIR im/gt çiftleri
  (`--toonout-dir/im`, `--toonout-dir/gt`) kullanılır — dataset indirme
  sorumluluğu BU SCRIPTTE DEĞİL. Çift başına 3 kopya: c00/c01 `bgr.
  compositing.compose` ile bg havuzuna kompozit + `augment` (renk jitter,
  JPEG artifact — make_composites.py'deki kalıbın BİREBİR aynısı), c02 =
  orijinal görüntü (compose YOK, yalnız augment — make_composites'ın `_o00`
  kopyalarıyla aynı mantık). `count` hedefine `ceil(count/3)` çift yeter;
  varsayılan 3600 hedefi ToonOut havuzunun ~%50'sine denk düşecek şekilde
  seçildi (sıralı-deterministik ilk N çift kullanılır).

SÖZLEŞMELER (bkz. scripts/make_composites.py):
- Dosya adı stem kalıbı: `{category}_{index:05d}_c{copy:02d}` — text'te copy
  hep 00 (her indeks bağımsız sentez), fx'te indeks = kaynak fg indeksi ve
  copy o kaynağın kopya sırası, illustration'da indeks = ToonOut çift indeksi
  ve c00/c01 kompozit, c02 orijinal. Manifest id'si = stem.
- Çıktı düzeni: `out_dir/im/{stem}.jpg` (RGB, JPEG q92) + `out_dir/gt/
  {stem}.png` (L modu 8-bit alpha) — `_save_pair` make_composites ile aynı.
- Manifest: her çift için `{"id": stem, "category": ...}` satırı JSONL'e
  APPEND (`out_manifest`, varsayılan `out_dir/manifest.jsonl`). Kategoriler
  benchmark.testset.CATEGORIES kümesinde OLMADIĞINDAN (text/fx yeni) o
  modülün append_entries/load_manifest'i BİLİNÇLİ olarak kullanılmaz.
- Determinizm: `_item_rng(seed, stem)` — make_composites.py'deki
  np.random.SeedSequence kalıbının birebir kopyası; aynı seed + aynı stem ->
  bit-identical çıktı, işlem sırasından ve atlanmış öğelerden bağımsız
  (resume güvenliği).
- İdempotentlik: im+gt çifti diskte zaten varsa üretim atlanır; dosya var ama
  manifest satırı eksikse (kaydetme ile append arasında kesinti) yalnız satır
  tamamlanır, dosya yeniden ÜRETİLMEZ.
- Bellek: görseller tek tek işlenir; 2048px üzeri kaynaklar LANCZOS ile
  küçültülür (`PIL.Image.MAX_IMAGE_PIXELS = None` — bkz. training/
  v3_veri_guncelleme_hucresi.py aynı not: 100MP+ akademik kaynaklar).

Kullanım:
    uv run python scripts/make_textfx.py --out-dir data/train_textfx \
        --bg-dir data/backgrounds --fg-dirs data/raw_train/p3m data/raw_train/camo \
        --toonout-dir data/raw_train/toonout --font-dir data/fonts --seed 42 \
        --counts text=4000,fx=3500,illustration=3600
    # --counts'ta verilmeyen kategori 0 sayılır (yalnız verilenler üretilir):
    uv run python scripts/make_textfx.py --out-dir out --bg-dir bgs --counts text=100
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from bgr.compositing import augment, compose

# ToonOut/fg kaynaklarında 100MP+ görsel olabilir; PIL'in 179MP "decompression
# bomb" eşiği güvenilir kaynaklar için kaldırılır (bkz. modül docstring'i).
Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FONT_EXTS = {".ttf", ".otf", ".ttc"}
MAX_SIDE = 2048
DEFAULT_COUNTS: dict[str, int] = {"text": 4000, "fx": 3500, "illustration": 3600}
FLAT_BG_PROB = 0.2
BADGE_PROB = 0.35
FX_ALPHA_MAX = 0.9
ILLUSTRATION_COPIES = 3  # c00/c01 compose+augment, c02 orijinal (yalnız augment)
# Parlak parıltı paleti (0-1 RGB): beyaz / altın / cyan aralığı (+ jitter).
_FX_PALETTE: list[tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (1.0, 0.85, 0.4),
    (1.0, 0.75, 0.2),
    (0.4, 0.95, 1.0),
    (0.7, 1.0, 1.0),
]
_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


# ==========================================================================
# Ortak yardımcılar (kaynak: scripts/make_composites.py — aynı sözleşmeler)
# ==========================================================================
def _item_rng(seed: int, key: str) -> np.random.Generator:
    """(global seed, öğe anahtarı) çiftinden bağımsız/deterministik rastgele akış.

    İşlem sırasından ve önceden atlanmış (zaten var olan) öğelerden ETKİLENMEZ —
    her öğe kendi id'sinden türetilen sabit bir alt-seed kullanır.
    (Kaynak: scripts/make_composites.py::_item_rng, birebir kopya.)
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    entropy = [seed & 0xFFFFFFFF] + [
        int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _save_pair(rgb: np.ndarray, alpha: np.ndarray, img_path: Path, gt_path: Path) -> None:
    """Kaynak: scripts/make_composites.py::_save_pair — aynı kayıt sözleşmesi."""
    img_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(img_path, format="JPEG", quality=92)
    Image.fromarray(np.round(alpha.clip(0, 1) * 255).astype(np.uint8), mode="L").save(gt_path)


def _load_rgb_capped(path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """RGB yükler; uzun kenar `max_side`'ı aşıyorsa LANCZOS ile küçültür (bellek)."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize(
            (max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS
        )
    return np.asarray(im, dtype=np.uint8)


def _load_alpha(path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """target_size = (w, h); boyut uyuşmuyorsa alpha yeniden ölçeklenir
    (kaynak: make_composites.py::_load_alpha)."""
    im = Image.open(path).convert("L")
    if im.size != target_size:
        im = im.resize(target_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _list_images(directory: Path | None) -> list[Path]:
    if not directory:
        return []
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def _pairs_from_dir(root: Path) -> list[tuple[Path, Path]]:
    """`root/im` + `root/gt` altındaki dosyaları stem'e göre eşler (sıralı)."""
    root = Path(root)
    gts = {p.stem: p for p in _list_images(root / "gt")}
    return [(p, gts[p.stem]) for p in _list_images(root / "im") if p.stem in gts]


def _load_manifest_ids(path: Path) -> set[str]:
    """Çıktı manifest'indeki id'ler (resume'da satır tekrarını önlemek için).

    benchmark.testset.load_manifest KULLANILMAZ: text/fx kategorileri o modülün
    CATEGORIES kümesinde yok (bkz. modül docstring'i sözleşmeler)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def _append_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ==========================================================================
# Arka plan seçimi (gerçek kırpma / düz-gradyan sentetik)
# ==========================================================================
def _synthetic_bg(rng: np.random.Generator, size: tuple[int, int]) -> np.ndarray:
    """Düz renk veya iki renkli lineer gradyan (text kategorisinin %20 dalı)."""
    w, h = size
    c1 = rng.integers(0, 256, 3).astype(np.float32)
    if rng.uniform() < 0.5:
        return np.ascontiguousarray(np.broadcast_to(c1.round(), (h, w, 3))).astype(np.uint8)
    c2 = rng.integers(0, 256, 3).astype(np.float32)
    horizontal = rng.uniform() < 0.5
    n = w if horizontal else h
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    grad = c1[None, :] * (1 - t) + c2[None, :] * t  # (n, 3)
    arr = grad[None, :, :] if horizontal else grad[:, None, :]
    return np.ascontiguousarray(np.broadcast_to(arr, (h, w, 3)).round()).astype(np.uint8)


def _bg_crop(rng: np.random.Generator, bg_paths: list[Path], size: tuple[int, int]) -> np.ndarray:
    """Havuzdan rastgele bir gerçek arka planın rastgele kırpılmış (w, h) parçası."""
    arr = _load_rgb_capped(bg_paths[int(rng.integers(0, len(bg_paths)))])
    bh, bw = arr.shape[:2]
    w, h = size
    scale = max(w / bw, h / bh) * float(rng.uniform(1.0, 1.4))  # cover + hafif zoom
    nw, nh = max(w, round(bw * scale)), max(h, round(bh * scale))
    if (nw, nh) != (bw, bh):
        arr = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    x0 = int(rng.integers(0, nw - w + 1))
    y0 = int(rng.integers(0, nh - h + 1))
    return np.ascontiguousarray(arr[y0 : y0 + h, x0 : x0 + w])


def _pick_bg(
    rng: np.random.Generator,
    bg_paths: list[Path],
    size: tuple[int, int],
    flat_prob: float = FLAT_BG_PROB,
) -> np.ndarray:
    if not bg_paths or rng.uniform() < flat_prob:
        return _synthetic_bg(rng, size)
    return _bg_crop(rng, bg_paths, size)


# ==========================================================================
# text kategorisi — sentetik yazı/logo render'ı
# ==========================================================================
def _load_font_paths(font_dir: Path | None) -> list[Path]:
    if not font_dir:
        return []
    font_dir = Path(font_dir)
    if not font_dir.is_dir():
        return []
    return sorted(p for p in font_dir.rglob("*") if p.suffix.lower() in FONT_EXTS)


def _renders_latin(font: ImageFont.ImageFont) -> bool:
    """Font Latin glifleri gerçekten çizebiliyor mu? Latin içermeyen fontlar
    (ör. macOS Supplemental'daki sembol/CJK fontları) her harfi AYNI "tofu"
    kutusuyla basar — "I" ve "W" maskeleri birebir aynıysa font kullanılamaz."""
    try:
        m_i, m_w = font.getmask("I"), font.getmask("W")
    except OSError:
        return False
    return m_i.size != m_w.size or bytes(m_i) != bytes(m_w)


def _get_font(font_paths: list[Path], rng: np.random.Generator, size: int) -> ImageFont.ImageFont:
    for _ in range(8):  # Latin desteği olmayan font seçilirse yeniden dene
        if not font_paths:
            break
        path = font_paths[int(rng.integers(0, len(font_paths)))]
        try:
            font = ImageFont.truetype(str(path), size)
        except OSError:
            continue  # bozuk/okunamayan font dosyası -> yeni deneme
        if _renders_latin(font):
            return font
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1: load_default() boyut parametresi almaz
        return ImageFont.load_default()


def _rand_text(rng: np.random.Generator) -> str:
    """1-3 kelimelik, harf/rakam karışımı marka-vari kısa string."""
    words = []
    for _ in range(int(rng.integers(1, 4))):
        n = int(rng.integers(3, 9))
        words.append("".join(_CHARS[int(rng.integers(0, len(_CHARS)))] for _ in range(n)))
    return " ".join(words)


def _bright_color(rng: np.random.Generator) -> tuple[int, int, int]:
    c = rng.integers(96, 256, 3)
    c[int(rng.integers(0, 3))] = int(rng.integers(192, 256))
    return (int(c[0]), int(c[1]), int(c[2]))


def _rand_color(rng: np.random.Generator) -> tuple[int, int, int]:
    c = rng.integers(0, 256, 3)
    return (int(c[0]), int(c[1]), int(c[2]))


def _draw_text_rgba(
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    stroke_width: int,
    stroke_fill: tuple[int, int, int] | None,
    pad: int,
) -> Image.Image:
    """Metni kendi sıkı-kadrajlı RGBA katmanına çizer (pad: efekt taşma payı)."""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    return img


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int = 5) -> list[tuple[float, float]]:
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        ang = -math.pi / 2 + k * math.pi / n
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _add_badge(base: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Yazının arkasına basit vektör rozet (daire / rounded-rect / yıldız) —
    logo görünümü. Rozet dolgusu yarı saydam olabilir (alpha 140-255)."""
    gw, gh = base.size
    m = max(4, int(0.2 * max(gw, gh)))
    group = Image.new("RGBA", (gw + 2 * m, gh + 2 * m), (0, 0, 0, 0))
    d = ImageDraw.Draw(group)
    color = _bright_color(rng) + (int(rng.integers(140, 256)),)
    shape = int(rng.integers(0, 3))
    gw2, gh2 = group.size
    if shape == 0:
        d.ellipse([m // 2, m // 2, gw2 - m // 2, gh2 - m // 2], fill=color)
    elif shape == 1:
        d.rounded_rectangle(
            [m // 2, m // 2, gw2 - m // 2, gh2 - m // 2],
            radius=max(2, min(gw2, gh2) // 6),
            fill=color,
        )
    else:
        r_out = min(gw2, gh2) / 2 - 1
        d.polygon(_star_points(gw2 / 2, gh2 / 2, r_out, r_out * 0.45), fill=color)
    group.alpha_composite(base, (m, m))
    return group


def _text_group(rng: np.random.Generator, canvas_min: int, font_paths: list[Path]) -> Image.Image:
    """Metin (+ opsiyonel kontur ve rozet) içeren rotasyonsuz RGBA grup katmanı."""
    text = _rand_text(rng)
    font_size = max(8, int(canvas_min * float(rng.uniform(0.05, 0.40))))
    font = _get_font(font_paths, rng, font_size)
    fill = (_bright_color(rng) if rng.uniform() < 0.7 else _rand_color(rng)) + (255,)
    stroke_width, stroke_fill = 0, None
    if rng.uniform() < 0.5:
        stroke_width = max(1, font_size // 12)
        stroke_fill = _rand_color(rng)
    pad = max(6, font_size // 2)  # gölge/glow taşma payı
    base = _draw_text_rgba(text, font, fill, stroke_width, stroke_fill, pad)
    if rng.uniform() < BADGE_PROB:
        base = _add_badge(base, rng)
    return base


def _decorate(group: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Gölge (offset + gaussian blur) ve/veya glow ekler.

    ARA-ALPHA GARANTİSİ: ikisi de rastgele seçilmediyse glow'a zorlanır — her
    text örneğinin gt'sinde 0/255 dışı ara alpha değerleri bulunur (glow ve
    blur'lu gölge yarı saydam alpha üretir; bkz. modül docstring'i)."""
    a = group.getchannel("A")
    out = Image.new("RGBA", group.size, (0, 0, 0, 0))
    use_shadow = rng.uniform() < 0.45
    use_glow = rng.uniform() < 0.6
    if not use_shadow and not use_glow:
        use_glow = True

    if use_shadow:
        off = max(1, int(0.02 * max(group.size) * float(rng.uniform(1.0, 3.0))))
        dx = off if rng.uniform() < 0.5 else -off
        dy = max(1, int(off * float(rng.uniform(0.5, 1.5))))
        sa = ImageChops.offset(a, dx, dy).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(0.8, 3.0)))
        )
        opacity = float(rng.uniform(0.4, 0.8))
        sa = Image.fromarray((np.asarray(sa, dtype=np.float32) * opacity).astype(np.uint8), "L")
        shadow_color = rng.integers(0, 64, 3)
        shadow = Image.new("RGBA", group.size, (int(shadow_color[0]), int(shadow_color[1]), int(shadow_color[2]), 0))
        shadow.putalpha(sa)
        out.alpha_composite(shadow)

    if use_glow:
        radius = max(1.5, 0.04 * max(group.size) * float(rng.uniform(0.5, 1.5)))
        ga = a.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius))
        strength = float(rng.uniform(0.35, 0.8))  # YARI SAYDAM — binarize edilmez
        ga = Image.fromarray((np.asarray(ga, dtype=np.float32) * strength).astype(np.uint8), "L")
        glow_color = (255, 255, 255) if rng.uniform() < 0.5 else _bright_color(rng)
        glow = Image.new("RGBA", group.size, glow_color + (0,))
        glow.putalpha(ga)
        out.alpha_composite(glow)

    out.alpha_composite(group)
    return out


def _render_text_sample(
    rng: np.random.Generator,
    size: tuple[int, int],
    bg_paths: list[Path],
    font_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Tek text örneği: (kompozit RGB uint8, alpha float32 [0,1])."""
    w, h = size
    group = _decorate(_text_group(rng, min(w, h), font_paths), rng)
    group = group.rotate(float(rng.uniform(-30, 30)), expand=True, resample=Image.BICUBIC)

    # kanvasa sığdır (büyük punto + rotasyon kanvası aşabilir)
    max_w, max_h = int(0.95 * w), int(0.95 * h)
    if group.width > max_w or group.height > max_h:
        s = min(max_w / group.width, max_h / group.height)
        group = group.resize(
            (max(1, int(group.width * s)), max(1, int(group.height * s))), Image.LANCZOS
        )

    x0 = int(rng.integers(0, w - group.width + 1))
    y0 = int(rng.integers(0, h - group.height + 1))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(group, (x0, y0))

    fg = np.asarray(canvas, dtype=np.float32)
    alpha = fg[..., 3] / 255.0
    bg = _pick_bg(rng, bg_paths, (w, h)).astype(np.float32)
    rgb = fg[..., :3] * alpha[..., None] + bg * (1 - alpha[..., None])
    return rgb.round().clip(0, 255).astype(np.uint8), alpha.astype(np.float32)


def gen_text(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    bg_paths: list[Path],
    font_paths: list[Path],
    seed: int,
    existing_ids: set[str],
    canvas_range: tuple[int, int] = (448, 768),
) -> tuple[list[dict], int, int]:
    """(manifest satırları, üretilen çift sayısı, atlanan çift sayısı) döndürür."""
    new_rows: list[dict] = []
    generated = skipped = 0
    lo, hi = canvas_range
    for i in range(count):
        stem = f"text_{i:05d}_c00"
        img_path = out_im_dir / f"{stem}.jpg"
        gt_path = out_gt_dir / f"{stem}.png"
        row = {"id": stem, "category": "text"}
        if img_path.exists() and gt_path.exists():
            skipped += 1
            if stem not in existing_ids:
                new_rows.append(row)  # dosya var, manifest satırı eksik -> yalnız satır
            continue
        rng = _item_rng(seed, stem)
        w = int(rng.integers(lo, hi + 1))
        h = int(rng.integers(lo, hi + 1))
        rgb, alpha = _render_text_sample(rng, (w, h), bg_paths, font_paths)
        _save_pair(rgb, alpha, img_path, gt_path)
        new_rows.append(row)
        generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# fx kategorisi — foreground etrafına prosedürel VFX parıltısı
# ==========================================================================
def _fx_color(rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(_FX_PALETTE[int(rng.integers(0, len(_FX_PALETTE)))], dtype=np.float32)
    return np.clip(base + rng.uniform(-0.08, 0.08, 3).astype(np.float32), 0.0, 1.0)


def _add_spot(acc: np.ndarray, rng: np.random.Generator) -> None:
    """acc (H, W float) üzerine gaussian parıltı veya 4 kollu yıldız ekler (max)."""
    h, w = acc.shape
    sigma = max(0.6, min(h, w) * float(rng.uniform(0.004, 0.02)))
    cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
    peak = float(rng.uniform(0.3, FX_ALPHA_MAX))  # yarı saydam ara değerler
    star = rng.uniform() < 0.5
    r = int(6 * sigma) + 1
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
    if star:
        s_long, s_short = 4.0 * sigma, 0.5 * sigma
        k = np.exp(-(xx**2 / (2 * s_long**2) + yy**2 / (2 * s_short**2)))
        k = np.maximum(k, np.exp(-(xx**2 / (2 * s_short**2) + yy**2 / (2 * s_long**2))))
        k = np.maximum(k, np.exp(-(xx**2 + yy**2) / (2 * sigma**2)))
    else:
        k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    np.maximum(acc[y0:y1, x0:x1], peak * k, out=acc[y0:y1, x0:x1])


def _streaks(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Lens-flare-vari ince, blur'lu ışık çizgileri (H, W float [0,1])."""
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    diag = math.hypot(w, h)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        ang = float(rng.uniform(0, math.pi))
        half = diag * float(rng.uniform(0.2, 0.7)) / 2
        dx, dy = math.cos(ang) * half, math.sin(ang) * half
        val = int(255 * float(rng.uniform(0.15, 0.5)))
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=val, width=int(rng.integers(1, 3)))
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.8, 2.5))))
    return np.asarray(layer, dtype=np.float32) / 255.0


def _render_fx_sample(
    rng: np.random.Generator,
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    bg_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Tek fx örneği: (kompozit RGB uint8, alpha float32 [0,1]).

    alpha = max(fg_alpha, fx_alpha); RGB = fg'nin gerçek bg'ye kompoziti üzerine
    parıltı enerjisinin screen blend'i (bkz. modül docstring'i)."""
    h, w = fg_alpha.shape
    elements: list[tuple[np.ndarray, np.ndarray]] = []  # (alpha haritası, renk)

    # 1) glow halkası — HER örnekte (ara-alpha garantisinin fx ayağı): nesne
    # alpha'sının dışa doğru MaxFilter+GaussianBlur kopyası, 0.15-0.5 çarpanlı.
    pil_a = Image.fromarray((fg_alpha * 255).astype(np.uint8), mode="L")
    ksz = 3 + 2 * int(rng.integers(0, 3))  # 3 / 5 / 7
    radius = max(1.5, min(h, w) * float(rng.uniform(0.02, 0.08)))
    halo = pil_a.filter(ImageFilter.MaxFilter(ksz)).filter(ImageFilter.GaussianBlur(radius))
    halo_a = (np.asarray(halo, dtype=np.float32) / 255.0) * float(rng.uniform(0.15, 0.5))
    elements.append((halo_a, _fx_color(rng)))

    # 2) parçacık parıltıları (gaussian çekirdek / 4 kollu yıldız)
    spots = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(5, 26))):
        _add_spot(spots, rng)
    elements.append((spots, _fx_color(rng)))

    # 3) ışık çizgileri
    if rng.uniform() < 0.7:
        elements.append((_streaks(rng, h, w), _fx_color(rng)))

    fx_alpha = np.zeros((h, w), dtype=np.float32)
    fx_energy = np.zeros((h, w, 3), dtype=np.float32)
    for a_map, color in elements:
        fx_alpha = 1 - (1 - fx_alpha) * (1 - a_map)  # alpha union'ı
        fx_energy = 1 - (1 - fx_energy) * (1 - a_map[..., None] * color[None, None, :])
    fx_alpha = fx_alpha.clip(0.0, FX_ALPHA_MAX)  # yarı saydamlık tavanı (0.15-0.9 bandı)

    bg = _pick_bg(rng, bg_paths, (w, h), flat_prob=0.0).astype(np.float32) / 255.0
    base = fg_rgb.astype(np.float32) / 255.0 * fg_alpha[..., None] + bg * (1 - fg_alpha[..., None])
    out = 1 - (1 - base) * (1 - fx_energy)  # screen: parıltılar additive görünür
    out_alpha = np.maximum(fg_alpha, fx_alpha)
    return (out * 255).round().clip(0, 255).astype(np.uint8), out_alpha.astype(np.float32)


def gen_fx(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    pairs: list[tuple[Path, Path]],
    bg_paths: list[Path],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """Kaynak fg çiftlerine kopyaları eşit dağıtır: indeks = kaynak sırası,
    copy = o kaynağın kopya sırası. (satırlar, üretilen, atlanan) döndürür."""
    if count > 0 and not pairs:
        raise SystemExit("fx için kaynak im/gt çifti bulunamadı (--fg-dirs kökleri im/ + gt/ içermeli)")
    base_copies, rem = divmod(count, len(pairs))
    assert base_copies + (1 if rem else 0) <= 100, (
        f"fx kopya sayısı kaynak başına 100'ü aşamaz (count={count}, kaynak={len(pairs)}): "
        f"2 haneli `_c<NN>` isimlendirmesi taşar."
    )
    new_rows: list[dict] = []
    generated = skipped = 0
    for idx, (im_path, gt_src) in enumerate(pairs):
        n_copies = base_copies + (1 if idx < rem else 0)
        if n_copies == 0:
            continue
        pending: list[str] = []
        for ci in range(n_copies):
            stem = f"fx_{idx:05d}_c{ci:02d}"
            if (out_im_dir / f"{stem}.jpg").exists() and (out_gt_dir / f"{stem}.png").exists():
                skipped += 1
                if stem not in existing_ids:
                    new_rows.append({"id": stem, "category": "fx"})
                continue
            pending.append(stem)
        if not pending:
            continue
        fg_rgb = _load_rgb_capped(im_path)
        fg_alpha = _load_alpha(gt_src, (fg_rgb.shape[1], fg_rgb.shape[0]))
        for stem in pending:
            rng = _item_rng(seed, stem)
            rgb, alpha = _render_fx_sample(rng, fg_rgb, fg_alpha, bg_paths)
            _save_pair(rgb, alpha, out_im_dir / f"{stem}.jpg", out_gt_dir / f"{stem}.png")
            new_rows.append({"id": stem, "category": "fx"})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# illustration kategorisi — hazır ToonOut çiftlerinden kompozit + orijinal
# ==========================================================================
def gen_illustration(
    count: int,
    out_im_dir: Path,
    out_gt_dir: Path,
    pairs: list[tuple[Path, Path]],
    bg_paths: list[Path],
    seed: int,
    existing_ids: set[str],
) -> tuple[list[dict], int, int]:
    """Çift başına 3 kopya: c00/c01 compose+augment (bgr.compositing —
    make_composites kalıbı), c02 orijinal görüntü (compose YOK, yalnız augment;
    make_composites `_o00` mantığı). (satırlar, üretilen, atlanan) döndürür."""
    if count > 0 and not pairs:
        raise SystemExit("illustration için ToonOut im/gt çifti bulunamadı (--toonout-dir/im + /gt)")
    if count > 0 and not bg_paths:
        raise SystemExit("illustration kompoziti için arka plan havuzu gerekli (--bg-dir)")
    n_pairs = min(len(pairs), math.ceil(count / ILLUSTRATION_COPIES))
    new_rows: list[dict] = []
    generated = skipped = emitted = 0
    for idx in range(n_pairs):
        im_path, gt_src = pairs[idx]
        stems: list[tuple[str, int]] = []
        for ci in range(ILLUSTRATION_COPIES):
            if emitted >= count:
                break
            stems.append((f"illustration_{idx:05d}_c{ci:02d}", ci))
            emitted += 1
        pending: list[tuple[str, int]] = []
        for stem, ci in stems:
            if (out_im_dir / f"{stem}.jpg").exists() and (out_gt_dir / f"{stem}.png").exists():
                skipped += 1
                if stem not in existing_ids:
                    new_rows.append({"id": stem, "category": "illustration"})
                continue
            pending.append((stem, ci))
        if not pending:
            continue
        fg_rgb = _load_rgb_capped(im_path)
        alpha = _load_alpha(gt_src, (fg_rgb.shape[1], fg_rgb.shape[0]))
        for stem, ci in pending:
            rng = _item_rng(seed, stem)
            if ci < ILLUSTRATION_COPIES - 1:  # c00/c01: gerçek bg'ye kompozit
                bg_rgb = _load_rgb_capped(bg_paths[int(rng.integers(0, len(bg_paths)))])
                out_rgb, out_alpha = compose(fg_rgb, alpha, bg_rgb, rng)
            else:  # c02: orijinal görüntü (raw ne ise o) — yalnız augment
                out_rgb, out_alpha = fg_rgb, alpha
            out_rgb, out_alpha = augment(out_rgb, out_alpha, rng)
            _save_pair(out_rgb, out_alpha, out_im_dir / f"{stem}.jpg", out_gt_dir / f"{stem}.png")
            new_rows.append({"id": stem, "category": "illustration"})
            generated += 1
    return new_rows, generated, skipped


# ==========================================================================
# Orkestrasyon
# ==========================================================================
def run(
    out_dir: Path,
    bg_dir: Path | None = None,
    fg_dirs: list[Path] | None = None,
    toonout_dir: Path | None = None,
    font_dir: Path | None = None,
    seed: int = 42,
    counts: dict[str, int] | None = None,
    out_manifest: Path | None = None,
    text_canvas: tuple[int, int] = (448, 768),
) -> dict[str, int]:
    """3 kategorinin üreticilerini koşturur; kategori -> yeni üretilen çift
    sayısı döndürür (yalnız >0 olanlar — make_composites.run() ile aynı kalıp).

    `counts`ta olmayan/0 olan kategori tamamen atlanır (girdileri de aranmaz).
    `text_canvas`: text kategorisinin kanvas kenar aralığı (testlerde küçük
    değerlerle hızlı koşu için parametrik)."""
    out_dir = Path(out_dir)
    counts = dict(DEFAULT_COUNTS) if counts is None else counts
    out_im_dir = out_dir / "im"
    out_gt_dir = out_dir / "gt"
    out_im_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(out_manifest) if out_manifest else out_dir / "manifest.jsonl"
    existing_ids = _load_manifest_ids(out_manifest)

    bg_paths = _list_images(Path(bg_dir)) if bg_dir else []
    font_paths = _load_font_paths(Path(font_dir) if font_dir else None)

    all_rows: list[dict] = []
    result: dict[str, int] = {}
    total_skipped = 0

    if counts.get("text", 0) > 0:
        rows, generated, skipped = gen_text(
            counts["text"], out_im_dir, out_gt_dir, bg_paths, font_paths, seed,
            existing_ids, canvas_range=text_canvas,
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["text"] = generated

    if counts.get("fx", 0) > 0:
        pairs: list[tuple[Path, Path]] = []
        for d in fg_dirs or []:
            pairs += _pairs_from_dir(Path(d))
        rows, generated, skipped = gen_fx(
            counts["fx"], out_im_dir, out_gt_dir, pairs, bg_paths, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["fx"] = generated

    if counts.get("illustration", 0) > 0:
        pairs = _pairs_from_dir(Path(toonout_dir)) if toonout_dir else []
        rows, generated, skipped = gen_illustration(
            counts["illustration"], out_im_dir, out_gt_dir, pairs, bg_paths, seed, existing_ids
        )
        all_rows += rows
        total_skipped += skipped
        if generated:
            result["illustration"] = generated

    # manifest'e yalnız yeni id'ler (run içi güvenlik dedup'u dahil)
    fresh: list[dict] = []
    seen = set(existing_ids)
    for row in all_rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    if fresh:
        _append_manifest(out_manifest, fresh)

    print(f"{sum(result.values())} yeni çift yazıldı, {total_skipped} zaten vardı (atlandı)")
    for category, n in sorted(result.items()):
        print(f"{category}: {n}")
    return result


def _parse_counts(spec: str) -> dict[str, int]:
    """'text=4000,fx=3500' -> dict; verilmeyen kategori 0 (atlanır)."""
    counts = {k: 0 for k in DEFAULT_COUNTS}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        if key not in DEFAULT_COUNTS or not value:
            raise SystemExit(
                f"geçersiz --counts parçası: {part!r} (beklenen: {'|'.join(DEFAULT_COUNTS)}=N)"
            )
        counts[key] = int(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="çıktı kökü (im/ + gt/ + manifest.jsonl)")
    parser.add_argument("--bg-dir", default=None, help="gerçek arka plan havuzu (BG-20k)")
    parser.add_argument(
        "--fg-dirs", nargs="*", default=[],
        help="fx kaynak kökleri; her kök im/ + gt/ alt dizinleri içermeli (stem eşleşmeli)",
    )
    parser.add_argument("--toonout-dir", default=None, help="ToonOut kökü (im/ + gt/)")
    parser.add_argument("--font-dir", default=None, help=".ttf/.otf/.ttc font havuzu (yoksa PIL varsayılanı)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts", default="text=4000,fx=3500,illustration=3600")
    parser.add_argument("--out-manifest", default=None, help="varsayılan: <out-dir>/manifest.jsonl")
    args = parser.parse_args()
    run(
        Path(args.out_dir),
        bg_dir=Path(args.bg_dir) if args.bg_dir else None,
        fg_dirs=[Path(d) for d in args.fg_dirs],
        toonout_dir=Path(args.toonout_dir) if args.toonout_dir else None,
        font_dir=Path(args.font_dir) if args.font_dir else None,
        seed=args.seed,
        counts=_parse_counts(args.counts),
        out_manifest=Path(args.out_manifest) if args.out_manifest else None,
    )


if __name__ == "__main__":
    main()
