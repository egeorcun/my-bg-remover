# Faz 1: Eğitimsiz Kalite Katmanı (Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Segmenter çıktısının üstüne eğitim gerektirmeyen kalite katmanı: color decontamination (FBA/ML foreground estimation), düşük güvenli bölge kenar rafinesi (CGM tarzı patch re-inference), `bgr` CLI'ı ve FastAPI servisi; kazanımlar ablation benchmark'ıyla ölçülür.

**Architecture:** `bgr/decontaminate.py` (alpha + RGB → temiz RGBA) ve `bgr/refiner.py` (belirsiz bölgelerde patch bazlı ikinci geçiş) bağımsız, saf fonksiyonlar. `bgr/pipeline.py` bunları `Segmenter` arayüzü arkasında birleştirir (`PipelineSegmenter`), böylece benchmark koşucusu pipeline varyantlarını sıradan model adı gibi koşar (`rmbg-2.0+refine`). CLI ve FastAPI bu pipeline'ı sarar.

**Tech Stack:** Mevcut stack + `pymatting` (MIT; çok seviyeli foreground estimation), `fastapi` + `uvicorn` + `python-multipart` (servis), `httpx` (TestClient).

## Global Constraints

- Proje kökü `/Users/egeo/Documents/Projects/my-bg-remover`; komutlar `uv run ...` ile kökten.
- Alpha sözleşmesi değişmez: float32, (H,W), [0,1], giriş çözünürlüğünde.
- `PipelineSegmenter` `bgr.segmenter.Segmenter` ABC'sini implement eder — benchmark koşucusu değişiklik gerektirmeden pipeline adlarını koşabilmeli.
- Model adı sözdizimi: `<base>[+refine]` (örn. `rmbg-2.0+refine`); decontamination alpha'yı DEĞİŞTİRMEZ (yalnız RGB'yi), bu yüzden alpha-metrik ablation'ında ayrı varyant değildir — RGBA çıktı üzerinden galeriyle değerlendirilir.
- Kullanıcı otonom mod istedi: hiçbir adımda kullanıcıya soru sorulmaz; kararlar rapor edilir.
- Yeni bağımlılıklar pyproject'e eklenir (`uv add`); testler hermetik (gerçek model yüklemeyen testler `.env.local`/ağ istemez).
- Commit mesajları Türkçe, conventional commits; sonuna: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Decontaminator (color decontamination)

**Files:**
- Create: `bgr/decontaminate.py`
- Modify: `pyproject.toml` (`uv add pymatting`)
- Test: `tests/test_decontaminate.py`

**Interfaces:**
- Produces: `bgr.decontaminate.decontaminate(image: PIL.Image.Image, alpha: np.ndarray) -> PIL.Image.Image` — RGBA döner; RGB kanalları pymatting `estimate_foreground_ml` ile tahmin edilmiş saf özne rengi, A kanalı verilen alpha (uint8'e np.round ile). Alpha shape'i görselle uyuşmazsa `ValueError`.
- Consumes: alpha sözleşmesi.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_decontaminate.py`:

```python
import numpy as np
import pytest
from PIL import Image

from bgr.decontaminate import decontaminate


@pytest.fixture
def red_on_green():
    """Kırmızı kare, yeşil zemin, kenarda 3px yumuşak (karışmış) geçiş."""
    w = h = 64
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :] = (0.0, 0.8, 0.0)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    from scipy import ndimage
    alpha = ndimage.gaussian_filter(alpha, 1.5).clip(0, 1).astype(np.float32)
    comp = alpha[..., None] * np.array([0.9, 0.1, 0.1]) + (1 - alpha[..., None]) * img
    pil = Image.fromarray((comp * 255).astype(np.uint8))
    return pil, alpha


def test_returns_rgba_same_size(red_on_green):
    pil, alpha = red_on_green
    out = decontaminate(pil, alpha)
    assert out.mode == "RGBA"
    assert out.size == pil.size


def test_edge_pixels_lose_green_spill(red_on_green):
    pil, alpha = red_on_green
    out = np.asarray(decontaminate(pil, alpha), dtype=np.float64) / 255.0
    band = (alpha > 0.2) & (alpha < 0.8)
    naive_rgb = np.asarray(pil, dtype=np.float64) / 255.0
    # kenar bandında yeşil kanal, naive kompozite göre belirgin azalmalı
    assert out[..., 1][band].mean() < naive_rgb[..., 1][band].mean() - 0.05
    # opak iç bölge değişmemeli (kırmızı kalmalı)
    core = alpha > 0.99
    assert abs(out[..., 0][core].mean() - naive_rgb[..., 0][core].mean()) < 0.05


def test_shape_mismatch_raises(red_on_green):
    pil, _ = red_on_green
    with pytest.raises(ValueError):
        decontaminate(pil, np.zeros((8, 8), dtype=np.float32))
```

- [ ] **Step 2: FAIL doğrula** — Run: `uv run pytest tests/test_decontaminate.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: Bağımlılığı ekle ve implement et**

Run: `uv add pymatting`

`bgr/decontaminate.py`:

```python
"""Kenar renk sızması temizliği (color decontamination).

Kenar pikselleri eski arka planla karışıktır; pymatting'in çok seviyeli
foreground estimation'ı her piksel için saf özne rengini çözer. Alpha
değişmez — yalnız RGB kanalları temizlenir.
"""
import numpy as np
from PIL import Image
from pymatting import estimate_foreground_ml


def decontaminate(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if alpha.shape != rgb.shape[:2]:
        raise ValueError(f"alpha shape {alpha.shape} != image {rgb.shape[:2]}")
    fg = estimate_foreground_ml(rgb, alpha.astype(np.float64))
    out = np.dstack([np.clip(fg, 0, 1), alpha.clip(0, 1)])
    return Image.fromarray(np.round(out * 255).astype(np.uint8), mode="RGBA")
```

- [ ] **Step 4: PASS doğrula** — Run: `uv run pytest tests/test_decontaminate.py -v` → 3 passed
- [ ] **Step 5: Commit** — `feat: color decontamination (pymatting foreground estimation)`

---

### Task 2: Edge Refiner (belirsiz bölge patch rafinesi)

**Files:**
- Create: `bgr/refiner.py`
- Test: `tests/test_refiner.py`

**Interfaces:**
- Produces: `bgr.refiner.refine_alpha(segmenter: Segmenter, image: PIL.Image.Image, alpha: np.ndarray, low: float = 0.05, high: float = 0.95, min_region: int = 256, context: float = 0.35, max_patches: int = 6) -> np.ndarray` — belirsiz bölgeleri (low<alpha<high) bileşenlere ayırır, en büyük `max_patches` bölgenin bbox'ını `context` oranında genişletip kırpar, kırpıntıyı segmenter'a yeniden verir (kırpıntı küçük olduğu için model efektif daha yüksek çözünürlükte görür), sonucu YALNIZ belirsiz bantta feather'lı harmanlar. Alpha sözleşmesini korur.
- Consumes: `Segmenter.predict_alpha`.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_refiner.py` (sahte segmenter — model yüklemez):

```python
import numpy as np
from PIL import Image

from bgr.refiner import refine_alpha


class SharpFakeSeg:
    """Kırpıntıda 'keskin' alpha döndürür: sol yarı 1, sağ yarı 0."""
    name = "sharp-fake"

    def __init__(self):
        self.calls = []

    def predict_alpha(self, image):
        w, h = image.size
        self.calls.append((w, h))
        a = np.zeros((h, w), dtype=np.float32)
        a[:, : w // 2] = 1.0
        return a


def _blurry_alpha(h=128, w=128):
    a = np.zeros((h, w), dtype=np.float32)
    a[:, : w // 2] = 1.0
    from scipy import ndimage
    return ndimage.gaussian_filter(a, 6).clip(0, 1).astype(np.float32)


def test_confident_alpha_untouched():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (64, 64))
    a = np.ones((64, 64), dtype=np.float32)  # tamamen emin
    out = refine_alpha(seg, img, a)
    assert seg.calls == []  # hiç patch koşmadı
    np.testing.assert_array_equal(out, a)


def test_uncertain_band_gets_sharper():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (128, 128))
    blurry = _blurry_alpha()
    out = refine_alpha(seg, img, blurry)
    assert len(seg.calls) >= 1
    band = (blurry > 0.05) & (blurry < 0.95)
    # rafine sonrası bantta ara-değerli piksel sayısı azalmalı (keskinleşme)
    mid_before = ((blurry > 0.2) & (blurry < 0.8) & band).sum()
    mid_after = ((out > 0.2) & (out < 0.8) & band).sum()
    assert mid_after < mid_before
    # emin bölgeler değişmedi
    np.testing.assert_allclose(out[~band], blurry[~band], atol=1e-6)


def test_contract_preserved():
    seg = SharpFakeSeg()
    img = Image.new("RGB", (96, 80))
    out = refine_alpha(seg, img, _blurry_alpha(80, 96))
    assert out.dtype == np.float32 and out.shape == (80, 96)
    assert out.min() >= 0 and out.max() <= 1
```

- [ ] **Step 2: FAIL doğrula** — `uv run pytest tests/test_refiner.py -v` → `ModuleNotFoundError`

- [ ] **Step 3: bgr/refiner.py implement et**

```python
"""CGM tarzı kenar rafinesi: modelin emin olamadığı bölgeleri kırpıp
aynı modele yüksek efektif çözünürlükte yeniden sorar, sonucu yalnız
belirsiz bantta feather'lı harmanlar.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

from bgr.segmenter import Segmenter


def _regions(band: np.ndarray, min_region: int, max_patches: int) -> list[tuple[int, int, int, int]]:
    labels, num = ndimage.label(ndimage.binary_dilation(band, iterations=4))
    if num == 0:
        return []
    sizes = ndimage.sum(band, labels, range(1, num + 1))
    order = np.argsort(sizes)[::-1]
    boxes = ndimage.find_objects(labels)
    out = []
    for i in order[:max_patches]:
        if sizes[i] < min_region:
            break
        sl = boxes[i]
        out.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
    return out


def refine_alpha(
    segmenter: Segmenter,
    image: Image.Image,
    alpha: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    min_region: int = 256,
    context: float = 0.35,
    max_patches: int = 6,
) -> np.ndarray:
    h, w = alpha.shape
    band = (alpha > low) & (alpha < high)
    out = alpha.copy()
    for y0, y1, x0, x1 in _regions(band, min_region, max_patches):
        cy, cx = int((y1 - y0) * context), int((x1 - x0) * context)
        yy0, yy1 = max(0, y0 - cy), min(h, y1 + cy)
        xx0, xx1 = max(0, x0 - cx), min(w, x1 + cx)
        crop = image.convert("RGB").crop((xx0, yy0, xx1, yy1))
        refined = segmenter.predict_alpha(crop)
        # feather: bant maskesini yumuşat, yalnız bant içinde harmanla
        local_band = band[yy0:yy1, xx0:xx1].astype(np.float32)
        weight = ndimage.gaussian_filter(local_band, 2).clip(0, 1)
        weight[local_band == 0] = 0.0
        region = out[yy0:yy1, xx0:xx1]
        out[yy0:yy1, xx0:xx1] = weight * refined + (1 - weight) * region
    return out.clip(0, 1).astype(np.float32)
```

- [ ] **Step 4: PASS doğrula** — `uv run pytest tests/test_refiner.py -v` → 3 passed
- [ ] **Step 5: Commit** — `feat: CGM tarzı kenar rafinesi (patch re-inference)`

---

### Task 3: PipelineSegmenter + registry '+refine' sözdizimi

**Files:**
- Create: `bgr/pipeline.py`
- Modify: `bgr/registry.py` (get_segmenter '+refine' parse eder)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `bgr.pipeline.PipelineSegmenter(base: Segmenter, refine: bool = False)` — `Segmenter` implement eder; `predict_alpha` = base + (opsiyonel) `refine_alpha`; `name` = `base.name` + (`"+refine"` if refine).
  - `bgr.pipeline.PipelineSegmenter.process(image: PIL.Image, decontaminate: bool = True) -> PIL.Image (RGBA)` — tam uçtan uca çıktı (CLI/servis bunu çağırır).
  - `bgr.registry.get_segmenter("rmbg-2.0+refine")` → refine'lı PipelineSegmenter (benchmark için).
- Consumes: Task 1-2 fonksiyonları, mevcut registry.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_pipeline.py`:

```python
import numpy as np
import pytest
from PIL import Image

from bgr.pipeline import PipelineSegmenter


class FlatFakeSeg:
    name = "flat-fake"

    def predict_alpha(self, image):
        w, h = image.size
        a = np.full((h, w), 0.5, dtype=np.float32)
        a[: h // 4] = 0.0
        a[-h // 4 :] = 1.0
        return a


def test_name_reflects_refine_flag():
    assert PipelineSegmenter(FlatFakeSeg()).name == "flat-fake"
    assert PipelineSegmenter(FlatFakeSeg(), refine=True).name == "flat-fake+refine"


def test_predict_alpha_contract():
    p = PipelineSegmenter(FlatFakeSeg())
    a = p.predict_alpha(Image.new("RGB", (32, 40)))
    assert a.dtype == np.float32 and a.shape == (40, 32)


def test_process_returns_rgba():
    p = PipelineSegmenter(FlatFakeSeg())
    out = p.process(Image.new("RGB", (32, 32), (200, 30, 30)), decontaminate=True)
    assert out.mode == "RGBA" and out.size == (32, 32)


def test_registry_parses_refine_suffix():
    from unittest.mock import patch
    from bgr.registry import get_segmenter
    with patch("bgr.registry.BiRefNetSegmenter") as m:
        m.return_value.name = "rmbg-2.0"
        seg = get_segmenter("rmbg-2.0+refine")
    assert seg.name == "rmbg-2.0+refine"


def test_registry_unknown_base_still_raises():
    from bgr.registry import get_segmenter
    with pytest.raises(KeyError):
        get_segmenter("yok+refine")
```

- [ ] **Step 2: FAIL doğrula** — `uv run pytest tests/test_pipeline.py -v`

- [ ] **Step 3: Implement et**

`bgr/pipeline.py`:

```python
"""Segmenter + refiner + decontaminator'ı tek arayüzde birleştirir."""
import numpy as np
from PIL import Image

from bgr.decontaminate import decontaminate as _decon
from bgr.refiner import refine_alpha
from bgr.segmenter import Segmenter


class PipelineSegmenter(Segmenter):
    def __init__(self, base: Segmenter, refine: bool = False):
        self.base = base
        self.refine = refine
        self.name = base.name + ("+refine" if refine else "")

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        alpha = self.base.predict_alpha(image)
        if self.refine:
            alpha = refine_alpha(self.base, image, alpha)
        return alpha

    def process(self, image: Image.Image, decontaminate: bool = True) -> Image.Image:
        alpha = self.predict_alpha(image)
        if decontaminate:
            return _decon(image, alpha)
        rgba = image.convert("RGB").copy()
        rgba.putalpha(Image.fromarray(np.round(alpha * 255).astype(np.uint8)))
        return rgba
```

`bgr/registry.py` — `get_segmenter`'ı şu hale getir (MODEL_SPECS ve _GATED_HELP aynı kalır):

```python
def get_segmenter(name: str) -> Segmenter:
    from bgr.pipeline import PipelineSegmenter

    base_name, _, suffix = name.partition("+")
    spec = MODEL_SPECS[base_name]  # bilinmeyen ad -> KeyError
    try:
        base = BiRefNetSegmenter(
            model_id=spec["model_id"], input_size=spec["input_size"], name=base_name
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e):
            raise RuntimeError(_GATED_HELP.format(model_id=spec["model_id"])) from e
        raise
    if suffix == "refine":
        return PipelineSegmenter(base, refine=True)
    if suffix:
        raise KeyError(f"bilinmeyen varyant: +{suffix}")
    return base
```

- [ ] **Step 4: PASS doğrula** — `uv run pytest tests/test_pipeline.py tests/test_registry.py -v -m "not slow"` (mevcut registry testleri de geçmeli)
- [ ] **Step 5: Commit** — `feat: PipelineSegmenter ve registry +refine varyantı`

---

### Task 4: `bgr` CLI

**Files:**
- Create: `bgr/cli.py`
- Modify: `pyproject.toml` (`[project.scripts] bgr = "bgr.cli:main"`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `bgr remove <girdi> -o <çıktı.png> [--model rmbg-2.0] [--refine] [--no-decontaminate]` — çıktı RGBA PNG. `uv run bgr remove ...` ile çalışır. Model yükleme yalnız `main()` içinde (import'ta değil).
- Consumes: `get_segmenter`, `PipelineSegmenter.process`.

- [ ] **Step 1: Failing test yaz**

`tests/test_cli.py`:

```python
import numpy as np
from unittest.mock import patch
from PIL import Image

from bgr.cli import main


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def test_remove_writes_rgba(tmp_path):
    src = tmp_path / "in.jpg"
    Image.new("RGB", (16, 16), (10, 120, 200)).save(src)
    dst = tmp_path / "out.png"
    with patch("bgr.cli.get_segmenter", return_value=FakeSeg()):
        main(["remove", str(src), "-o", str(dst), "--no-decontaminate"])
    out = Image.open(dst)
    assert out.mode == "RGBA" and out.size == (16, 16)
```

- [ ] **Step 2: FAIL doğrula** — `uv run pytest tests/test_cli.py -v`

- [ ] **Step 3: Implement et**

`bgr/cli.py`:

```python
"""bgr CLI: uv run bgr remove girdi.jpg -o cikti.png"""
import argparse

from PIL import Image

from bgr.pipeline import PipelineSegmenter
from bgr.registry import get_segmenter


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bgr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rm = sub.add_parser("remove", help="arka planı sil")
    rm.add_argument("input")
    rm.add_argument("-o", "--output", required=True)
    rm.add_argument("--model", default="rmbg-2.0")
    rm.add_argument("--refine", action="store_true")
    rm.add_argument("--no-decontaminate", action="store_true")
    a = ap.parse_args(argv)

    seg = get_segmenter(a.model)
    pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
    if a.refine and not pipe.refine:
        pipe = PipelineSegmenter(pipe.base, refine=True)
    out = pipe.process(Image.open(a.input), decontaminate=not a.no_decontaminate)
    out.save(a.output)
    print(f"kaydedildi: {a.output}")


if __name__ == "__main__":
    main()
```

`pyproject.toml`'a ekle:

```toml
[project.scripts]
bgr = "bgr.cli:main"
```

Sonra `uv sync` (script'in kurulması için).

- [ ] **Step 4: PASS doğrula** — `uv run pytest tests/test_cli.py -v`; canlı duman testi: `set -a && source .env.local && set +a && uv run bgr remove data/testset/images/trans460_pexels-pixabay-34487.jpg -o /tmp/bgr_cli_test.png` → RGBA PNG oluşmalı (gerçek model, ~1 dk).
- [ ] **Step 5: Commit** — `feat: bgr CLI (remove komutu)`

---

### Task 5: FastAPI servisi

**Files:**
- Create: `serving/app.py`
- Modify: `pyproject.toml` (`uv add fastapi uvicorn python-multipart httpx`)
- Test: `tests/test_serving.py`

**Interfaces:**
- Produces: `POST /remove` (multipart `file` + query `model=rmbg-2.0&refine=false&decontaminate=true`) → `image/png` RGBA. `GET /health` → `{"status":"ok","models":[...]}`. Model lazy-load + süreç içi cache (`_SEGMENTERS: dict`). Çalıştırma: `uv run uvicorn serving.app:app --port 8756`.
- Consumes: `get_segmenter`, `PipelineSegmenter`.

- [ ] **Step 1: Failing testleri yaz**

`tests/test_serving.py`:

```python
import io

import numpy as np
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image


class FakeSeg:
    name = "fake"

    def predict_alpha(self, image):
        w, h = image.size
        return np.ones((h, w), dtype=np.float32)


def _client():
    from serving.app import app
    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_remove_returns_png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (50, 60, 70)).save(buf, format="JPEG")
    buf.seek(0)
    with patch("serving.app._load_segmenter", return_value=FakeSeg()):
        r = _client().post(
            "/remove?decontaminate=false",
            files={"file": ("in.jpg", buf, "image/jpeg")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    out = Image.open(io.BytesIO(r.content))
    assert out.mode == "RGBA" and out.size == (16, 16)


def test_unknown_model_400():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    buf.seek(0)
    r = _client().post("/remove?model=yok", files={"file": ("x.png", buf, "image/png")})
    assert r.status_code == 400
```

- [ ] **Step 2: FAIL doğrula** — `uv run pytest tests/test_serving.py -v`

- [ ] **Step 3: Bağımlılıklar + implement**

Run: `uv add fastapi uvicorn python-multipart httpx`

`serving/app.py`:

```python
"""Lokal bg-remove servisi: uv run uvicorn serving.app:app --port 8756"""
import io

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from bgr.pipeline import PipelineSegmenter
from bgr.registry import MODEL_SPECS, get_segmenter

app = FastAPI(title="my-bg-remover")
_SEGMENTERS: dict[str, object] = {}


def _load_segmenter(name: str):
    if name not in _SEGMENTERS:
        _SEGMENTERS[name] = get_segmenter(name)
    return _SEGMENTERS[name]


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_SPECS)}


@app.post("/remove")
async def remove(
    file: UploadFile,
    model: str = "rmbg-2.0",
    refine: bool = False,
    decontaminate: bool = True,
):
    try:
        seg = _load_segmenter(model + ("+refine" if refine else ""))
    except KeyError:
        raise HTTPException(400, f"bilinmeyen model: {model}")
    img = Image.open(io.BytesIO(await file.read()))
    pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
    out = pipe.process(img, decontaminate=decontaminate)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
```

- [ ] **Step 4: PASS doğrula** — `uv run pytest tests/test_serving.py -v` → 3 passed
- [ ] **Step 5: Uçtan uca duman testi** — arka planda `set -a && source .env.local && set +a && uv run uvicorn serving.app:app --port 8756` başlat; `curl -s -F "file=@data/testset/images/p3m_p_<herhangi>.jpg" "http://127.0.0.1:8756/remove?decontaminate=false" -o /tmp/serve_test.png` → RGBA PNG; sunucuyu kapat.
- [ ] **Step 6: Commit** — `feat: FastAPI bg-remove servisi`

---

### Task 6: RGBA çıktılar + ablation koşusu + rapor

**Files:**
- Modify: `benchmark/run.py` (`--rgba` bayrağı: her satır için `<out>/<model>/rgba/<id>.png` decontaminated RGBA üretir, idempotent), `benchmark/gallery.py` (model hücresi: `rgba/` varsa onu, yoksa composites/ mantığını kullanır)
- Create: `docs/reports/2026-07-faz1-ablation.md`
- Test: `tests/test_run.py` (+1 rgba testi), `tests/test_gallery.py` (+1 tercih testi)

**Interfaces:**
- Produces: ablation sonuç raporu; galeri artık decontaminated çıktı gösterir (Ideogram'la adil karşılaştırma).
- Consumes: tüm önceki task'lar.

- [ ] **Step 1: run.py'ye --rgba ekle (TDD)** — test: FakeSeg ile `run_benchmark(["fake"], ..., rgba=True)` → `out/fake/rgba/a.png` var ve RGBA; ikinci çağrı idempotent. Implementasyon: satır döngüsünde `if rgba:` bloğu — `PipelineSegmenter(seg).process(img)` yerine kaydedilmiş alpha PNG'yi yükleyip `decontaminate(img, alpha)` çağır (model tekrar koşmasın); CLI'ya `--rgba` bayrağı.
- [ ] **Step 2: gallery.py tercihi (TDD)** — test: `results/m1/rgba/a.png` varsa galeri onu embed eder (composites üretmez); yoksa mevcut composite davranışı.
- [ ] **Step 3: Ablation koşusu** — `set -a && source .env.local && set +a && uv run python -m benchmark.run --models rmbg-2.0+refine --manifest data/testset/manifest.jsonl --out results/baseline` (metrics.json merge sayesinde önceki modellerin yanına eklenir; ~130 görsel × refine patch'leri, MPS'te uzun sürebilir — arka planda koş, logla). Ardından `--models rmbg-2.0 --rgba` ile RGBA üretimi (model koşmaz, alpha cache'ten).
- [ ] **Step 4: Galeriyi yenile** — gallery komutunu `--models birefnet-hr,rmbg-2.0,rmbg-2.0+refine` ile koş.
- [ ] **Step 5: Ablation raporu yaz** — `docs/reports/2026-07-faz1-ablation.md`: kategori bazlı metrik tablosu (rmbg-2.0 vs rmbg-2.0+refine, özellikle transparent/thin'de Grad ve MAE değişimi), refiner'ın kazandırdığı/kaybettirdiği örnek id'leri, decontamination'ın galeriden 2-3 örnekle nitel değerlendirmesi, Faz 2'ye öneriler.
- [ ] **Step 6: Tam suite + commit** — `uv run pytest -v -m "not slow"` → hepsi yeşil; commit `feat: ablation koşusu ve RGBA galeri entegrasyonu` + `docs: Faz 1 ablation raporu`.

---

## Self-Review Notları

- **Spec kapsaması:** Faz 1 = Decontaminator ✓ (T1), Edge Refiner bayraklı ✓ (T2+T3), CLI+FastAPI ✓ (T4+T5), modül kazançları ablation'la ✓ (T6). Router: gap analizi gerekçe sunmadığı için bilinçli olarak V1'den ertelendi (spec'in Faz 1 maddesindeki Router, baseline bulgusuyla geçersizleşti — raporda belgeli).
- **Tip tutarlılığı:** `PipelineSegmenter` `Segmenter` ABC'sine uyar; `get_segmenter` dönüş tipi değişmedi; alpha sözleşmesi her modülde aynı.
- **Bilinen risk:** refine koşusu MPS'te yavaş olabilir (patch başına tam forward); `max_patches=6` sınırı ve arka plan koşusu bunu yönetir.
