"""Local bg-remove service: uv run uvicorn serving.app:app --port 8756"""
import io
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from bgr.pipeline import PipelineSegmenter
from bgr.registry import MODEL_SPECS, get_segmenter

app = FastAPI(title="my-bg-remover")
_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_MODEL = "lucida"
_SEGMENTERS: dict[str, object] = {}
_SEGMENTERS_LOCK = threading.Lock()


def _load_segmenter(name: str):
    if name not in _SEGMENTERS:
        with _SEGMENTERS_LOCK:
            if name not in _SEGMENTERS:
                _SEGMENTERS[name] = get_segmenter(name)
    return _SEGMENTERS[name]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_SPECS)}


@app.get("/models")
def models():
    return {"models": sorted(MODEL_SPECS), "default": _DEFAULT_MODEL}


@app.post("/remove")
def remove(
    file: UploadFile,
    model: str = "rmbg-2.0",
    refine: bool = False,
    decontaminate: bool = True,
):
    try:
        seg = _load_segmenter(model + ("+refine" if refine else ""))
    except KeyError:
        raise HTTPException(400, f"unknown model: {model}")
    try:
        img = Image.open(io.BytesIO(file.file.read()))
        img.load()
    except Exception:
        raise HTTPException(400, "invalid image file")
    pipe = seg if isinstance(seg, PipelineSegmenter) else PipelineSegmenter(seg)
    out = pipe.process(img, decontaminate=decontaminate)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
