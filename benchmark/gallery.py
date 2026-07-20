"""Side-by-side HTML gallery: original | model alpha composites | Ideogram."""
import argparse
import html
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

from benchmark.testset import load_manifest

_CSS = """
body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
.row{display:flex;gap:8px;margin:8px 0;align-items:flex-start}
.cell{text-align:center;font-size:11px}
img{max-width:260px;max-height:260px;
 background:repeating-conic-gradient(#666 0% 25%,#999 0% 50%) 0/16px 16px}
h2{border-bottom:1px solid #444;padding-bottom:4px}
"""


def _img_cell(out_dir: Path, src: Path, label: str) -> str:
    rel = os.path.relpath(src, out_dir)
    return f'<div class="cell"><img src="{html.escape(rel)}"><br>{html.escape(label)}</div>'


def _build_composite(image_path: Path, mask_path: Path, composite_path: Path) -> Path:
    """Original image (RGB) + model mask (alpha channel) -> RGBA composite, cached."""
    if composite_path.exists():
        return composite_path
    rgb = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size)
    rgba = rgb.copy()
    rgba.putalpha(mask)
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(composite_path)
    return composite_path


def build_gallery(manifest_path: str, results_dir: str, models: list[str], out_html: str) -> None:
    rows = load_manifest(manifest_path)
    results = Path(results_dir)
    out = Path(out_html)
    out_dir = out.parent
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row["category"]].append(row)

    parts = [
        '<meta charset="utf-8">',
        f"<style>{_CSS}</style><h1>bg-remover benchmark</h1>",
    ]
    for cat in sorted(by_cat):
        parts.append(f"<h2>{html.escape(cat)}</h2>")
        for row in by_cat[cat]:
            cells = [_img_cell(out_dir, Path(row["image"]).resolve(), "original")]
            if row["gt_alpha"]:
                cells.append(_img_cell(out_dir, Path(row["gt_alpha"]).resolve(), "GT"))
            for m in models:
                mask_path = results / m / f"{row['id']}.png"
                if mask_path.exists():
                    rgba_path = results / m / "rgba" / f"{row['id']}.png"
                    if rgba_path.exists():
                        cells.append(_img_cell(out_dir, rgba_path.resolve(), m))
                    else:
                        composite_path = results / m / "composites" / f"{row['id']}.png"
                        _build_composite(Path(row["image"]).resolve(), mask_path.resolve(), composite_path)
                        cells.append(_img_cell(out_dir, composite_path.resolve(), m))
            ideo = results.parent / "ideogram" / f"{row['id']}.png"
            if ideo.exists():
                cells.append(_img_cell(out_dir, ideo.resolve(), "ideogram"))
            parts.append(f'<div class="row" id="{html.escape(row["id"])}">{"".join(cells)}</div>')
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_gallery(a.manifest, a.results, a.models.split(","), a.out)
    print(f"gallery: {a.out}")


if __name__ == "__main__":
    main()
