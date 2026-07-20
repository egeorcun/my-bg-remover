"""Single-image inference: uv run python scripts/infer.py input.jpg output.png"""
import sys

import numpy as np
from PIL import Image

from bgr.segmenter import BiRefNetSegmenter


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    seg = BiRefNetSegmenter(
        model_id="ZhengPeng7/BiRefNet_HR", input_size=2048, name="birefnet-hr"
    )
    img = Image.open(src)
    alpha = seg.predict_alpha(img)
    rgba = img.convert("RGB").copy()
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    rgba.save(dst)
    print(f"saved: {dst}")


if __name__ == "__main__":
    main()
