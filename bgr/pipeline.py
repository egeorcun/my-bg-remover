"""Combines segmenter + refiner + decontaminator behind a single interface."""
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
