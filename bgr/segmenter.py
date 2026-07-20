"""Segmenter interface and the BiRefNet-family implementation.

Contract: predict_alpha(PIL.Image) -> np.float32 (H, W), [0, 1],
at the same resolution as the input image.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Segmenter(ABC):
    name: str

    @abstractmethod
    def predict_alpha(self, image: Image.Image) -> np.ndarray: ...


class BiRefNetSegmenter(Segmenter):
    """All HF models built on the BiRefNet architecture (BiRefNet_HR, RMBG-2.0...)."""

    def __init__(self, model_id: str, input_size: int, name: str):
        from transformers import AutoModelForImageSegmentation

        self.name = name
        self.input_size = input_size
        self.device = get_device()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float32
        )
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        inp = self.transform(rgb).unsqueeze(0).to(self.device)
        preds = self.model(inp)[-1].sigmoid().cpu()
        alpha = transforms.functional.resize(preds[0], rgb.size[::-1])[0]
        return alpha.clamp(0, 1).numpy().astype(np.float32)


class LocalBiRefNetSegmenter(BiRefNetSegmenter):
    """BiRefNet loaded with our own fine-tuned checkpoint.

    The architecture is built from `arch_id` (HF, `trust_remote_code=True`) —
    that is only a starting point to fetch the right class/code; the weights
    are then COMPLETELY overridden with our own checkpoint at `ckpt_path`.

    Checkpoint format: `save_and_sync_checkpoint` in
    `training/train_colab.ipynb` — `torch.save({"model": state_dict,
    "optimizer": ..., "lr_scheduler": ..., "epoch": int}, path)`. If the
    `state_dict` was trained under `torch.compile` its keys may carry the
    `_orig_mod.` prefix (same behavior as the official BiRefNet `train.py`:
    the prefix is saved WITHOUT being stripped and cleaned up at load time —
    see `utils.check_state_dict`).
    """

    def __init__(
        self,
        ckpt_path: str,
        input_size: int,
        name: str,
        arch_id: str = "ZhengPeng7/BiRefNet_HR",
    ):
        super().__init__(model_id=arch_id, input_size=input_size, name=name)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" not in payload:
            raise KeyError(
                f"checkpoint has no 'model' key ({ckpt_path}); "
                f"keys found: {sorted(payload.keys())}"
            )
        state_dict = payload["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {
                k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
            }
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            diff = self.model.load_state_dict(state_dict, strict=False)
            raise RuntimeError(
                "strict load_state_dict failed: the checkpoint does not FULLY "
                "match the architecture (NO silent partial load was performed).\n"
                f"  missing keys ({len(diff.missing_keys)}): {diff.missing_keys}\n"
                f"  unexpected keys ({len(diff.unexpected_keys)}): {diff.unexpected_keys}\n"
                f"  checkpoint: {ckpt_path}, arch: {arch_id}"
            ) from e
        self.model.to(self.device).eval()


class InSPyReNetSegmenter(Segmenter):
    """InSPyReNet (ACCV 2022) — via the `transparent-background` package.

    The package does its own pre/post-processing; `process(..., type="map")`
    returns a grayscale alpha map at the same size as the input. The alpha
    contract matches the other segmenters: float32, (H, W), [0, 1], at input
    resolution. Device: pinned to CPU because the package does not officially
    support MPS (slow but deterministic; acceptable since the benchmark is a
    one-off run)."""

    def __init__(self, name: str = "inspyrenet"):
        from transparent_background import Remover

        self.name = name
        self.remover = Remover(mode="base", device="cpu")

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        out = self.remover.process(image.convert("RGB"), type="map")
        alpha = np.asarray(out.convert("L"), dtype=np.float32) / 255.0
        return alpha
