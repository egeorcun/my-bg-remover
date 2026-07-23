"""Segmenter construction by name. Adding a new model = adding a row to MODEL_SPECS."""
from bgr.segmenter import BiRefNetSegmenter, LocalBiRefNetSegmenter, Segmenter

MODEL_SPECS: dict[str, dict] = {
    "lucida": {"model_id": "egeorcun/lucida", "input_size": 1024},
    "birefnet-hr": {"model_id": "ZhengPeng7/BiRefNet_HR", "input_size": 2048},
    "inspyrenet": {"engine": "inspyrenet"},
    "rmbg-2.0": {"model_id": "briaai/RMBG-2.0", "input_size": 1024},
    "bgr-v1": {
        "ckpt": "data/checkpoints/epoch_1.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "bgr-v2": {
        "ckpt": "data/checkpoints/epoch_2.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "bgr-v3": {
        "ckpt": "data/checkpoints/epoch_3.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "bgr-v4": {
        "ckpt": "data/checkpoints/epoch_4.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v5": {
        "ckpt": "data/checkpoints/epoch_5.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v6": {
        "ckpt": "data/checkpoints/epoch_6.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v7": {
        "ckpt": "data/checkpoints/epoch_7.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    # v8 = the discarded alpha^2 epoch (kept for comparisons; see the
    # SAMPLER_PRESET_V9 docstring in training/train_colab_lib.py).
    "lucida-v8": {
        "ckpt": "data/checkpoints/epoch_8_v8bug.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v9": {
        "ckpt": "data/checkpoints/epoch_8_v9.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v11probe": {
        "ckpt": "data/checkpoints/epoch_10.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v12": {
        "ckpt": "data/checkpoints/epoch_12.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v11": {
        "ckpt": "data/checkpoints/epoch_11.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
    "lucida-v10": {
        "ckpt": "data/checkpoints/epoch_9.pth",
        "arch_id": "ZhengPeng7/BiRefNet_HR",
        "input_size": 1024,
    },
}

_GATED_HELP = (
    "{model_id} is a gated model. Do the following:\n"
    "1) accept the license at https://huggingface.co/{model_id}\n"
    "2) log in with `huggingface-cli login`"
)


def get_segmenter(name: str) -> Segmenter:
    from bgr.pipeline import PipelineSegmenter

    base_name, _, suffix = name.partition("+")
    if suffix and suffix != "refine":
        raise KeyError(f"unknown variant: +{suffix}")
    spec = MODEL_SPECS[base_name]  # unknown name -> KeyError
    model_id = spec.get("model_id", spec.get("arch_id"))
    try:
        if spec.get("engine") == "inspyrenet":
            from bgr.segmenter import InSPyReNetSegmenter

            base = InSPyReNetSegmenter(name=base_name)
        elif "ckpt" in spec:
            base = LocalBiRefNetSegmenter(
                ckpt_path=spec["ckpt"],
                input_size=spec["input_size"],
                name=base_name,
                arch_id=spec["arch_id"],
            )
        else:
            base = BiRefNetSegmenter(
                model_id=spec["model_id"], input_size=spec["input_size"], name=base_name
            )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e):
            raise RuntimeError(_GATED_HELP.format(model_id=model_id)) from e
        raise
    if suffix == "refine":
        return PipelineSegmenter(base, refine=True)
    return base
