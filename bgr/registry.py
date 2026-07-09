"""İsimle segmenter üretimi. Yeni model eklemek = MODEL_SPECS'e satır eklemek."""
from bgr.segmenter import BiRefNetSegmenter, Segmenter

MODEL_SPECS: dict[str, dict] = {
    "birefnet-hr": {"model_id": "ZhengPeng7/BiRefNet_HR", "input_size": 2048},
    "rmbg-2.0": {"model_id": "briaai/RMBG-2.0", "input_size": 1024},
}

_GATED_HELP = (
    "{model_id} gated bir model. Şunları yap:\n"
    "1) https://huggingface.co/{model_id} adresinde lisansı onayla\n"
    "2) `huggingface-cli login` ile giriş yap"
)


def get_segmenter(name: str) -> Segmenter:
    spec = MODEL_SPECS[name]  # bilinmeyen ad -> KeyError
    try:
        return BiRefNetSegmenter(
            model_id=spec["model_id"], input_size=spec["input_size"], name=name
        )
    except Exception as e:  # GatedRepoError / 401
        if "gated" in str(e).lower() or "401" in str(e):
            raise RuntimeError(_GATED_HELP.format(model_id=spec["model_id"])) from e
        raise
