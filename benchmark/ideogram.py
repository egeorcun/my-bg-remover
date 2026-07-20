"""fal.ai Ideogram remove-background reference outputs ($0.01/image).

Requires the FAL_KEY env variable. Idempotent: the API is not called if the
output already exists.
"""
import os
from pathlib import Path

import fal_client
import requests

ENDPOINT = "fal-ai/ideogram/remove-background"


def _download(url: str, path: str) -> None:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    Path(path).write_bytes(r.content)


def fetch_reference(image_path: str, out_path: str) -> None:
    if Path(out_path).exists():
        return
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set: https://fal.ai/dashboard/keys")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    url = fal_client.upload_file(image_path)
    result = fal_client.subscribe(ENDPOINT, arguments={"image_url": url})
    _download(result["image"]["url"], out_path)
