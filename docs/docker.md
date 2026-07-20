# Lucida with Docker

Docker image and embedded web UI to get Lucida running with a single command.

## Quick start

```bash
# Build the image (from the repo root)
docker build -t lucida .

# Run — mount the HF cache to a volume so the model downloads only once
docker run -p 8000:8000 -v hf-cache:/root/.cache/huggingface lucida
```

Open **http://localhost:8000** in your browser:

1. Drag and drop an image (or click to pick one).
2. Pick a model from the dropdown — default **lucida** (`egeorcun/lucida`, downloaded from HuggingFace).
3. The result is previewed on a transparency checkerboard background; save it with **Download PNG**.

## About the model weights

The weights are **not embedded** in the image. The selected model is
downloaded from HuggingFace on the first request and written under `HF_HOME`
(`/root/.cache/huggingface`). Thanks to the `-v hf-cache:...` volume above, no
re-download happens when the container restarts. Without the volume, every new
container downloads the model from scratch (lucida ~1 GB).

Note: the `bgr-v1`...`lucida-v7` entries require a local checkpoint
(`data/checkpoints/*.pth`); those files are not in the image. Inside the
container use the HF-based models: `lucida`, `birefnet-hr`, `rmbg-2.0`,
`inspyrenet`.

## Expected latency (CPU)

The image is built with the CPU-only torch wheel. At 1024 px resolution,
BiRefNet takes **roughly 5-15 seconds per image** on CPU (depending on core
count and image size). The first request additionally includes the model
download + load time.

## Running on a GPU

To run on an NVIDIA GPU:

1. Switch the torch install line in the `Dockerfile` to the CUDA wheel:

   ```dockerfile
   RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. Rebuild the image and run with `--gpus all`
   (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

   ```bash
   docker build -t lucida-gpu .
   docker run --gpus all -p 8000:8000 -v hf-cache:/root/.cache/huggingface lucida-gpu
   ```

## API

The API can also be used directly, without the web UI:

```bash
# Remove the background (returns a PNG)
curl -F "file=@photo.jpg" "http://localhost:8000/remove?model=lucida" -o out.png

# Available models
curl http://localhost:8000/models

# Health check
curl http://localhost:8000/health
```

`/remove` parameters: `model` (default `rmbg-2.0`), `refine` (bool),
`decontaminate` (bool, default `true`).
