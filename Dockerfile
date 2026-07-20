# Lucida background remover — CPU image.
# Model weights are not embedded in the image; downloaded from HuggingFace on the first request.
# For a persistent cache: -v hf-cache:/root/.cache/huggingface
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

# opencv (a transparent-background dependency) needs libGL at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch from the CPU wheel first — the default CUDA wheels bloat the
# image needlessly (by ~GBs). For a GPU image drop this line and use
# `--index-url https://download.pytorch.org/whl/cu124`.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Package install (the wheel contains only bgr/ and benchmark/).
COPY pyproject.toml ./
COPY bgr/ bgr/
COPY benchmark/ benchmark/
RUN pip install .

# The serving code is not part of the package; imported from the workdir.
COPY serving/ serving/

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
