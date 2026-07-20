#!/usr/bin/env bash
# The full benchmark + gallery + comparison sequence to run once the
# fine-tune checkpoint (data/checkpoints/epoch_1.pth) arrives.
#
# Prerequisite: data/checkpoints/epoch_1.pth must exist (the bgr-v1 registry
# entry expects it, see bgr/registry.py MODEL_SPECS["bgr-v1"]).
#
# Usage:
#   bash scripts/benchmark_v1.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CKPT="data/checkpoints/epoch_1.pth"
MANIFEST="data/testset/manifest.jsonl"
OUT="results/baseline"
ALL_MODELS="birefnet-hr,rmbg-2.0,rmbg-2.0+refine,bgr-v1,bgr-v1+refine"

if [ ! -f "$CKPT" ]; then
  echo "ERROR: $CKPT not found. The checkpoint has not arrived yet." >&2
  exit 1
fi

echo "=== 1/4: bgr-v1 + bgr-v1+refine benchmark run (merging while preserving the metrics.json baselines) ==="
uv run python -m benchmark.run \
  --models bgr-v1,bgr-v1+refine \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --rgba

echo "=== 2/4: gallery refresh (5 models + ideogram) ==="
uv run python -m benchmark.gallery \
  --manifest "$MANIFEST" \
  --results "$OUT" \
  --models "$ALL_MODELS" \
  --out "$OUT/gallery.html"

echo "=== 3/4: comparison table (Markdown) ==="
uv run python scripts/compare_v1.py --metrics "$OUT/metrics.json" \
  | tee "$OUT/bgr-v1-comparison.md"

echo "=== 4/4: done ==="
echo "metrics.json : $OUT/metrics.json"
echo "gallery.html : $OUT/gallery.html"
echo "comparison   : $OUT/bgr-v1-comparison.md"
