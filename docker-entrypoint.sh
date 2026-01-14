#!/bin/bash
set -e

MODEL_DIR="${MODEL_PATH:-pretrained_models/SoulX-Podcast-1.7B}"

echo "=============================================="
echo "  SoulX-Podcast Worker Container"
echo "=============================================="
echo "Worker ID: ${WORKER_ID:-not-set}"
echo "Model Path: $MODEL_DIR"
echo "GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Model is pre-baked in Docker image
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "Model: Pre-loaded in image"
else
    echo "WARNING: Model not found at $MODEL_DIR"
fi

echo ""
echo "Starting worker..."
exec python run_worker.py "$@"
