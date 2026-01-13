#!/bin/bash
set -e

MODEL_DIR="${MODEL_PATH:-pretrained_models/SoulX-Podcast-1.7B}"

echo "=============================================="
echo "  SoulX-Podcast Worker Container"
echo "=============================================="
echo "Worker ID: ${WORKER_ID:-not-set}"
echo "Model Path: $MODEL_DIR"
echo "GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Check if model exists, download if not
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Model not found. Downloading..."
    huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "Model downloaded successfully!"
else
    echo "Model found at $MODEL_DIR"
fi

echo ""
echo "Starting worker..."
exec python run_worker.py "$@"
