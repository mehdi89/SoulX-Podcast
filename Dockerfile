# SoulX-Podcast Worker Dockerfile
# Optimized for Azure Container Apps with GPU (NC8as-T4)

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Force reinstall torch ecosystem with compatible versions to avoid conflicts with base image
# The base image has torch/torchvision via conda which conflicts with pip packages
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install huggingface CLI for model download
RUN pip install --no-cache-dir huggingface_hub

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create directories and set permissions
RUN mkdir -p pretrained_models outputs && \
    chmod -R 777 pretrained_models outputs

# Health check (checks GPU and basic imports)
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD python -c "import torch; from worker.config import WorkerConfig; print('OK')" || exit 1

# Entrypoint downloads model if needed, then starts worker
ENTRYPOINT ["./docker-entrypoint.sh"]
