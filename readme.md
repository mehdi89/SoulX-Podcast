# SoulX-Podcast (TubeOnAI Fork)

A simplified fork of [SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast) optimized for English podcast generation with TubeOnAI integration.

## What's Different in This Fork

- **English-only** - Removed Chinese dialects and Mandarin content
- **Pre-configured voices** - S1 = Alice (female), S2 = Frank (male)
- **Simplified WebUI** - Single-page interface, just paste script and generate
- **Simplified API** - Plain text input, no file uploads needed
- **Production Worker** - Pull-based worker for TubeOnAI integration with multi-GPU support

## Quick Start

### Installation

```bash
# Clone repo
git clone https://github.com/mehdi89/SoulX-Podcast.git
cd SoulX-Podcast

# Create environment
conda create -n soulxpodcast -y python=3.11
conda activate soulxpodcast
pip install -r requirements.txt

# Download model
huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B --local-dir pretrained_models/SoulX-Podcast-1.7B
```

### WebUI

```bash
python webui.py --model_path pretrained_models/SoulX-Podcast-1.7B
# Open http://localhost:7860
```

Paste your script using `[S1]` and `[S2]` tags:
```
[S1] Welcome to the podcast!
[S2] Thanks for having me.
[S1] Let's dive in.
```

### API Server

```bash
python run_api.py --model pretrained_models/SoulX-Podcast-1.7B
# API docs at http://localhost:8000/docs
```

**Generate podcast:**
```bash
# Create task
curl -X POST "http://localhost:8000/generate-async" \
  -F "dialogue_text=[S1] Hello! [S2] Hi there!"

# Check status
curl "http://localhost:8000/task/{task_id}"

# Download result
curl "http://localhost:8000/download/{task_id}.wav" -o podcast.wav
```

### CLI

```bash
./generate_podcast.sh \
  --script example/podcast_script/script_openai.json \
  --output outputs/podcast.wav
```

## Script Formats

### Plain Text (WebUI & API)
```
[S1] First speaker says this.
[S2] Second speaker responds.
[S1] Back to first speaker.
```

### JSON (CLI)
```json
{
  "speakers": {
    "S1": {
      "prompt_audio": "example/audios/en-Alice_woman.wav",
      "prompt_text": "Sample text in Alice's voice."
    },
    "S2": {
      "prompt_audio": "example/audios/en-Frank_man.wav",
      "prompt_text": "Sample text in Frank's voice."
    }
  },
  "text": [
    ["S1", "First speaker dialogue."],
    ["S2", "Second speaker response."]
  ]
}
```

## Features

- **Two-speaker podcasts** with natural English voices
- **Paralinguistic controls**: `<|laughter|>`, `<|sigh|>`, `<|breathing|>`
- **Custom voices** via `/generate-async-custom` endpoint
- **24kHz audio output**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate-async` | POST | Generate with pre-configured voices |
| `/generate-async-custom` | POST | Generate with custom voice files |
| `/task/{task_id}` | GET | Check task status |
| `/download/{filename}` | GET | Download generated audio |
| `/health` | GET | Health check |

## Production Deployment (Azure Container Apps)

The worker runs on **Azure Container Apps** with GPU support (NC8as-T4), using Azure Queue for job distribution. Deployment is automated via GitHub Actions.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TubeOnAI Backend                            │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐   │
│  │  Web App │───▶│   Laravel    │───▶│     Azure Queue         │   │
│  │  (User)  │    │   Backend    │    │  (podcast-jobs queue)   │   │
│  └──────────┘    └──────────────┘    └───────────┬─────────────┘   │
│                         │                        │                  │
│                         │ Notifications          │                  │
│                         ▼                        │                  │
│              ┌──────────────────┐                │                  │
│              │ WebSocket/Email  │                │                  │
│              └──────────────────┘                │                  │
└──────────────────────────────────────────────────┼──────────────────┘
                                                   │
                    ┌──────────────────────────────▼──────────────────┐
                    │           Azure Container Apps (GPU)            │
                    │  ┌─────────────────────────────────────────┐    │
                    │  │  Podcast Worker Container               │    │
                    │  │  - Receives jobs from Azure Queue       │    │
                    │  │  - Generates audio with SoulX-Podcast   │    │
                    │  │  - Uploads to S3                        │    │
                    │  │  - Reports completion to API            │    │
                    │  └─────────────────────────────────────────┘    │
                    │                      │                          │
                    │                      ▼                          │
                    │              ┌─────────────┐                    │
                    │              │     S3      │                    │
                    │              │   Storage   │                    │
                    │              └─────────────┘                    │
                    └─────────────────────────────────────────────────┘
```

### Deployment Flow

```
GitHub (main branch) → GitHub Actions → Azure Container Registry → Azure Container Apps
```

Pushing to `main` automatically triggers:
1. Docker image build (includes pre-baked model for fast cold starts)
2. Push to Azure Container Registry
3. Deploy to Azure Container Apps with GPU

### CI/CD Configuration

The GitHub Actions workflow (`.github/workflows/`) handles deployment:
- **Trigger**: Push to `main` branch
- **Authentication**: Azure OIDC (federated credentials)
- **Container App**: `tubeonai-container-gpu`
- **Resource Group**: `TubeOnAI`

### Local Development

```bash
# Clone and setup
git clone https://github.com/mehdi89/SoulX-Podcast.git
cd SoulX-Podcast
conda create -n soulxpodcast -y python=3.11
conda activate soulxpodcast
pip install -r requirements.txt

# Download model
huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B \
  --local-dir pretrained_models/SoulX-Podcast-1.7B

# Configure worker
cp worker/.env.example worker/.env
nano worker/.env  # Edit with your credentials

# Run worker locally
python run_worker.py
```

### Configuration

Environment variables (set in Azure Container Apps or `worker/.env` for local dev):

```bash
# Worker identity
WORKER_ID=podcast-worker-1

# TubeOnAI API
TUBEONAI_API_URL=https://api.tubeonai.com/podcast-worker/v1
TUBEONAI_API_TOKEN=your-token-here

# Azure Queue (primary job source)
AZURE_QUEUE_ENABLED=true
AZURE_QUEUE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_QUEUE_NAME=podcast-jobs

# S3 Storage (for generated audio)
S3_BUCKET=your-bucket
S3_ACCESS_KEY=your-key
S3_SECRET_KEY=your-secret
S3_REGION=us-east-1

# Model
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B

# Settings
POLL_INTERVAL=10  # Seconds between queue polls when no messages
```

### Docker Configuration

The `Dockerfile` is optimized for Azure Container Apps:
- Base image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`
- Pre-bakes the 1.7B model (~4-5GB) for faster cold starts
- Exposes port 8080 for health checks
- Health endpoint at `/health` for Azure liveness probes

### Worker API (for TubeOnAI Backend)

The worker uses these endpoints from TubeOnAI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs/{id}` | GET | Fetch job details |
| `/jobs/{id}/complete` | POST | Mark job complete with S3 URL |
| `/jobs/{id}/failed` | POST | Mark job as failed |

Jobs are distributed via **Azure Storage Queue** (not API polling).

## Original Project

This is a fork of [Soul-AILab/SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast).

See the original repo for:
- Technical paper and research details
- Chinese dialect support
- Full feature set

## License

Apache 2.0 - See [LICENSE](LICENSE)
