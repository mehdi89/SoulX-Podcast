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

## Production Worker (TubeOnAI Integration)

The worker is a pull-based service that polls TubeOnAI for podcast generation jobs. It supports multiple GPU servers for horizontal scaling.

### Architecture

```
TubeOnAI Backend                    GPU Workers
┌─────────────┐                ┌─────────────────┐
│  Job Queue  │◄───── poll ────│  GPU Server 1   │
│  (Database) │                └────────┬────────┘
│             │                         │
│             │                ┌────────▼────────┐
│             │                │       S3        │
│             │                │    (Storage)    │
│             │                └─────────────────┘
└─────────────┘
```

### Quick Setup (New GPU Server)

```bash
# One-command setup
curl -fsSL https://raw.githubusercontent.com/mehdi89/SoulX-Podcast/main/setup_worker.sh | bash

# Configure
cd SoulX-Podcast
nano worker/.env  # Add your credentials

# Run
conda activate soulxpodcast
python run_worker.py
```

### Manual Setup

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

# Run worker
python run_worker.py
```

### Configuration

Edit `worker/.env`:

```bash
# Worker identity (unique per server)
WORKER_ID=gpu-server-1
SERVER_NAME=Azure ML East US

# TubeOnAI API
TUBEONAI_API_URL=https://api.tubeonai.com/podcast-worker/v1
TUBEONAI_API_TOKEN=your-token-here

# S3 Storage
S3_BUCKET=your-bucket
S3_ACCESS_KEY=your-key
S3_SECRET_KEY=your-secret
S3_REGION=us-east-1

# Settings
POLL_INTERVAL=10      # Seconds between job polls
HEARTBEAT_INTERVAL=60 # Seconds between heartbeats
```

### Run as System Service

```bash
# Install service
sudo cp worker/podcast-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable podcast-worker
sudo systemctl start podcast-worker

# Check status
sudo systemctl status podcast-worker
sudo journalctl -u podcast-worker -f  # View logs
```

### Worker API (for TubeOnAI Backend)

The worker expects these endpoints from TubeOnAI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs/claim` | POST | Claim a pending job |
| `/jobs/{id}/complete` | POST | Mark job complete with S3 URL |
| `/jobs/{id}/failed` | POST | Mark job as failed |
| `/workers/heartbeat` | POST | Send health status |

See [Integration Design Doc](docs/plans/2026-01-13-tubeonai-integration-design.md) for full API specs.

## Original Project

This is a fork of [Soul-AILab/SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast).

See the original repo for:
- Technical paper and research details
- Chinese dialect support
- Full feature set

## License

Apache 2.0 - See [LICENSE](LICENSE)
