# CLAUDE.md - SoulX-Podcast Project Guide

## Project Overview

SoulX-Podcast is a podcast generation system that converts text scripts to natural-sounding audio with multiple speakers. This is a simplified fork optimized for English-only TubeOnAI integration.

## Key Architecture

```
SoulX-Podcast/
├── webui.py              # Gradio web interface (simplified)
├── run_api.py            # FastAPI server launcher
├── run_worker.py         # Production worker entry point
├── Dockerfile            # Azure Container Apps deployment
├── docker-entrypoint.sh  # Container startup script
├── api/                  # REST API (for local dev/testing)
│   ├── main.py          # Endpoints: /generate-async, /task/{id}, /download
│   ├── service.py       # Model inference wrapper
│   ├── tasks.py         # Async task queue manager
│   └── models.py        # Pydantic schemas
├── worker/               # Production worker (Azure Container Apps)
│   ├── main.py          # Main loop - Azure Queue consumer
│   ├── config.py        # Environment configuration
│   ├── api_client.py    # TubeOnAI API client
│   ├── queue_client.py  # Azure Storage Queue client
│   ├── s3_client.py     # S3 upload client
│   └── processor.py     # Audio generation wrapper
├── soulxpodcast/         # Core model code
│   ├── models/          # SoulXPodcast model class
│   ├── engine/          # LLM engines (HF, VLLM)
│   └── utils/           # Text processing, audio handling
├── cli/                  # Command-line tools
│   └── podcast.py       # CLI for JSON script generation
├── .github/workflows/    # GitHub Actions CI/CD
└── example/
    ├── audios/          # Voice samples (en-Alice, en-Frank)
    └── podcast_script/  # Example JSON scripts
```

## Pre-configured Voices

- **S1** = Alice (female host) - `example/audios/en-Alice_woman.wav`
- **S2** = Frank (male guest) - `example/audios/en-Frank_man.wav`

Hardcoded in both `webui.py` and `api/main.py`.

## Script Format

Plain text with speaker tags:
```
[S1] Welcome to the show!
[S2] Thanks for having me.
```

## Running the Project

```bash
# Activate environment
conda activate soulxpodcast

# WebUI (port 7860)
python webui.py --model_path pretrained_models/SoulX-Podcast-1.7B

# API (port 8000)
python run_api.py --model pretrained_models/SoulX-Podcast-1.7B

# CLI
python cli/podcast.py --json_path script.json --model_path pretrained_models/SoulX-Podcast-1.7B --output_path output.wav
```

## API Quick Reference

```bash
# Simple generation (uses hardcoded voices)
POST /generate-async
  -F "dialogue_text=[S1] Hello! [S2] Hi!"
  -F "seed=1988"

# Custom voices
POST /generate-async-custom
  -F "prompt_audio=@voice1.wav"
  -F "prompt_audio=@voice2.wav"
  -F "prompt_texts=[\"text1\", \"text2\"]"
  -F "dialogue_text=[S1] Hello! [S2] Hi!"

# Check status
GET /task/{task_id}

# Download result
GET /download/{filename}
```

## Common Tasks

### Adding a new voice
1. Add WAV file to `example/audios/`
2. Update constants in `webui.py` and `api/main.py`

### Modifying generation parameters
- `webui.py`: `generate_podcast()` function
- `api/main.py`: `/generate-async` endpoint defaults

### Changing default seed
- WebUI: `EXAMPLE_SCRIPT` and `seed_input` default
- API: `seed` Form parameter default

## Dependencies

Key packages: `torch`, `transformers`, `gradio`, `fastapi`, `s3tokenizer`

Model requires ~4-5GB disk space and GPU recommended.

## TubeOnAI Integration (Production)

Deployed on **Azure Container Apps** with GPU (NC8as-T4). Job flow:

```
TubeOnAI Backend → Azure Queue → Worker Container → S3 → Backend Callback
```

1. TubeOnAI creates job, pushes message to Azure Storage Queue
2. Worker receives message, fetches job details from API
3. Worker generates audio using SoulX-Podcast model
4. Worker uploads WAV to S3
5. Worker calls API to mark job complete
6. TubeOnAI notifies user via WebSocket/email

### Deployment

Push to `main` triggers GitHub Actions → builds Docker image → deploys to Azure Container Apps.

### Key Files

- `Dockerfile` - Container with pre-baked model
- `worker/main.py` - Queue consumer loop + health endpoint
- `worker/queue_client.py` - Azure Storage Queue client
- `.github/workflows/*.yml` - CI/CD pipeline
