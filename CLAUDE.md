# CLAUDE.md - SoulX-Podcast Project Guide

## Project Overview

SoulX-Podcast is a podcast generation system that converts text scripts to natural-sounding audio with multiple speakers. This is a simplified fork optimized for English-only TubeOnAI integration.

## Key Architecture

```
SoulX-Podcast/
├── webui.py              # Gradio web interface (simplified)
├── run_api.py            # FastAPI server launcher
├── api/                  # REST API
│   ├── main.py          # Endpoints: /generate-async, /task/{id}, /download
│   ├── service.py       # Model inference wrapper
│   ├── tasks.py         # Async task queue manager
│   └── models.py        # Pydantic schemas
├── soulxpodcast/         # Core model code
│   ├── models/          # SoulXPodcast model class
│   ├── engine/          # LLM engines (HF, VLLM)
│   └── utils/           # Text processing, audio handling
├── cli/                  # Command-line tools
│   └── podcast.py       # CLI for JSON script generation
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

## TubeOnAI Integration

This fork is designed to receive podcast scripts from TubeOnAI (YouTube summary service) and generate audio. The simplified API accepts plain text, making integration straightforward:

1. TubeOnAI generates script with `[S1]`/`[S2]` tags
2. POST to `/generate-async`
3. Poll `/task/{id}` until complete
4. Download from `/download/{filename}`
