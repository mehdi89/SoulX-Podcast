# SoulX-Podcast Setup and Usage Guide

## Overview

SoulX-Podcast is a podcast generation system that converts text scripts to natural-sounding audio with multiple speakers.

**Pre-configured voices:**
- **S1** = Alice (female host)
- **S2** = Frank (male guest)

## Quick Start

### Running the WebUI

```bash
python webui.py --model_path pretrained_models/SoulX-Podcast-1.7B
# Opens at http://0.0.0.0:7860
```

The WebUI provides a simple interface:
1. Paste your podcast script using `[S1]` and `[S2]` speaker tags
2. Click "Generate Podcast Audio"
3. Listen to or download the result

### Running the API Server

```bash
python run_api.py --model pretrained_models/SoulX-Podcast-1.7B --port 8000
# API docs at http://localhost:8000/docs
```

### API Usage

**Simple endpoint (recommended):**
```bash
# Create podcast generation task
curl -X POST "http://localhost:8000/generate-async" \
  -F "dialogue_text=[S1] Welcome to our podcast! [S2] Thanks for having me!" \
  -F "seed=1988"

# Response: {"task_id": "abc123...", "status": "pending", ...}

# Check task status
curl "http://localhost:8000/task/{task_id}"

# Download when complete
curl "http://localhost:8000/download/{task_id}.wav" -o output.wav
```

**Custom voices endpoint:**
```bash
curl -X POST "http://localhost:8000/generate-async-custom" \
  -F "prompt_audio=@voice1.wav" \
  -F "prompt_audio=@voice2.wav" \
  -F 'prompt_texts=["Reference text 1", "Reference text 2"]' \
  -F "dialogue_text=[S1] Hello! [S2] Hi there!"
```

## Script Format

Use `[S1]` for Alice (female) and `[S2]` for Frank (male):

```
[S1] Welcome to today's episode!
[S2] Thanks for having me on the show.
[S1] Let's dive into our topic.
[S2] Sounds great, let's get started!
```

## CLI Usage

```bash
./generate_podcast.sh --script example/podcast_script/script_openai.json --output outputs/my_podcast.wav
```

Or directly with Python:
```bash
python cli/podcast.py \
  --json_path example/podcast_script/script_openai.json \
  --model_path pretrained_models/SoulX-Podcast-1.7B \
  --output_path outputs/podcast.wav \
  --seed 1988
```

## JSON Script Format (for CLI)

```json
{
  "speakers": {
    "S1": {
      "prompt_audio": "example/audios/en-Alice_woman.wav",
      "prompt_text": "Sample text spoken in Alice's voice."
    },
    "S2": {
      "prompt_audio": "example/audios/en-Frank_man.wav",
      "prompt_text": "Sample text spoken in Frank's voice."
    }
  },
  "text": [
    ["S1", "First speaker's dialogue."],
    ["S2", "Second speaker's response."],
    ["S1", "First speaker continues..."]
  ]
}
```

## Environment Setup

```bash
# Create conda environment
conda create -n soulxpodcast -y python=3.11
conda activate soulxpodcast

# Install dependencies
pip install -r requirements.txt

# Download model
huggingface-cli download Soul-AILab/SoulX-Podcast-1.7B --local-dir pretrained_models/SoulX-Podcast-1.7B
```

## Features

- **Two-speaker podcasts** with pre-configured English voices
- **Zero-shot voice cloning** with custom audio (via advanced endpoint)
- **Paralinguistic controls**: `<|laughter|>`, `<|sigh|>`, `<|breathing|>`
- **Web UI** for easy testing
- **REST API** for integration with other services

## Notes

- Model requires ~4-5GB disk space
- Generation time varies by dialogue length (typically 30-120 seconds)
- GPU recommended for faster inference
- Audio output: 24kHz WAV format
