# SoulX-Podcast (TubeOnAI Fork)

A simplified fork of [SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast) optimized for English podcast generation with TubeOnAI integration.

## What's Different in This Fork

- **English-only** - Removed Chinese dialects and Mandarin content
- **Pre-configured voices** - S1 = Alice (female), S2 = Frank (male)
- **Simplified WebUI** - Single-page interface, just paste script and generate
- **Simplified API** - Plain text input, no file uploads needed

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

## Original Project

This is a fork of [Soul-AILab/SoulX-Podcast](https://github.com/Soul-AILab/SoulX-Podcast).

See the original repo for:
- Technical paper and research details
- Chinese dialect support
- Full feature set

## License

Apache 2.0 - See [LICENSE](LICENSE)
