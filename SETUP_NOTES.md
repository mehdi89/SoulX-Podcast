# SoulX-Podcast Setup and Testing Guide

## Setup Summary

The SoulX-Podcast-1.7B model has been successfully installed and tested on this system.

### Installation Details

- **Location**: `/home/azureuser/tubeonai/SoulX-Podcast`
- **Conda Environment**: `soulxpodcast` (Python 3.11)
- **Model**: `Soul-AILab/SoulX-Podcast-1.7B`
- **Model Path**: `pretrained_models/SoulX-Podcast-1.7B`

## Usage

### Method 1: Using the Convenience Script

A bash script has been created to simplify podcast generation:

```bash
./generate_podcast.sh --script <path_to_json> --output <output_wav> [--seed <number>]
```

**Examples:**
```bash
# Generate English podcast
./generate_podcast.sh --script example/podcast_script/script_english.json --output outputs/my_english_podcast.wav

# Generate Mandarin podcast with specific seed
./generate_podcast.sh --script example/podcast_script/script_mandarin.json --output outputs/my_mandarin_podcast.wav --seed 7
```

### Method 2: Using Python Directly

```bash
# Activate environment
deactivate 2>/dev/null
source /anaconda/etc/profile.d/conda.sh
conda activate soulxpodcast

# Set PYTHONPATH
export PYTHONPATH=/home/azureuser/tubeonai/SoulX-Podcast

# Run generation
cd /home/azureuser/tubeonai/SoulX-Podcast
python cli/podcast.py \
  --json_path example/podcast_script/script_english.json \
  --model_path pretrained_models/SoulX-Podcast-1.7B \
  --output_path outputs/my_podcast.wav \
  --seed 42
```

### Method 3: Using the Original Example Scripts

```bash
cd /home/azureuser/tubeonai/SoulX-Podcast
deactivate 2>/dev/null
source /anaconda/etc/profile.d/conda.sh
conda activate soulxpodcast
bash example/infer_dialogue.sh
```

## Available Example Scripts

The model comes with pre-built example scripts in `example/podcast_script/`:

- `script_english.json` - English dialogue
- `script_mandarin.json` - Mandarin Chinese dialogue
- `script_henan.json` - Henan dialect
- `script_sichuan.json` - Sichuan dialect
- `script_yue.json` - Cantonese (Yue) dialect

## Podcast Script Format

Podcast scripts are JSON files with the following structure:

```json
{
  "speakers": {
    "S1": {
      "prompt_audio": "path/to/audio.wav",
      "prompt_text": "Sample text in the target voice"
    },
    "S2": {
      "prompt_audio": "path/to/audio.wav",
      "prompt_text": "Sample text in the target voice"
    }
  },
  "text": [
    ["S1", "First speaker's dialogue"],
    ["S2", "Second speaker's dialogue"],
    ["S1", "First speaker's next line"]
  ]
}
```

## Test Results

Successfully generated podcasts:
- `outputs/mandarin.wav` (1.8MB) - Mandarin Chinese dialogue
- `outputs/english.wav` (1.9MB) - English dialogue

Both files were generated successfully with the model producing natural-sounding multi-speaker podcast audio.

## Features

- **Multi-speaker dialogue**: Supports conversations between multiple speakers
- **Zero-shot voice cloning**: Clone voices from audio prompts
- **Multiple languages**: English and Mandarin Chinese
- **Chinese dialects**: Sichuanese, Henanese, and Cantonese
- **Paralinguistic events**: Supports laughter, sighs, and other natural speech features

## Notes

- The model requires approximately 4-5GB of disk space
- Generation time varies depending on dialogue length (typically 1-2 minutes)
- A CUDA-capable GPU is recommended but not required
- The environment must be activated before each use
- Make sure to deactivate any Python venv before activating the conda environment

## Troubleshooting

If you encounter `ModuleNotFoundError: No module named 'torch'`:
- Ensure you've deactivated any active venv: `deactivate`
- Activate the correct conda environment: `conda activate soulxpodcast`
- Verify Python path: `which python` should show `/anaconda/envs/soulxpodcast/bin/python`

If you encounter `ModuleNotFoundError: No module named 'soulxpodcast'`:
- Set PYTHONPATH: `export PYTHONPATH=/home/azureuser/tubeonai/SoulX-Podcast`
