#!/bin/bash
# Script to generate podcasts using SoulX-Podcast-1.7B

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Deactivate any existing venv and activate conda environment
deactivate 2>/dev/null
if [ -f "/anaconda/etc/profile.d/conda.sh" ]; then
    source /anaconda/etc/profile.d/conda.sh
    conda activate soulxpodcast
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate soulxpodcast
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate soulxpodcast
fi

# Set PYTHONPATH to project root
export PYTHONPATH="$SCRIPT_DIR"

# Default values
SCRIPT_PATH=""
OUTPUT_PATH=""
SEED=1988
MODEL_PATH="pretrained_models/SoulX-Podcast-1.7B"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --script <path_to_json> --output <output_wav> [--seed <number>] [--model <model_path>]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$SCRIPT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
  echo "Error: --script and --output are required"
  echo ""
  echo "Usage: $0 --script <path_to_json> --output <output_wav> [--seed <number>]"
  echo ""
  echo "Examples:"
  echo "  $0 --script example/podcast_script/script_openai.json --output outputs/my_podcast.wav"
  echo "  $0 --script my_script.json --output outputs/podcast.wav --seed 42"
  exit 1
fi

# Run the podcast generation
cd "$SCRIPT_DIR"
python cli/podcast.py \
  --json_path "$SCRIPT_PATH" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUTPUT_PATH" \
  --seed "$SEED"
