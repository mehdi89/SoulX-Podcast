#!/bin/bash
# Script to generate podcasts using SoulX-Podcast-1.7B

# Deactivate any existing venv and activate conda environment
deactivate 2>/dev/null
source /anaconda/etc/profile.d/conda.sh
conda activate soulxpodcast

# Set PYTHONPATH
export PYTHONPATH=/home/azureuser/tubeonai/SoulX-Podcast

# Default values
SCRIPT_PATH=""
OUTPUT_PATH=""
SEED=42

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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --script <path_to_json> --output <output_wav> [--seed <number>]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$SCRIPT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
  echo "Error: --script and --output are required"
  echo "Usage: $0 --script <path_to_json> --output <output_wav> [--seed <number>]"
  echo ""
  echo "Examples:"
  echo "  $0 --script example/podcast_script/script_english.json --output outputs/my_podcast.wav"
  echo "  $0 --script example/podcast_script/script_mandarin.json --output outputs/mandarin_podcast.wav --seed 7"
  exit 1
fi

# Run the podcast generation
cd /home/azureuser/tubeonai/SoulX-Podcast
python cli/podcast.py \
  --json_path "$SCRIPT_PATH" \
  --model_path pretrained_models/SoulX-Podcast-1.7B \
  --output_path "$OUTPUT_PATH" \
  --seed "$SEED"
