export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

model_dir=pretrained_models/SoulX-Podcast-1.7B
input_file=example/podcast_script/script_openai.json

python cli/podcast.py \
        --json_path ${input_file} \
        --model_path ${model_dir} \
        --output_path outputs/podcast.wav \
        --seed 1988
