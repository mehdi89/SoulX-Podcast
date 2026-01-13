import json
import datetime
import os


def generate_time_index():
    """Generate a time-based unique key, e.g. '20251023-001'."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def podcast_format_parser(data, output_dir="outputs"):
    """
    Parse the original multi-speaker podcast JSON to the target flattened format.
    The key will be a time-based unique ID.

    Args:
        data (dict): input JSON data with 'speakers' and 'text' fields
        output_dir (str): directory for output wav file path

    Returns:
        dict: converted format
    """
    speakers = data.get("speakers", {})
    text_entries = data.get("text", [])

    # Create speaker name -> numeric ID mapping
    spk2id = {name: idx for idx, name in enumerate(speakers.keys())}

    # Collect prompts
    prompt_text = []
    prompt_wav = []

    for name in speakers:
        prompt_text.append(speakers[name].get("prompt_text", ""))
        prompt_wav.append(speakers[name].get("prompt_audio", ""))

    # Collect dialogue text and speaker sequence
    text_list = []
    spk_list = []
    for turn in text_entries:
        if len(turn) == 2:
            spk_name, utt_text = turn
            text = f'[{spk_name}]{utt_text}'
            text_list.append(text)
            spk_list.append(spk2id.get(spk_name, -1))

    # Generate time-based key
    key = generate_time_index()
    wav_path = os.path.join(output_dir, f"{key}.wav")

    result = {
        "key": key,
        "prompt_text": prompt_text,
        "prompt_wav": prompt_wav,
        "text": text_list,
        "spk": spk_list,
        "wav": wav_path,
        "use_dialect_prompt": False,
        "dialect_prompt_text": ["", ""]
    }

    return result


# Example usage
if __name__ == "__main__":
    with open("example/podcast_script/script_openai.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = podcast_format_parser(data)
    print(json.dumps(converted, ensure_ascii=False, indent=2))
