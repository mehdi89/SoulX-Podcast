import re
import os
import gradio as gr
from tqdm import tqdm
from itertools import chain
from argparse import ArgumentParser
import sys
import importlib.util
from datetime import datetime
import soundfile as sf

import torch
import numpy as np
import random
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import (
    PodcastInferHandler,
    SPK_DICT, TEXT_START, TEXT_END, AUDIO_START, TASK_PODCAST
)

# Output directory for saved podcasts
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded English voices
S1_PROMPT_WAV = "example/audios/en-Alice_woman.wav"
S2_PROMPT_WAV = "example/audios/en-Frank_man.wav"
S1_PROMPT_TEXT = "Welcome to Tech Talk, where we discuss the latest developments in artificial intelligence and technology."
S2_PROMPT_TEXT = "I'm excited to share my thoughts on how AI is transforming our world and what the future might hold."

# Example script for reference
EXAMPLE_SCRIPT = """[S1] Hello and welcome to today's podcast! I'm your host Alice.
[S2] Thanks for having me! I'm Frank, and I'm excited to be here.
[S1] Today we're going to talk about something really interesting.
[S2] Yes, it's going to be a great conversation!"""

model: SoulXPodcast = None
dataset: PodcastInferHandler = None


def save_audio(audio_data, sample_rate=24000, prefix="podcast"):
    """Save audio to outputs directory and return the filepath."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, audio_data, sample_rate)
    return filepath

def initiate_model(config: Config, enable_tn: bool=False):
    global model
    if model is None:
        model = SoulXPodcast(config)

    global dataset
    if dataset is None:
        dataset = PodcastInferHandler(model.llm.tokenizer, None, config)


def process_single(target_text_list, prompt_wav_list, prompt_text_list):
    """Process dialogue for synthesis."""
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S[1-2]\])(.+)'
        match = re.match(pattern, target_text)
        if match:
            text, spk = match.group(2), int(match.group(1)[2])-1
            spks.append(spk)
            texts.append(text)

    global dataset
    dataitem = {
        "key": "001",
        "prompt_text": prompt_text_list,
        "prompt_wav": prompt_wav_list,
        "text": texts,
        "spk": spks,
    }
    dataset.update_datasource([dataitem])

    data = dataset[0]
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
    spk_emb_for_flow = torch.tensor(data["spk_emb"])
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(data["mel"], batch_first=True, padding_value=0)
    prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
    text_tokens_for_llm = data["text_tokens"]
    prompt_text_tokens_for_llm = data["prompt_text_tokens"]
    spk_ids = data["spks_list"]
    sampling_params = SamplingParams(use_ras=True, win_size=25, tau_r=0.2)
    infos = [data["info"]]

    return {
        "prompt_mels_for_llm": prompt_mels_for_llm,
        "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
        "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
        "text_tokens_for_llm": text_tokens_for_llm,
        "prompt_mels_for_flow_ori": prompt_mels_for_flow,
        "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
        "spk_emb_for_flow": spk_emb_for_flow,
        "sampling_params": sampling_params,
        "spk_ids": spk_ids,
        "infos": infos,
        "use_dialect_prompt": False,
    }


def generate_podcast(script: str, seed: int = 1988):
    """Generate podcast audio from script with speaker tags."""
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Parse script into dialogue segments
    target_text_list = re.findall(r"(\[S[1-2]\][^\[\]]+)", script)
    target_text_list = [text.strip() for text in target_text_list if text.strip()]

    if not target_text_list:
        gr.Warning("Invalid script format. Use [S1] and [S2] speaker tags.")
        return None

    # Validate all segments have proper speaker tags
    for text in target_text_list:
        if not (text.startswith("[S1]") or text.startswith("[S2]")):
            gr.Warning("Each segment must start with [S1] or [S2]")
            return None

    # Use hardcoded English voices
    prompt_wav_list = [S1_PROMPT_WAV, S2_PROMPT_WAV]
    prompt_text_list = [S1_PROMPT_TEXT, S2_PROMPT_TEXT]

    # Process and generate
    progress_bar = gr.Progress(track_tqdm=True)
    data = process_single(target_text_list, prompt_wav_list, prompt_text_list)
    results_dict = model.forward_longform(**data)

    # Concatenate all generated audio segments
    target_audio = None
    for i in range(len(results_dict['generated_wavs'])):
        if target_audio is None:
            target_audio = results_dict['generated_wavs'][i]
        else:
            target_audio = torch.concat([target_audio, results_dict['generated_wavs'][i]], axis=1)

    audio_numpy = target_audio.cpu().squeeze(0).numpy()

    # Auto-save to outputs directory
    filepath = save_audio(audio_numpy, 24000, "podcast")
    print(f"[INFO] Saved podcast to: {filepath}")

    return (24000, audio_numpy)


def generate_podcast_streaming(script: str, seed: int = 1988):
    """Generate podcast audio with streaming - yields audio after each turn."""
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Parse script into dialogue segments
    target_text_list = re.findall(r"(\[S[1-2]\][^\[\]]+)", script)
    target_text_list = [text.strip() for text in target_text_list if text.strip()]

    if not target_text_list:
        gr.Warning("Invalid script format. Use [S1] and [S2] speaker tags.")
        yield None, "Error: Invalid script format"
        return

    # Validate all segments have proper speaker tags
    for text in target_text_list:
        if not (text.startswith("[S1]") or text.startswith("[S2]")):
            gr.Warning("Each segment must start with [S1] or [S2]")
            yield None, "Error: Missing speaker tags"
            return

    # Use hardcoded English voices
    prompt_wav_list = [S1_PROMPT_WAV, S2_PROMPT_WAV]
    prompt_text_list = [S1_PROMPT_TEXT, S2_PROMPT_TEXT]

    # Process data preparation
    data = process_single(target_text_list, prompt_wav_list, prompt_text_list)

    total_turns = len(target_text_list)
    yield None, f"Processing {total_turns} dialogue turns..."

    # Generate with streaming using the model's turn-by-turn processing
    accumulated_audio = None

    for i, (turn_text, wav) in enumerate(generate_turns_streaming(data)):
        if accumulated_audio is None:
            accumulated_audio = wav
        else:
            accumulated_audio = torch.concat([accumulated_audio, wav], axis=1)

        # Get speaker from the turn text
        speaker = "Alice" if turn_text.startswith("[S1]") else "Frank"
        status = f"Generated turn {i+1}/{total_turns}: {speaker}"

        # Yield the accumulated audio so far
        yield (24000, accumulated_audio.cpu().squeeze(0).numpy()), status

    # Save final audio
    final_audio = accumulated_audio.cpu().squeeze(0).numpy()
    filepath = save_audio(final_audio, 24000, "podcast")
    print(f"[INFO] Saved podcast to: {filepath}")

    yield (24000, final_audio), f"Complete! Saved to {filepath}"


@torch.inference_mode()
def generate_turns_streaming(data):
    """Generator that yields (turn_text, audio) for each turn as it's generated."""
    global model

    # Extract parameters
    prompt_mels_for_llm = data["prompt_mels_for_llm"]
    prompt_mels_lens_for_llm = data["prompt_mels_lens_for_llm"]
    prompt_text_tokens_for_llm = data["prompt_text_tokens_for_llm"]
    text_tokens_for_llm = data["text_tokens_for_llm"]
    prompt_mels_for_flow_ori = data["prompt_mels_for_flow_ori"]
    spk_emb_for_flow = data["spk_emb_for_flow"]
    sampling_params = data["sampling_params"]
    spk_ids = data["spk_ids"]
    use_dialect_prompt = data.get("use_dialect_prompt", False)
    infos = data.get("infos", [{}])

    # Get turn texts from infos for status display
    turn_texts = []
    if infos and len(infos) > 0 and "text" in infos[0]:
        for i, text in enumerate(infos[0].get("text", [])):
            spk = infos[0].get("spk", [])[i] if "spk" in infos[0] else 0
            turn_texts.append(f"[S{spk+1}]{text}")

    prompt_size, turn_size = len(prompt_mels_for_llm), len(text_tokens_for_llm)

    # Audio tokenization
    prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = model.audio_tokenizer.quantize(
        prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
    )

    # Align speech token with speech feat
    prompt_speech_tokens = []
    prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []

    for prompt_index in range(prompt_size):
        prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_index].item()
        prompt_speech_token = prompt_speech_tokens_ori[prompt_index, :prompt_speech_token_len]
        prompt_mel = prompt_mels_for_flow_ori[prompt_index]
        prompt_mel_len = prompt_mel.shape[0]
        if prompt_speech_token_len * 2 > prompt_mel_len:
            prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
            prompt_mel_len = torch.tensor([prompt_mel_len]).cuda()
        else:
            prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].cuda()
            prompt_mel_len = torch.tensor([prompt_speech_token_len * 2]).cuda()
        prompt_speech_tokens.append(prompt_speech_token)
        prompt_mels_for_flow.append(prompt_mel)
        prompt_mels_lens_for_flow.append(prompt_mel_len)

    # Prepare LLM inputs
    from transformers import DynamicCache
    from soulxpodcast.config import AutoPretrainedConfig

    prompt_inputs = []
    history_inputs = []

    for i in range(prompt_size):
        speech_tokens_i = [token + model.config.hf_config.speech_token_offset for token in prompt_speech_tokens[i].tolist()]
        speech_tokens_i += [model.config.hf_config.eos_token_id]
        prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i)
        history_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i)

    # LLM generation - turn by turn with yield
    inputs = list(chain.from_iterable(prompt_inputs))
    cache_config = AutoPretrainedConfig().from_dataclass(model.llm.config.hf_config)
    past_key_values = DynamicCache(config=cache_config)
    valid_turn_size = prompt_size

    for i in range(turn_size):
        # Reset cache if needed
        if valid_turn_size > model.config.max_turn_size or len(inputs) > model.config.turn_tokens_threshold:
            prompt_text_bound = max(model.config.prompt_context, len(history_inputs) - model.config.history_text_context - model.config.history_context)
            inputs = list(chain.from_iterable(
                history_inputs[:model.config.prompt_context] +
                history_inputs[prompt_text_bound:-model.config.history_context] +
                prompt_inputs[-model.config.history_context:]
            ))
            valid_turn_size = model.config.prompt_context + len(history_inputs) - prompt_text_bound
            past_key_values = DynamicCache(config=cache_config)
        valid_turn_size += 1

        inputs.extend(text_tokens_for_llm[i])
        llm_outputs = model.llm.generate(inputs, sampling_params, past_key_values=past_key_values)

        inputs.extend(llm_outputs['token_ids'])
        prompt_inputs.append(text_tokens_for_llm[i] + llm_outputs['token_ids'])
        history_inputs.append(text_tokens_for_llm[i][:-1])

        # Prepare Flow inputs
        turn_spk = spk_ids[i]
        generated_speech_tokens = [token - model.config.hf_config.speech_token_offset for token in llm_outputs['token_ids'][:-1]]
        prompt_speech_token = prompt_speech_tokens[turn_spk].tolist()
        flow_input = torch.tensor([prompt_speech_token + generated_speech_tokens])
        flow_inputs_len = torch.tensor([len(prompt_speech_token) + len(generated_speech_tokens)])

        # Flow generation
        start_idx = spk_ids[i]
        prompt_mels = prompt_mels_for_flow[start_idx][None]
        prompt_mels_lens = prompt_mels_lens_for_flow[start_idx][None]
        spk_emb = spk_emb_for_flow[start_idx:start_idx+1]

        with torch.amp.autocast("cuda", dtype=torch.float16 if model.config.hf_config.fp16_flow else torch.float32):
            generated_mels, generated_mels_lens = model.flow(
                flow_input.cuda(), flow_inputs_len.cuda(),
                prompt_mels, prompt_mels_lens, spk_emb.cuda(),
                streaming=False, finalize=True
            )

        # HiFi-GAN generation
        mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]
        wav, _ = model.hift(speech_feat=mel)

        # Get turn text for status
        turn_text = turn_texts[i] if i < len(turn_texts) else f"[S{spk_ids[i]+1}] Turn {i+1}"
        yield turn_text, wav


def render_interface() -> gr.Blocks:
    """Render the simplified Gradio interface with streaming support."""
    with gr.Blocks(
        title="SoulX-Podcast Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .status-text { font-family: monospace; padding: 10px; background: #1a1a2e; color: #00d4aa; border-radius: 6px; }
        .generate-btn { min-height: 50px !important; }
        """
    ) as page:

        gr.Markdown(
            """
            # SoulX-Podcast Generator
            Generate natural-sounding podcast audio from text scripts.

            **Voices:** S1 = Alice (female host) | S2 = Frank (male guest)
            """,
            elem_classes="main-header"
        )

        with gr.Row():
            with gr.Column(scale=3):
                script_input = gr.Textbox(
                    label="Podcast Script",
                    placeholder="[S1] Welcome to the show!\n[S2] Thanks for having me!\n[S1] Let's get started...",
                    lines=12,
                    value=EXAMPLE_SCRIPT,
                    info="Use [S1] for Alice (female) and [S2] for Frank (male)"
                )
            with gr.Column(scale=1):
                seed_input = gr.Number(
                    label="Seed",
                    value=1988,
                    step=1,
                    info="Change for different voice variations"
                )

                status_display = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    elem_classes="status-text"
                )

        with gr.Row():
            generate_btn = gr.Button(
                "Generate Podcast (Streaming)",
                variant="primary",
                size="lg",
                elem_classes="generate-btn"
            )

            generate_full_btn = gr.Button(
                "Generate Full (Wait for Complete)",
                variant="secondary",
                size="lg",
            )

        output_audio = gr.Audio(
            label="Generated Podcast",
            interactive=False,
            streaming=True,
        )

        # Wire up the streaming generate button
        generate_btn.click(
            fn=generate_podcast_streaming,
            inputs=[script_input, seed_input],
            outputs=[output_audio, status_display],
        )

        # Wire up the full generate button (non-streaming)
        generate_full_btn.click(
            fn=lambda script, seed: (generate_podcast(script, seed), "Complete!"),
            inputs=[script_input, seed_input],
            outputs=[output_audio, status_display],
        )

    return page


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='Path to the SoulX-Podcast model')
    parser.add_argument('--llm_engine',
                        type=str,
                        default="hf",
                        help='LLM engine: "hf" (HuggingFace) or "vllm"')
    parser.add_argument('--fp16_flow',
                        action='store_true',
                        help='Enable FP16 for flow model')
    parser.add_argument('--seed',
                        type=int,
                        default=1988,
                        help='Default random seed')
    parser.add_argument('--port',
                        type=int,
                        default=7860,
                        help='Gradio server port')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initialize model
    hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
        initial_values={"fp16_flow": args.fp16_flow},
        json_file=f"{args.model_path}/soulxpodcast_config.json"
    )

    llm_engine = args.llm_engine
    if llm_engine == "vllm":
        if not importlib.util.find_spec("vllm"):
            llm_engine = "hf"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING]: VLLM not installed, using HF engine.")

    config = Config(
        model=args.model_path,
        enforce_eager=True,
        llm_engine=llm_engine,
        hf_config=hf_config
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    initiate_model(config)
    print("[INFO] SoulX-Podcast model loaded successfully")

    page = render_interface()
    page.queue()
    page.launch(share=False, server_name="0.0.0.0", server_port=args.port)
