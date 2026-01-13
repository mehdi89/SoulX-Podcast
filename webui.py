import re
import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import importlib.util
from datetime import datetime

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

    return (24000, target_audio.cpu().squeeze(0).numpy())


def render_interface() -> gr.Blocks:
    """Render the simplified Gradio interface."""
    with gr.Blocks(
        title="SoulX-Podcast Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .info-box { background: #f0f4f8; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
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
            gr.Markdown(
                """
                **Script Format:**
                - Use `[S1]` for Alice (female host)
                - Use `[S2]` for Frank (male guest)
                - Example: `[S1] Hello everyone! [S2] Thanks for having me!`
                """,
                elem_classes="info-box"
            )

        with gr.Row():
            with gr.Column(scale=3):
                script_input = gr.Textbox(
                    label="Podcast Script",
                    placeholder="[S1] Welcome to the show!\n[S2] Thanks for having me!\n[S1] Let's get started...",
                    lines=15,
                    value=EXAMPLE_SCRIPT,
                )
            with gr.Column(scale=1):
                seed_input = gr.Number(
                    label="Seed",
                    value=1988,
                    step=1,
                    info="Change for different voice variations"
                )

        generate_btn = gr.Button(
            "Generate Podcast Audio",
            variant="primary",
            size="lg",
        )

        output_audio = gr.Audio(
            label="Generated Podcast",
            interactive=False,
        )

        # Wire up the generate button
        generate_btn.click(
            fn=generate_podcast,
            inputs=[script_input, seed_input],
            outputs=[output_audio],
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
