"""
Audio processor - wraps SoulX-Podcast model for generation.
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import WorkerConfig


logger = logging.getLogger(__name__)


class PodcastProcessor:
    """Wrapper around SoulX-Podcast model for audio generation."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.model = None
        self.dataset = None
        self._initialized = False

        # Hardcoded English voices (same as webui.py)
        self.s1_prompt_wav = "example/audios/en-Alice_woman.wav"
        self.s2_prompt_wav = "example/audios/en-Frank_man.wav"
        self.s1_prompt_text = "Welcome to Tech Talk, where we discuss the latest developments in artificial intelligence and technology."
        self.s2_prompt_text = "I'm excited to share my thoughts on how AI is transforming our world and what the future might hold."

    def initialize(self):
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return

        logger.info(f"Loading model from {self.config.model_path}")

        # Import here to avoid loading heavy dependencies until needed
        import importlib.util
        from datetime import datetime

        import s3tokenizer
        from tqdm import tqdm

        from soulxpodcast.config import Config, SoulXPodcastLLMConfig
        from soulxpodcast.models.soulxpodcast import SoulXPodcast
        from soulxpodcast.utils.dataloader import PodcastInferHandler

        # Check for VLLM
        llm_engine = "hf"
        if importlib.util.find_spec("vllm"):
            llm_engine = "vllm"
            logger.info("Using VLLM engine")
        else:
            logger.info("Using HuggingFace engine (VLLM not available)")

        # Load config
        hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={"fp16_flow": True},
            json_file=f"{self.config.model_path}/soulxpodcast_config.json"
        )

        config = Config(
            model=self.config.model_path,
            enforce_eager=True,
            llm_engine=llm_engine,
            hf_config=hf_config
        )

        # Initialize model
        self.model = SoulXPodcast(config)
        self.dataset = PodcastInferHandler(self.model.llm.tokenizer, None, config)

        self._initialized = True
        logger.info("Model loaded successfully")

    def generate(self, script: str, seed: int = 1988) -> tuple[str, int]:
        """
        Generate podcast audio from script.

        Args:
            script: Script with [S1] and [S2] speaker tags
            seed: Random seed for voice variation

        Returns:
            Tuple of (path to generated audio file, duration in seconds)
        """
        self.initialize()

        import random
        import s3tokenizer

        from soulxpodcast.config import SamplingParams

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Parse script into dialogue segments
        target_text_list = re.findall(r"(\[S[1-2]\][^\[\]]+)", script)
        target_text_list = [text.strip() for text in target_text_list if text.strip()]

        if not target_text_list:
            raise ValueError("Invalid script format. Use [S1] and [S2] speaker tags.")

        # Validate all segments have proper speaker tags
        for text in target_text_list:
            if not (text.startswith("[S1]") or text.startswith("[S2]")):
                raise ValueError("Each segment must start with [S1] or [S2]")

        logger.info(f"Processing {len(target_text_list)} dialogue turns")

        # Log GPU memory before generation
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory before generation: {gpu_mem_gb:.2f}GB / {gpu_total_gb:.2f}GB")

        # Use hardcoded English voices
        prompt_wav_list = [self.s1_prompt_wav, self.s2_prompt_wav]
        prompt_text_list = [self.s1_prompt_text, self.s2_prompt_text]

        # Process dialogue
        data = self._process_single(target_text_list, prompt_wav_list, prompt_text_list)

        # Generate audio
        logger.info("Starting model.forward_longform...")
        results_dict = self.model.forward_longform(**data)
        logger.info("model.forward_longform completed")

        # Log GPU memory after generation
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after generation: {gpu_mem_gb:.2f}GB")

        # Concatenate all generated audio segments
        logger.info(f"Concatenating {len(results_dict['generated_wavs'])} audio segments...")
        target_audio = None
        for i in range(len(results_dict['generated_wavs'])):
            if target_audio is None:
                target_audio = results_dict['generated_wavs'][i]
            else:
                target_audio = torch.concat([target_audio, results_dict['generated_wavs'][i]], axis=1)

        audio_numpy = target_audio.cpu().squeeze(0).numpy()
        logger.info(f"Audio concatenation complete, moving to CPU")

        # Calculate duration
        sample_rate = 24000
        duration_seconds = int(len(audio_numpy) / sample_rate)

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"podcast_{seed}.wav")
        sf.write(temp_path, audio_numpy, sample_rate)

        logger.info(f"Generated audio: {temp_path} ({duration_seconds}s)")

        return temp_path, duration_seconds

    def _process_single(self, target_text_list, prompt_wav_list, prompt_text_list):
        """Process dialogue for synthesis (adapted from webui.py)."""
        import s3tokenizer

        from soulxpodcast.config import SamplingParams

        spks, texts = [], []
        for target_text in target_text_list:
            pattern = r'(\[S[1-2]\])(.+)'
            match = re.match(pattern, target_text)
            if match:
                text, spk = match.group(2), int(match.group(1)[2]) - 1
                spks.append(spk)
                texts.append(text)

        dataitem = {
            "key": "001",
            "prompt_text": prompt_text_list,
            "prompt_wav": prompt_wav_list,
            "text": texts,
            "spk": spks,
        }
        self.dataset.update_datasource([dataitem])

        data = self.dataset[0]
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])
        spk_emb_for_flow = torch.tensor(data["spk_emb"])
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(
            data["mel"], batch_first=True, padding_value=0
        )
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


def get_audio_duration(file_path: str) -> int:
    """Get audio duration in seconds."""
    try:
        info = sf.info(file_path)
        return int(info.duration)
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0
