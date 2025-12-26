import gc
import torch
from huggingface_hub import snapshot_download

from mira.decoder import AudioDecoder
from mira.encoder import AudioEncoder


class TTSCodec:
    """Device-aware TTS codec for encoding/decoding audio."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        d_path = snapshot_download("YatharthS/MiraTTS")
        d_path = f"{d_path}/decoders"
        self.audio_decoder = AudioDecoder(d_path, device=device)
        self.audio_encoder = AudioEncoder(d_path, device=device)

    def encode(self, audio, encode_semantic: bool = False, duration: int = 8):
        """Encode audio file into tokens."""
        if encode_semantic:
            speech_tokens, context_tokens = self.audio_encoder.encode(
                audio, True, duration=duration
            )
            return speech_tokens, context_tokens
        else:
            context_tokens = self.audio_encoder.encode(audio, False, duration=duration)
            return context_tokens

    def format_prompt(
        self,
        text: str,
        context_tokens: str,
        extra_tokens,
        semantic_tokens: str = None,
        transcript: str = None,
    ) -> str:
        """Format the prompt for the LLM."""
        if transcript:
            prompt = (
                f"<|task_tts|><|start_text|>{transcript}{text}<|end_text|>"
                f"<|context_audio_start|>{context_tokens}<|context_audio_end|>"
                f"<|prompt_speech_start|>{semantic_tokens}"
            )
        else:
            prompt = (
                f"<|task_tts|><|start_text|>{text}<|end_text|>"
                f"<|context_audio_start|>{context_tokens}<|context_audio_end|>"
                f"<|prompt_speech_start|>"
            )
        return prompt

    def c_cache(self):
        """Clear memory cache."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decode(self, speech_tokens: str, context_tokens: str, test_var=None):
        """Decode tokens back to audio."""
        wav = self.audio_decoder.detokenize(context_tokens, speech_tokens)
        return wav
