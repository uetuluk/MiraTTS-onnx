import torch
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Union

from mira.codec import TTSCodec
from transformers import AutoTokenizer, LogitsProcessorList
from optimum.onnxruntime import ORTModelForCausalLM

from mira.utils import clear_cache, split_text
from mira.sampling import MinPLogitsProcessor


class MiraTTS:
    """MiraTTS with ONNX Runtime backend for fully ONNX-compliant inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        device: str = "cuda",
        provider: str = "CUDAExecutionProvider",
        verbose: bool = False,
    ):
        """
        Initialize MiraTTS with ONNX Runtime backend.

        Args:
            model_path: Path to ONNX model directory OR HuggingFace model ID.
                       - Local ONNX: "./onnx_model" (pre-converted)
                       - HuggingFace: "YatharthS/MiraTTS" (auto-converts on first use)
            n_ctx: Maximum context length (kept for API compatibility)
            device: Device to use ("cuda" or "cpu")
            provider: ONNX Runtime execution provider
            verbose: Print verbose output
        """
        self.device = device
        self.n_ctx = n_ctx
        self.verbose = verbose

        # Determine if model_path is a local ONNX directory or HuggingFace ID
        model_path_obj = Path(model_path)
        is_local_onnx = model_path_obj.exists() and (
            (model_path_obj / "model.onnx").exists()
            or (model_path_obj / "decoder_model.onnx").exists()
        )

        # Load tokenizer
        if is_local_onnx:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path_obj))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check available providers and configure
        import onnxruntime as ort
        available_providers = ort.get_available_providers()

        if provider not in available_providers:
            if "CUDAExecutionProvider" in available_providers:
                provider = "CUDAExecutionProvider"
            else:
                provider = "CPUExecutionProvider"
                self.device = "cpu"
            if self.verbose:
                print(f"Using provider: {provider}")

        provider_options = {}
        if provider == "CUDAExecutionProvider":
            provider_options = {"device_id": 0}

        # Load ONNX model
        if is_local_onnx:
            if self.verbose:
                print(f"Loading pre-converted ONNX model from {model_path}")
            self.llm = ORTModelForCausalLM.from_pretrained(
                str(model_path_obj),
                use_cache=True,
                provider=provider,
                provider_options=provider_options,
            )
        else:
            if self.verbose:
                print(f"Converting {model_path} to ONNX (this may take a while)...")
            self.llm = ORTModelForCausalLM.from_pretrained(
                model_path,
                export=True,
                use_cache=True,
                provider=provider,
                provider_options=provider_options,
            )

        # Move model to device if needed
        if device == "cuda" and hasattr(self.llm, "to"):
            self.llm = self.llm.to(device)

        # Default generation parameters (matching original llama-cpp defaults)
        self.top_p = 0.95
        self.top_k = 50
        self.temperature = 0.8
        self.max_new_tokens = 1024
        self.repetition_penalty = 1.2
        self.min_p = 0.05

        # Initialize codec with device
        self.codec = TTSCodec(device=self.device)

        # Cache EOS token ID
        self._eos_token_id = self.tokenizer.eos_token_id
        if self._eos_token_id is None:
            self._eos_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def set_params(
        self,
        top_p: float = 0.95,
        top_k: int = 50,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
    ):
        """Set sampling parameters for generation."""
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p

    def c_cache(self):
        """Clear memory cache."""
        clear_cache()

    def split_text(self, text: str) -> List[str]:
        """Split text into sentences."""
        return split_text(text)

    def encode_audio(self, audio_file: str) -> str:
        """Encode audio into context tokens."""
        context_tokens = self.codec.encode(audio_file)
        return context_tokens

    def _generate_tokens(self, formatted_prompt: str) -> str:
        """
        Generate speech tokens from a formatted prompt.

        Uses transformers' generate() method with ONNX Runtime backend.
        Supports top_k, top_p, temperature, repetition_penalty, and min_p sampling.
        """
        # Tokenize prompt (add_special_tokens=False since prompt has special tokens)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Move inputs to device if using CUDA
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Build logits processor list for min_p sampling
        logits_processor = LogitsProcessorList()
        if self.min_p > 0:
            logits_processor.append(MinPLogitsProcessor(min_p=self.min_p))

        # Generate using transformers' generate method
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self._eos_token_id,
                eos_token_id=self._eos_token_id,
                logits_processor=logits_processor,
            )

        # Extract only the generated tokens (exclude prompt)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_length:]

        # Decode with special tokens to preserve <|speech_token_XXX|> format
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return generated_text

    def generate(self, text: str, context_tokens: str) -> torch.Tensor:
        """
        Generate speech from input text.

        Args:
            text: Input text to synthesize
            context_tokens: Encoded context tokens from reference audio

        Returns:
            torch.Tensor: Audio waveform at 48kHz
        """
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)
        generated_text = self._generate_tokens(formatted_prompt)
        audio = self.codec.decode(generated_text, context_tokens)
        return audio

    def batch_generate(
        self,
        prompts: List[str],
        context_tokens: List[str],
    ) -> torch.Tensor:
        """
        Generate speech from multiple text prompts.

        Note: ONNX Runtime processes sequentially (no native batching).
        For true batching, consider using multiple ONNX sessions.

        Args:
            prompts: List of input texts
            context_tokens: List of context tokens (cycled if shorter than prompts)

        Returns:
            torch.Tensor: Concatenated audio waveforms
        """
        audios = []
        for prompt, context_token in zip(prompts, cycle(context_tokens)):
            formatted_prompt = self.codec.format_prompt(prompt, context_token, None)
            generated_text = self._generate_tokens(formatted_prompt)
            audio = self.codec.decode(generated_text, context_token)
            audios.append(audio)

        audios = torch.cat(audios, dim=0)
        return audios

    def save_onnx_model(self, output_dir: str):
        """
        Save the current ONNX model to disk.

        Useful after auto-conversion from HuggingFace model.

        Args:
            output_dir: Directory to save the ONNX model and tokenizer
        """
        self.llm.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if self.verbose:
            print(f"ONNX model saved to {output_dir}")
