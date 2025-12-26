import re
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path
from safetensors.torch import load_file
from FastAudioSR import FASR

from ncodec.decoder.layers import (
    Snake1d,
    WNConv1d,
    ResidualUnit,
    WNConvTranspose1d,
    init_weights,
)


def remove_weight_norm_recursive(m):
    """Recursively removes weight normalization from a module."""
    try:
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            nn.utils.remove_weight_norm(m)
    except Exception:
        pass


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        kernel_sizes,
        d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class AudioTokenizer:
    """Device-aware audio tokenizer."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        model_config = {
            "input_channel": 1024,
            "channels": 1536,
            "rates": [8, 5, 4, 2],
            "kernel_sizes": [16, 11, 8, 4],
        }
        self.detokenizer = Decoder(**model_config)
        self.detokenizer.apply(remove_weight_norm_recursive)

        state_dict = load_file(model_path)
        self.detokenizer.load_state_dict(state_dict, strict=False)

        # Device-aware placement
        self.detokenizer = self.detokenizer.eval().float()
        if device == "cuda" and torch.cuda.is_available():
            self.detokenizer = self.detokenizer.to("cuda").half()
        else:
            self.detokenizer = self.detokenizer.to("cpu")

    def decode(self, x):
        return self.detokenizer(x)


class AudioDecoder:
    """Device-aware audio decoder using ONNX Runtime."""

    def __init__(self, decoder_paths: str, device: str = "cuda"):
        self.device = device
        self._torch_device = self._get_torch_device()

        sess_options = ort.SessionOptions()
        providers = self._get_providers()

        self.processor_detokenizer = ort.InferenceSession(
            f"{decoder_paths}/processer.onnx", sess_options, providers=providers
        )
        self.audio_detokenizer = AudioTokenizer(
            f"{decoder_paths}/detokenizer.safetensors", device=device
        )
        self.upsampler = FASR(f"{decoder_paths}/upsampler.pth")
        if self._torch_device == "cuda":
            _ = self.upsampler.model.half()
        else:
            # Ensure upsampler is on CPU with float32
            self.upsampler.model = self.upsampler.model.float().cpu()
            self.upsampler.device = torch.device("cpu")  # Fix internal device reference

    def _get_torch_device(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_providers(self):
        available = ort.get_available_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" in available:
            return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @torch.inference_mode()
    def detokenize(self, context_tokens: str, speech_tokens: str) -> torch.Tensor:
        """Convert tokens back to audio waveform."""
        speech_tokens_arr = (
            torch.tensor(
                [int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)]
            )
            .long()
            .unsqueeze(0)
        ).numpy()

        context_tokens_arr = (
            torch.tensor(
                [int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)]
            )
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        ).numpy().astype(np.int32)

        x = self.processor_detokenizer.run(
            ["preprocessed_output"],
            {"context_tokens": context_tokens_arr, "speech_tokens": speech_tokens_arr},
        )
        x = torch.from_numpy(x[0]).to(self._torch_device)

        # Match dtype to decoder model
        if self._torch_device == "cuda":
            x = x.half()
            lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)
            u_wav = self.upsampler.run(lowres_wav.half())
        else:
            x = x.float()
            lowres_wav = self.audio_detokenizer.decode(x).squeeze(0)
            u_wav = self.upsampler.run(lowres_wav.float())

        return u_wav
