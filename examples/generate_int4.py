#!/usr/bin/env python3
"""
Example: Generate audio using the INT4 quantized MiraTTS model.

This example demonstrates how to use the quantized model for
faster and more memory-efficient TTS inference.

Usage:
    python examples/generate_int4.py --text "Hello, world!" --reference reference.wav --output output.wav
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import scipy.io.wavfile as wavfile
import torch

from mira.model import MiraTTS


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio using INT4 quantized MiraTTS model"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="Hello! This is a test of the INT4 quantized MiraTTS model.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        default="reference.wav",
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_int4.wav",
        help="Output audio file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="./models/int4",
        help="Path to INT4 quantized model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    args = parser.parse_args()

    # Check reference audio exists
    if not Path(args.reference).exists():
        print(f"Error: Reference audio not found: {args.reference}")
        sys.exit(1)

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Run quantization first: python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./mira_onnx_genai_int4")
        sys.exit(1)

    # Set provider based on device
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"

    # Load model
    print(f"Loading INT4 model from {args.model}...")
    start = time.time()
    tts = MiraTTS(args.model, device=args.device, provider=provider)
    print(f"Model loaded in {time.time() - start:.2f}s")

    # Encode reference audio
    print(f"Encoding reference audio: {args.reference}")
    context = tts.encode_audio(args.reference)

    # Generate audio
    print(f"Generating: {args.text}")
    start = time.time()
    audio = tts.generate(args.text, context)
    gen_time = time.time() - start

    # Calculate stats
    audio_duration = audio.shape[0] / 48000
    rtf = gen_time / audio_duration

    print(f"Generated {audio_duration:.2f}s of audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")

    # Save output
    wavfile.write(args.output, 48000, audio.cpu().numpy())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
