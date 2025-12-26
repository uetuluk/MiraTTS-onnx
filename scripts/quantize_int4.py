#!/usr/bin/env python3
"""
Quantize MiraTTS ONNX model to INT4 using onnxruntime-genai.

This script converts the FP32 ONNX model to INT4 for faster inference
and reduced memory usage.

Requirements:
    pip install --pre onnxruntime-genai onnx-ir

Usage:
    python scripts/quantize_int4.py --input ./mira_onnx --output ./mira_onnx_int4
    python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./mira_onnx_int4
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import onnxruntime_genai
        print(f"onnxruntime-genai version: {onnxruntime_genai.__version__}")
    except ImportError:
        print("Error: onnxruntime-genai not installed.")
        print("Install with: pip install --pre onnxruntime-genai")
        sys.exit(1)

    try:
        import onnx_ir
        print(f"onnx-ir version: {onnx_ir.__version__}")
    except ImportError:
        print("Error: onnx-ir not installed.")
        print("Install with: pip install onnx-ir")
        sys.exit(1)


def quantize_model(
    input_path: str,
    output_path: str,
    execution_provider: str = "cpu",
    block_size: int = 128,
    accuracy_level: int = 4,
):
    """
    Quantize model to INT4 using onnxruntime-genai model builder.

    Args:
        input_path: Path to input model (local ONNX dir or HuggingFace model ID)
        output_path: Path to save quantized model
        execution_provider: Target execution provider (cpu, cuda, dml)
        block_size: Quantization block size (16, 32, 64, 128, 256)
        accuracy_level: Accuracy level 1-4 (4 = highest accuracy)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Determine if input is local path or HuggingFace model
    is_local = input_path.exists()

    # Build command
    cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-o", str(output_path),
        "-p", "int4",
        "-e", execution_provider,
        "--extra_options", f"int4_accuracy_level={accuracy_level}", f"int4_block_size={block_size}",
    ]

    if is_local:
        cmd.extend(["-m", str(input_path)])
        print(f"Quantizing local model: {input_path}")
    else:
        cmd.extend(["-m", str(input_path)])
        print(f"Quantizing HuggingFace model: {input_path}")

    print(f"Output: {output_path}")
    print(f"Execution provider: {execution_provider}")
    print(f"Block size: {block_size}")
    print(f"Accuracy level: {accuracy_level}")
    print()
    print("Running quantization...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run quantization
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Error: Quantization failed with return code {result.returncode}")
        sys.exit(1)

    print()
    print("Quantization complete!")

    # Copy config files for ORTModelForCausalLM compatibility
    print("Copying config files for ORTModelForCausalLM compatibility...")

    config_files = ["config.json", "generation_config.json"]
    source_dir = input_path if is_local else None

    # If source is HuggingFace, try to find cached config or download
    if not is_local:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(input_path))
            config.save_pretrained(str(output_path))
            print(f"  Saved config.json from HuggingFace")
        except Exception as e:
            print(f"  Warning: Could not download config: {e}")
    else:
        for config_file in config_files:
            src = source_dir / config_file
            dst = output_path / config_file
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
                print(f"  Copied {config_file}")

    # Print output size
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"\nOutput model size: {total_size / (1024**2):.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Quantize MiraTTS ONNX model to INT4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quantize from local ONNX directory
    python scripts/quantize_int4.py --input ./mira_onnx --output ./mira_onnx_int4

    # Quantize from HuggingFace model
    python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./mira_onnx_int4

    # Quantize for CUDA with custom settings
    python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./mira_onnx_int4 \\
        --execution-provider cuda --block-size 64 --accuracy-level 4
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to local ONNX model directory",
    )
    input_group.add_argument(
        "--model", "-m",
        type=str,
        help="HuggingFace model ID (e.g., YatharthS/MiraTTS)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--execution-provider", "-e",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "dml", "webgpu"],
        help="Target execution provider (default: cpu)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        choices=[16, 32, 64, 128, 256],
        help="Quantization block size (default: 128)",
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Accuracy level 1-4, higher is more accurate (default: 4)",
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Determine input path
    input_path = args.input if args.input else args.model

    # Run quantization
    quantize_model(
        input_path=input_path,
        output_path=args.output,
        execution_provider=args.execution_provider,
        block_size=args.block_size,
        accuracy_level=args.accuracy_level,
    )

    print("\nTo use the quantized model:")
    print(f"  from mira.model import MiraTTS")
    print(f"  tts = MiraTTS('{args.output}', device='cpu')")


if __name__ == "__main__":
    main()
