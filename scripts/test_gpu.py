import time
import psutil
import os
import torch
import soundfile as sf
from mira.model import MiraTTS


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory():
    """Get GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def main():
    print("=" * 60)
    print("MiraTTS ONNX - GPU (CUDA) Inference Test")
    print("=" * 60)

    initial_memory = get_memory_usage()
    initial_gpu = get_gpu_memory()
    print(f"\nInitial RAM usage: {initial_memory:.1f} MB")
    print(f"Initial GPU usage: {initial_gpu:.1f} MB")

    # Load model on GPU
    print("\n[1/4] Loading MiraTTS with CUDA backend...")
    load_start = time.perf_counter()
    tts = MiraTTS(
        "./models/int4",
        device="cuda",
        provider="CUDAExecutionProvider",
        verbose=True,
    )
    load_time = time.perf_counter() - load_start
    post_load_memory = get_memory_usage()
    post_load_gpu = get_gpu_memory()
    print(f"Model load time: {load_time:.2f}s")
    print(f"RAM after loading: {post_load_memory:.1f} MB (+{post_load_memory - initial_memory:.1f} MB)")
    print(f"GPU after loading: {post_load_gpu:.1f} MB (+{post_load_gpu - initial_gpu:.1f} MB)")

    # Encode reference audio
    print("\n[2/4] Encoding reference audio...")
    encode_start = time.perf_counter()
    context = tts.encode_audio("reference.wav")
    encode_time = time.perf_counter() - encode_start
    print(f"Encode time: {encode_time:.2f}s")
    print(f"Context tokens length: {len(context)} chars")

    # Generate speech
    text = "Hello! This is a test of MiraTTS running on GPU with ONNX Runtime and CUDA acceleration."
    print(f"\n[3/4] Generating speech for: '{text}'")

    generate_start = time.perf_counter()
    audio = tts.generate(text, context)
    generate_time = time.perf_counter() - generate_start

    audio_duration = len(audio) / 48000
    rtf = generate_time / audio_duration  # Real-time factor

    print(f"Generation time: {generate_time:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Real-time factor: {rtf:.2f}x (lower is better, <1 = faster than realtime)")

    # Save output
    print("\n[4/4] Saving audio...")
    output_path = "output_gpu.wav"
    sf.write(output_path, audio.cpu().float().numpy(), 48000)
    print(f"Audio saved to {output_path}")

    # Final stats
    final_memory = get_memory_usage()
    final_gpu = get_gpu_memory()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total RAM usage: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)")
    print(f"Total GPU usage: {final_gpu:.1f} MB (+{final_gpu - initial_gpu:.1f} MB)")
    print(f"Total time: {load_time + encode_time + generate_time:.2f}s")
    print(f"  - Model load: {load_time:.2f}s")
    print(f"  - Audio encode: {encode_time:.2f}s")
    print(f"  - Speech generation: {generate_time:.2f}s")
    print(f"Real-time factor: {rtf:.2f}x")


if __name__ == "__main__":
    main()
