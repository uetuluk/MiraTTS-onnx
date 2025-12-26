import time
import psutil
import os
import soundfile as sf
from mira.model import MiraTTS


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    print("=" * 60)
    print("MiraTTS ONNX - CPU Inference Test")
    print("=" * 60)

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.1f} MB")

    # Load model on CPU
    print("\n[1/4] Loading MiraTTS with CPU backend...")
    load_start = time.perf_counter()
    tts = MiraTTS(
        "./models/int4",
        device="cpu",
        provider="CPUExecutionProvider",
        verbose=True,
    )
    load_time = time.perf_counter() - load_start
    post_load_memory = get_memory_usage()
    print(f"Model load time: {load_time:.2f}s")
    print(f"Memory after loading: {post_load_memory:.1f} MB (+{post_load_memory - initial_memory:.1f} MB)")

    # Encode reference audio
    print("\n[2/4] Encoding reference audio...")
    encode_start = time.perf_counter()
    context = tts.encode_audio("reference.wav")
    encode_time = time.perf_counter() - encode_start
    print(f"Encode time: {encode_time:.2f}s")
    print(f"Context tokens length: {len(context)} chars")

    # Generate speech
    text = "Hello! This is a test of MiraTTS running on CPU with ONNX Runtime."
    print(f"\n[3/4] Generating speech for: '{text}'")

    # Use fewer tokens for CPU (faster test)
    tts.max_new_tokens = 512

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
    output_path = "output_cpu.wav"
    sf.write(output_path, audio.cpu().float().numpy(), 48000)
    print(f"Audio saved to {output_path}")

    # Final stats
    final_memory = get_memory_usage()
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total memory usage: {final_memory:.1f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
    print(f"Total time: {load_time + encode_time + generate_time:.2f}s")
    print(f"  - Model load: {load_time:.2f}s")
    print(f"  - Audio encode: {encode_time:.2f}s")
    print(f"  - Speech generation: {generate_time:.2f}s")


if __name__ == "__main__":
    main()
