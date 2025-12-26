import soundfile as sf
from mira.model import MiraTTS


def main():
    # Load from pre-converted ONNX (fast)
    # Or use 'YatharthS/MiraTTS' to auto-convert from HuggingFace
    #
    # For GPU: device="cuda" (default)
    # For CPU: device="cpu", provider="CPUExecutionProvider"
    print("Loading MiraTTS with ONNX backend...")
    tts = MiraTTS("./models/int4", device="cuda", verbose=True)
    print("Model loaded!")

    # Encode reference audio for voice cloning
    print("Encoding reference audio...")
    context = tts.encode_audio("reference.wav")
    print(f"Context tokens (first 100 chars): {context[:100]}...")

    # Generate speech
    text = "Hello! This is a test of MiraTTS with ONNX Runtime backend."
    print(f"Generating speech for: '{text}'")
    audio = tts.generate(text, context)
    print(f"Audio shape: {audio.shape}")

    # Save output
    output_path = "output_onnx.wav"
    sf.write(output_path, audio.cpu().float().numpy(), 48000)
    print(f"Audio saved to {output_path}")

    # Optional: Save ONNX model if you loaded from HuggingFace
    # tts.save_onnx_model('./mira_onnx')


if __name__ == "__main__":
    main()
