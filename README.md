# MiraTTS
[MiraTTS](https://huggingface.co/YatharthS/MiraTTS) is a finetune of the excellent [Spark-TTS](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) model for enhanced realism and stability performing on par with closed source models. 
This repository also heavily optimizes Mira with [Lmdeploy](https://github.com/InternLM/lmdeploy) and boosts quality by using [FlashSR](https://github.com/ysharma3501/FlashSR) to generate high quality audio at over **100x** realtime!

https://github.com/user-attachments/assets/262088ae-068a-49f2-8ad6-ab32c66dcd17

## Key benefits
- Incredibly fast: Over 100x realtime by using Lmdeploy and batching.
- High quality: Generates clear and crisp 48khz audio outputs which is much higher quality then most models.
- Memory efficient: Works within 6gb vram.
- Low latency: Latency can be low as 100ms.

## Installation

```bash
# Install from this fork (ONNX Runtime backend)
pip install git+https://github.com/uetuluk/MiraTTS-onnx.git

# Or install dependencies manually
pip install onnxruntime-genai torch soundfile
```

## Usage

### Using the INT4 Quantized Model (Recommended)

Download the pre-quantized INT4 model from HuggingFace:

```bash
# Download INT4 model (~857MB)
huggingface-cli download uetuluk2/MiraTTS-onnx-int4 --local-dir ./models/int4
```

```python
from mira.model import MiraTTS
import scipy.io.wavfile as wavfile

# Load INT4 model (857MB, 2x faster, 15x less memory)
tts = MiraTTS('./models/int4', device='cpu')

# Encode reference audio for voice cloning
context = tts.encode_audio('your_reference.wav')

# Generate speech
text = "Hello! This is a test of the INT4 quantized model."
audio = tts.generate(text, context)

# Save output
wavfile.write('output.wav', 48000, audio.cpu().numpy())
```

### Using from HuggingFace (Auto-downloads FP32 model)

```python
from mira.model import MiraTTS

# Downloads and converts model on first use (~2.5GB)
tts = MiraTTS('YatharthS/MiraTTS')

context = tts.encode_audio('reference.wav')
audio = tts.generate("Your text here", context)
```

### Batch Generation

```python
texts = ["First sentence.", "Second sentence."]
context = [tts.encode_audio('reference.wav')]

audio = tts.batch_generate(texts, context)
```

### GPU Inference

```python
# For CUDA GPU
tts = MiraTTS('./models/int4', device='cuda', provider='CUDAExecutionProvider')
```

## Quantization

To create your own INT4 quantized model:

```bash
python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./models/int4
```

See [docs/ONNX.md](docs/ONNX.md) for detailed documentation on model formats and performance.

Examples can be seen in the [huggingface model](https://huggingface.co/YatharthS/MiraTTS)

I recommend reading these 2 blogs to better easily understand LLM tts models and how I optimize them
- How they work: https://huggingface.co/blog/YatharthS/llm-tts-models
- How to optimize them: https://huggingface.co/blog/YatharthS/making-neutts-200x-realtime

## Training
Released training code! You can now train the model to be multilingual, multi-speaker, or support audio events on any local or cloud gpu!

Kaggle notebook: https://www.kaggle.com/code/yatharthsharma888/miratts-training

Colab notebook: https://colab.research.google.com/drive/1IprDyaMKaZrIvykMfNrxWFeuvj-DQPII?usp=sharing

## Next steps
- [x] Release code and model
- [x] Release training code
- [ ] Support low latency streaming
- [ ] Release native 48khz bicodec
      
## Final notes
Thanks very much to the authors of Spark-TTS and unsloth. Thanks for checking out this repository as well.

Stars would be well appreciated, thank you.

Email: yatharthsharma3501@gmail.com
