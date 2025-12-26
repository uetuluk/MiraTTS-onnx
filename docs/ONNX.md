# ONNX Model Documentation

This document covers the ONNX model formats available for MiraTTS and how to use them.

## Model Formats

### INT4 Quantized (Recommended)

The INT4 quantized model offers the best balance of performance and quality:

| Property | Value |
|----------|-------|
| Size | ~857 MB |
| Location | `models/int4/` |
| Precision | 4-bit weights |
| Speed | ~2x faster than FP32 |
| Memory | ~15x less than FP32 |

**Usage:**
```python
from mira.model import MiraTTS

tts = MiraTTS('./models/int4', device='cpu')
```

### FP32 from HuggingFace

The full-precision FP32 model can be downloaded from HuggingFace on first use:

| Property | Value |
|----------|-------|
| Size | ~2.5 GB |
| Source | `YatharthS/MiraTTS` |
| Precision | 32-bit floating point |

**Usage:**
```python
from mira.model import MiraTTS

# Downloads and converts model on first use
tts = MiraTTS('YatharthS/MiraTTS')
```

## Performance Comparison

| Model | Size | Load Time | Generation Speed | Memory Usage |
|-------|------|-----------|------------------|--------------|
| INT4 | 857 MB | ~2s | ~2x baseline | ~400 MB |
| FP32 | 2.5 GB | ~5s | 1x baseline | ~6 GB |

*Benchmarks on CPU. GPU performance varies by hardware.*

## Quantization

### Creating INT4 Model

To create your own INT4 quantized model from the HuggingFace model:

```bash
python scripts/quantize_int4.py --model YatharthS/MiraTTS --output ./models/int4
```

### Quantization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--execution-provider` | `cpu` | Target device: `cpu`, `cuda`, `dml`, `webgpu` |
| `--block-size` | `128` | Quantization block size: 16, 32, 64, 128, 256 |
| `--accuracy-level` | `4` | Accuracy level 1-4 (4 = highest) |

**Example with custom settings:**
```bash
python scripts/quantize_int4.py \
    --model YatharthS/MiraTTS \
    --output ./models/int4_cuda \
    --execution-provider cuda \
    --block-size 64 \
    --accuracy-level 4
```

### Requirements

```bash
pip install --pre onnxruntime-genai onnx-ir
```

## Device Configuration

### CPU Inference

```python
tts = MiraTTS('./models/int4', device='cpu')
```

### GPU Inference (CUDA)

```python
tts = MiraTTS('./models/int4', device='cuda', provider='CUDAExecutionProvider')
```

### GPU Inference (DirectML - Windows)

```python
tts = MiraTTS('./models/int4', device='cuda', provider='DmlExecutionProvider')
```

## Troubleshooting

### Model not found error

Ensure the model path contains:
- `model.onnx` or `model.onnx.data`
- `config.json`
- `generation_config.json`
- `genai_config.json`

### Out of memory

Try the INT4 model which uses ~15x less memory than FP32.

### Slow inference on GPU

Ensure you have:
1. CUDA toolkit installed (for NVIDIA GPUs)
2. Correct ONNX Runtime package: `onnxruntime-gpu` instead of `onnxruntime`
3. Specified the correct execution provider

### Quantization fails

1. Install dependencies: `pip install --pre onnxruntime-genai onnx-ir`
2. Ensure sufficient disk space (~3GB for intermediate files)
3. Check Python version compatibility (3.10+ recommended)
