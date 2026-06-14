# GPU Build Guide

This image-generation app uses Stable Diffusion through Diffusers and Gradio.
The GPU image is intended for NVIDIA GPUs, including RTX 50-series Blackwell
cards such as the RTX 5080.

## Why The GPU Dockerfile Uses CUDA 12.8

RTX 50-series GPUs use compute capability `12.0` (`sm_120`). Older CUDA 12.1 /
PyTorch 2.4-era builds commonly support up to `sm_90`, which can produce errors
such as:

- `NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible`
- `no kernel image is available for execution on the device`

The Dockerfile now uses:

- `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Official PyTorch CUDA 12.8 wheels from `https://download.pytorch.org/whl/cu128`
- Runtime dependencies from `requirements-no-torch.txt`

This is much faster and less fragile than compiling PyTorch from source inside
the app image.

## Build And Run

```bash
cd gpu-version
docker compose build --no-cache
docker compose up
```

Open:

```text
http://localhost:7860
```

## Verify GPU Access

Before building the app, confirm Docker can see the NVIDIA runtime:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

After building:

```bash
cd gpu-version
docker compose run --rm ai-image-generator-gpu /app/verify-system.sh
```

Expected log details for RTX 5080:

- CUDA is available
- PyTorch CUDA runtime is `12.8`
- GPU compute capability is `12.0`

## First-Run Model Download

The first generation may take time because the model is downloaded into Docker
volumes:

- `model_cache:/app/cache`
- `huggingface_cache:/app/cache/hub`

Set `HF_HUB_OFFLINE=1` only after the model is already cached.

For gated models, set one of:

```bash
HF_TOKEN=...
HUGGING_FACE_HUB_TOKEN=...
```

## Useful Environment Variables

- `MODEL_ID`: override the model, default is `runwayml/stable-diffusion-v1-5`
- `DISABLE_CPU_FALLBACK=1`: fail instead of falling back to CPU after GPU errors
- `HF_HUB_OFFLINE=1`: force cached model files only

## Troubleshooting

### CUDA Not Available In Container

Check host driver and NVIDIA Container Toolkit:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### Out Of Memory During Generation

Start with:

- `512x512`
- `20` inference steps
- guidance `7.5`

Increase resolution only after the baseline works.

### Model Load Fails

Check network access and the Hugging Face cache volume. If the cache is empty
and `HF_HUB_OFFLINE=1` is set, model loading will fail.

### Optional Acceleration Packages

Do not add `flash-attn`, `xformers`, or `bitsandbytes` casually. They are native
packages and must match the installed PyTorch/CUDA wheel. The app works without
them by using Diffusers attention slicing.
