# Agent Notes

## Project Summary

This repository contains two Dockerized Gradio apps for Stable Diffusion image
generation:

- `gpu-version/`: NVIDIA CUDA runtime, preferred path for image generation.
- `cpu-version/`: CPU fallback path with separate dependencies.

Both apps expose Gradio on port `7860` and cache Hugging Face model files in
Docker volumes.

## GPU Runtime Notes

RTX 50-series GPUs such as the RTX 5080 are Blackwell cards with CUDA compute
capability `12.0` (`sm_120`). Do not restore older CUDA 12.1 / PyTorch 2.4
source-build assumptions or `TORCH_CUDA_ARCH_LIST` values that stop at `9.0`;
those builds can fail with compatibility or missing CUDA kernel errors.

The GPU Dockerfile installs PyTorch from the official `cu128` wheel index and
uses a CUDA 12.8 base image. Keep PyTorch installation separate from ordinary
Python requirements so the CUDA wheel index does not affect unrelated packages.

## Common Commands

From the repository root:

```bash
python3 -m py_compile gpu-version/app.py cpu-version/app.py
```

GPU build and run:

```bash
cd gpu-version
docker compose build --no-cache
docker compose up
```

CPU fallback:

```bash
cd cpu-version
docker compose up --build
```

GPU container verification:

```bash
cd gpu-version
docker compose run --rm ai-image-generator-gpu /app/verify-system.sh
```

## Model And Cache Behavior

The default SD 1.5 model is `stable-diffusion-v1-5/stable-diffusion-v1-5`.
Override it with:

```bash
MODEL_ID=some/model-id docker compose up
```

The app uses these cache paths by default:

- `HF_HOME=/app/cache`
- `HF_HUB_CACHE=/app/cache/hub`

Set `HF_HUB_OFFLINE=1` only when the model is already cached. First-run GPU
startup needs network access to download the model unless the cache volume is
pre-populated.

If using a gated Hugging Face model, pass a token through `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`.

## Debugging Checklist

- Confirm Docker can see the GPU with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`.
- Check app logs for `PyTorch CUDA runtime`, GPU name, and compute capability.
- For RTX 5080, expect compute capability `12.0`.
- If generation runs out of memory, test `512x512`, 20 steps, guidance `7.5`.
- If the model cannot load, verify the Hugging Face cache volume and network
  access before changing code.

## Editing Guidance

- Keep GPU and CPU dependency sets separate.
- Prefer Docker/runtime fixes over broad app rewrites.
- Avoid adding optional native packages such as `flash-attn`, `xformers`, or
  `bitsandbytes` unless they are pinned and verified with the active PyTorch
  CUDA wheel.
- Preserve Docker volume cache paths unless intentionally migrating cache data.
