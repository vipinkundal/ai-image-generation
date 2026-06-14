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
docker compose up -d --build
```

The Compose file uses stable names:

- Image: `ai-image-gpu:latest`
- Container: `ai-image-gpu`

If you already have the old auto-named container, remove it once before using
the stable name:

```bash
docker compose down
docker compose up -d --build
```

Normal rebuilds will reuse Docker's build cache. Avoid `--no-cache` unless you
are intentionally rebuilding every dependency layer.

Docker keeps old untagged image IDs after rebuilds. To remove this app's old
dangling images and the previous auto-named container, run:

```bash
./clean-docker-images.sh
```

If you prefer plain `docker run`, use the fixed-name script instead of Docker
Desktop's image run button:

```bash
./run-gpu-container.sh
```

That script always starts image `ai-image-gpu:latest` as container
`ai-image-gpu`.

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

### Preload The Model Into A Docker Volume

If you run the app with plain `docker run --rm` and no volume, the downloaded
model is deleted when the container exits. Use a named volume to keep the model
cache across runs:

```bash
docker volume create ai-image-model-cache
docker run --rm -v ai-image-model-cache:/app/cache ai-image-gpu python3 /app/download_model.py
docker run -d --name ai-image-gpu --gpus all -p 7860:7860 -v ai-image-model-cache:/app/cache ai-image-gpu:latest
```

The first command downloads the model into the volume. The second command starts
the app with that same cache mounted, so generation loads local files instead of
fetching model files from Hugging Face.

### Bake The Model Into The Image

You can also build a self-contained image with the model already downloaded:

```bash
docker build \
  --build-arg DOWNLOAD_MODEL=true \
  --build-arg MODEL_ID=runwayml/stable-diffusion-v1-5 \
  -t ai-image-gpu .
```

Do not mount an empty `/app/cache` volume over a baked image cache, because the
volume hides the files stored in the image layer.

For gated models, set one of:

```bash
HF_TOKEN=...
HUGGING_FACE_HUB_TOKEN=...
```

## Useful Environment Variables

- `MODEL_ID`: override the model, default is `runwayml/stable-diffusion-v1-5`
- `HF_HUB_OFFLINE=1`: force cached model files only
- `PRELOAD_GPU=1`: load the GPU pipeline at app startup, default enabled
- `GPU_MEMORY_PERCENT=80`: default PyTorch GPU memory limit shown in the UI
- `MAX_GENERATION_DIMENSION=1024`: largest side used for the actual diffusion
  pass before upscaling
- `MAX_OUTPUT_DIMENSION=4096`: largest final output side exposed in the UI
- `ENABLE_ATTENTION_SLICING=1`: reduce VRAM use but usually slow generation
- `ENABLE_TORCH_COMPILE=1`: compile the UNet; first run is slower, repeated
  same-size generations may become faster

## Speed Notes

The GPU app uses a DPM multistep scheduler by default, so `16-20` steps is a
good starting range for `512x512`. Higher step counts increase runtime almost
linearly.

For best speed:

- Use the shared model cache volume so generation does not download files.
- Keep output size at `512x512` while testing prompts.
- Leave `ENABLE_ATTENTION_SLICING=0` unless you hit VRAM limits.
- Try `ENABLE_TORCH_COMPILE=1` only if you generate many images at the same
  size and can tolerate a slower first generation.

The GPU memory slider limits how much visible VRAM PyTorch may use. It does not
pre-allocate VRAM and higher values are not automatically faster, but raising it
can help prevent out-of-memory errors when using larger images or heavier
settings.

## 4K Output

The app can output up to `4096px` on either side. For example, use
`3840x2160` for 4K UHD.

Stable Diffusion v1.5 is not designed to directly generate huge 4K latents.
Instead, the app generates at a safer internal size, capped by
`MAX_GENERATION_DIMENSION`, and then upscales the final image to the requested
output pixels. This keeps GPU memory use realistic while still producing a 4K
image file.

If you raise `MAX_GENERATION_DIMENSION`, quality may improve for large outputs,
but runtime and VRAM use increase quickly.

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

Increase output resolution only after the baseline works. For 4K output, keep
`MAX_GENERATION_DIMENSION=1024` first, then raise it only if the GPU has enough
VRAM.

### Model Load Fails

Check network access and the Hugging Face cache volume. If the cache is empty
and `HF_HUB_OFFLINE=1` is set, model loading will fail.

### Optional Acceleration Packages

Do not add `flash-attn`, `xformers`, or `bitsandbytes` casually. They are native
packages and must match the installed PyTorch/CUDA wheel. The app works without
them by using Diffusers attention slicing.
