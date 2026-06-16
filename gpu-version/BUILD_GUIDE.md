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

Use the **Offline Mode** checkbox in the web interface, or set
`HF_HUB_OFFLINE=1`, only after the model is already cached.

## Runtime Model Selection

The web interface can inspect the visible GPU, show available VRAM, and recommend
the best supported model for the automatically selected per-model GPU memory
cap. Supported runtime choices are loaded on demand and cached in the existing
Docker volumes.

Changing between supported models in the UI does **not** rebuild the Docker
image. The first generation with a new model downloads that model into the cache
volume; later runs reuse the cached files.

Rebuild the Docker image only when app code or Python dependencies change:

```bash
docker compose up -d --build
```

Current built-in runtime choices:

- `stabilityai/stable-diffusion-xl-base-1.0`: recommended quality upgrade for
  GPUs with more than 16 GiB of usable VRAM. On 16 GB-class cards, select it
  manually only with a lower internal generation limit.
- `segmind/SSD-1B`: SDXL-compatible lower-VRAM option for GPUs with around
  10 GB or more usable VRAM.
- `stabilityai/sdxl-turbo`: fast SDXL draft model that uses very few steps.
- `stable-diffusion-v1-5/stable-diffusion-v1-5`: legacy, fast, low-VRAM option.

Newer model families such as FLUX.2 or Z-Image need separate pipeline and
dependency support before they can be added safely to this app image.

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
  --build-arg MODEL_ID=stable-diffusion-v1-5/stable-diffusion-v1-5 \
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

- `MODEL_ID`: override the model, default is
  `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `MODEL_PIPELINE_TYPE`: pipeline type for a custom `MODEL_ID`; supported
  values are `sd15` and `sdxl`
- `DEFAULT_MODEL_CHOICE`: optional UI model label to select by default
- `HF_HUB_OFFLINE=1`: force cached model files only
- `PRELOAD_GPU=1`: load the GPU pipeline at app startup, default disabled so
  the UI offline switch can be selected before model loading
- `MAX_GPU_MEMORY_PERCENT=95`: highest automatic GPU memory cap allowed for the
  app, leaving headroom for driver and non-PyTorch allocations
- `MAX_GENERATION_DIMENSION=1024`: largest side used for the actual diffusion
  pass before upscaling
- `MAX_OUTPUT_DIMENSION=4096`: largest final output side exposed in the UI
- `ENABLE_ATTENTION_SLICING=1`: reduce VRAM use but usually slow generation
- `ENABLE_VAE_SLICING=1`: decode SDXL images in smaller pieces to reduce VRAM
  pressure, default enabled
- `ENABLE_VAE_TILING=1`: tile VAE decode to reduce SDXL peak VRAM, default
  enabled
- `ENABLE_TORCH_COMPILE=1`: compile the UNet; first run is slower, repeated
  same-size generations may become faster

## Speed Notes

The GPU app keeps SDXL-family models on their native scheduler and uses the DPM
multistep scheduler for SD 1.5. For most models, `16-20` steps is a good
starting range. Higher step counts increase runtime almost linearly.

The first generation after starting the container can be much slower than later
prompts because it may include model download, cache verification, and loading
the model into GPU memory. The status panel separates model load/cache time from
diffusion generation time so you can tell where the delay is happening.

SDXL at `1024x1024` is substantially heavier than SD 1.5 at `512x512`. Use SDXL
for better quality, but test prompts at fewer steps or smaller sizes when you
are iterating quickly.

For large final outputs such as `2200x1800`, keep the **Internal Generation
Limit** at `768px` first on 16 GB-class GPUs. The app will generate smaller
internally for that aspect ratio, then upscale to `2200x1800`. Raising the
internal limit toward `896px` or `1024px` improves detail but can exceed the
available VRAM during SDXL VAE decode.

For best speed:

- Use the shared model cache volume so generation does not download files.
- Keep output size at `512x512` while testing prompts.
- Leave `ENABLE_ATTENTION_SLICING=0` unless you hit VRAM limits.
- Keep `ENABLE_VAE_SLICING=1` and `ENABLE_VAE_TILING=1` for SDXL on 16 GB GPUs.
- Try `ENABLE_TORCH_COMPILE=1` only if you generate many images at the same
  size and can tolerate a slower first generation.

The app selects the PyTorch GPU memory cap automatically from the selected model.
If a model runs out of memory, choose a lighter model, reduce the output size, or
lower the **Internal Generation Limit**.

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

If an error says a process has a huge value such as `17179869184 GiB` in use,
that is a PyTorch error-message formatting issue. The number is bytes, not GiB;
`17179869184` bytes is `16 GiB`. The app normalizes this message in the UI.

The automatic GPU cap is a hard per-process limit. SDXL Base uses the highest
configured cap, while lighter models use lower caps. Avoid `100%` for normal use
because the NVIDIA driver and non-PyTorch allocations still need headroom,
especially during SDXL VAE decode.

### CUDA Device Not Ready

If generation fails with `CUDA driver error: device not ready`, the CUDA context
may be unhealthy after an earlier out-of-memory failure or a heavy SDXL decode.
The app unloads its pipelines after this error, but the most reliable recovery
is restarting the container:

```bash
docker compose restart ai-image-generator-gpu
```

For SDXL on a 16 GB GPU, keep VAE slicing and tiling enabled, and try SSD-1B,
SDXL Turbo, or Stable Diffusion 1.5 if SDXL Base keeps running out of memory.

### Model Load Fails

Check network access and the Hugging Face cache volume. If the cache is empty
and `HF_HUB_OFFLINE=1` is set, model loading will fail.

### Blank Or Black SDXL Output

If Docker logs show `invalid value encountered in cast` from Diffusers image
processing, the model likely produced NaN values during image decode. The GPU app
keeps the SDXL VAE decode path in float32 to avoid this. Rebuild the image after
pulling this code change.

### Optional Acceleration Packages

Do not add `flash-attn`, `xformers`, or `bitsandbytes` casually. They are native
packages and must match the installed PyTorch/CUDA wheel. The app works without
them by using Diffusers attention slicing.
