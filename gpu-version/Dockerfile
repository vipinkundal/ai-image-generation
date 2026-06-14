# GPU Dockerfile for AI Image Generator.
#
# RTX 50-series cards are Blackwell GPUs and need CUDA/PyTorch builds that
# support sm_120. CUDA 12.1 / PyTorch 2.4-era builds only cover older
# architectures, so use official CUDA 12.8 PyTorch wheels instead of compiling
# PyTorch from source in this project image.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.title="ai-image-gpu" \
      org.opencontainers.image.description="GPU Stable Diffusion image generator" \
      org.opencontainers.image.source="gpu-version"

WORKDIR /app

ARG MODEL_ID=runwayml/stable-diffusion-v1-5
ARG DOWNLOAD_MODEL=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/app/cache \
    HF_HUB_CACHE=/app/cache/hub \
    HUGGINGFACE_HUB_CACHE=/app/cache/hub \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    MODEL_ID=${MODEL_ID} \
    PRELOAD_GPU=1 \
    GPU_MEMORY_PERCENT=80 \
    ENABLE_ATTENTION_SLICING=0 \
    ENABLE_TORCH_COMPILE=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_CUDNN_V8_API_ENABLED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch separately so its CUDA wheel index does not affect ordinary
# Python package resolution.
RUN python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements-no-torch.txt .
RUN python3 -m pip install -r requirements-no-torch.txt

COPY download_model.py .
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then python3 download_model.py; fi

COPY app.py .

RUN mkdir -p /app/cache /app/cache/hub && \
    printf '%s\n' \
      '#!/bin/bash' \
      'set -e' \
      'echo "GPU and PyTorch verification"' \
      'python3 - <<'"'"'PY'"'"'' \
      'import sys' \
      'import torch' \
      'print(f"Python: {sys.version.split()[0]}")' \
      'print(f"PyTorch: {torch.__version__}")' \
      'print(f"PyTorch CUDA runtime: {torch.version.cuda}")' \
      'print(f"CUDA available: {torch.cuda.is_available()}")' \
      'if torch.cuda.is_available():' \
      '    for idx in range(torch.cuda.device_count()):' \
      '        props = torch.cuda.get_device_properties(idx)' \
      '        capability = ".".join(map(str, torch.cuda.get_device_capability(idx)))' \
      '        print(f"GPU {idx}: {props.name} ({props.total_memory / 1024**3:.1f} GB, sm_{capability.replace(chr(46), chr(95))})")' \
      '    x = torch.randn(16, 16, device="cuda")' \
      '    print(f"CUDA test tensor mean: {x.mean().item():.4f}")' \
      'else:' \
      '    raise SystemExit("CUDA is not available inside the container")' \
      'PY' \
      > /app/verify-system.sh && chmod +x /app/verify-system.sh

EXPOSE 7860

CMD ["/bin/bash", "-c", "/app/verify-system.sh && python3 -u app.py"]
