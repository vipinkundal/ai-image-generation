"""
Simple AI Image Generator.

GPU is preferred when CUDA is available. CPU runs only when explicitly selected.
"""

import gc
import os
import re
import threading
import time
import traceback
from collections import OrderedDict

# Set Hugging Face cache directories before importing diffusers/transformers.
os.environ.setdefault("HF_HOME", "/app/cache")
os.environ.setdefault("HF_HUB_CACHE", "/app/cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/cache/hub")

import gradio as gr
import torch
from diffusers import (
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image, ImageDraw, ImageFont, ImageStat


MODEL_ID = os.getenv("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
MODEL_PIPELINE_TYPE = os.getenv("MODEL_PIPELINE_TYPE", "sd15").lower()
CACHE_DIR = os.getenv("HF_HOME", "/app/cache")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}
PRELOAD_GPU = os.getenv("PRELOAD_GPU", "0").lower() in {"1", "true", "yes"}
ENABLE_ATTENTION_SLICING = os.getenv("ENABLE_ATTENTION_SLICING", "0").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_VAE_SLICING = os.getenv("ENABLE_VAE_SLICING", "1").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_VAE_TILING = os.getenv("ENABLE_VAE_TILING", "1").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "0").lower() in {
    "1",
    "true",
    "yes",
}
MAX_GPU_MEMORY_PERCENT = min(100, max(40, int(os.getenv("MAX_GPU_MEMORY_PERCENT", "95"))))
MAX_GENERATION_DIMENSION = int(os.getenv("MAX_GENERATION_DIMENSION", "1024"))
MAX_OUTPUT_DIMENSION = int(os.getenv("MAX_OUTPUT_DIMENSION", "4096"))

CUDA_AVAILABLE = torch.cuda.is_available()
gpu_pipeline = None
cpu_pipeline = None
gpu_pipeline_choice = None
cpu_pipeline_choice = None
current_gpu_memory_fraction = None

MODEL_OPTIONS = OrderedDict(
    [
        (
            "SDXL Base 1.0 (high VRAM)",
            {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "pipeline_type": "sdxl",
                "model_family": "sdxl",
                "min_vram_gb": 10,
                "recommended_vram_gb": 16,
                "gpu_memory_percent": 95,
                "use_dpm_scheduler": False,
                "default_steps": 24,
                "default_guidance": 6.5,
                "default_size": 1024,
                "default_generation_limit": 768,
                "description": "Highest-quality bundled model, but heavy on 16 GB-class GPUs.",
            },
        ),
        (
            "SSD-1B (SDXL-compatible, lower VRAM)",
            {
                "model_id": "segmind/SSD-1B",
                "pipeline_type": "sdxl",
                "model_family": "sdxl",
                "min_vram_gb": 8,
                "recommended_vram_gb": 10,
                "gpu_memory_percent": 90,
                "use_dpm_scheduler": False,
                "default_steps": 20,
                "default_guidance": 7.0,
                "default_size": 768,
                "default_generation_limit": 768,
                "description": "Smaller SDXL-style model for tighter VRAM budgets.",
            },
        ),
        (
            "SDXL Turbo (fast, low steps)",
            {
                "model_id": "stabilityai/sdxl-turbo",
                "pipeline_type": "auto_text2image",
                "model_family": "sdxl",
                "min_vram_gb": 8,
                "recommended_vram_gb": 10,
                "gpu_memory_percent": 90,
                "use_dpm_scheduler": False,
                "default_steps": 4,
                "default_guidance": 0.0,
                "default_size": 768,
                "default_generation_limit": 768,
                "description": "Fast SDXL variant for quick drafts with very few steps.",
            },
        ),
        (
            "Stable Diffusion 1.5 (legacy, low VRAM)",
            {
                "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "pipeline_type": "sd15",
                "model_family": "sd15",
                "min_vram_gb": 4,
                "recommended_vram_gb": 6,
                "gpu_memory_percent": 80,
                "use_dpm_scheduler": True,
                "default_steps": 20,
                "default_guidance": 7.5,
                "default_size": 512,
                "default_generation_limit": 768,
                "description": "Older model, fastest and lightest option.",
            },
        ),
    ]
)

if MODEL_ID not in {option["model_id"] for option in MODEL_OPTIONS.values()}:
    MODEL_OPTIONS[f"Configured MODEL_ID ({MODEL_PIPELINE_TYPE})"] = {
        "model_id": MODEL_ID,
        "pipeline_type": MODEL_PIPELINE_TYPE,
        "model_family": MODEL_PIPELINE_TYPE,
        "min_vram_gb": 4 if MODEL_PIPELINE_TYPE == "sd15" else 10,
        "recommended_vram_gb": 6 if MODEL_PIPELINE_TYPE == "sd15" else 16,
        "gpu_memory_percent": 80 if MODEL_PIPELINE_TYPE == "sd15" else 95,
        "use_dpm_scheduler": MODEL_PIPELINE_TYPE == "sd15",
        "default_steps": 20,
        "default_guidance": 7.5,
        "default_size": 512,
        "default_generation_limit": 768,
        "description": "Custom model from MODEL_ID. Set MODEL_PIPELINE_TYPE=sd15 or sdxl.",
    }

SUPPORTED_PIPELINES = {
    "auto_text2image": AutoPipelineForText2Image,
    "sd15": StableDiffusionPipeline,
    "sdxl": StableDiffusionXLPipeline,
}

REQUESTED_DEFAULT_MODEL_CHOICE = os.getenv("DEFAULT_MODEL_CHOICE")
DEFAULT_MODEL_CHOICE = None

if CUDA_AVAILABLE:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def round_down_to_multiple(value, multiple):
    return max(multiple, int(value) // multiple * multiple)


def calculate_generation_size(output_width, output_height, generation_limit=None):
    """Pick a diffusion size that preserves aspect ratio and can be upscaled."""
    configured_limit = MAX_GENERATION_DIMENSION if generation_limit is None else int(generation_limit)
    max_generation_dimension = clamp(configured_limit, 256, MAX_OUTPUT_DIMENSION)
    largest_output_dimension = max(output_width, output_height)

    if largest_output_dimension <= max_generation_dimension:
        return output_width, output_height, False

    scale = max_generation_dimension / largest_output_dimension
    generation_width = round_down_to_multiple(output_width * scale, 8)
    generation_height = round_down_to_multiple(output_height * scale, 8)

    return generation_width, generation_height, True


def upscale_image(image, output_width, output_height):
    if image.size == (output_width, output_height):
        return image

    return image.resize((output_width, output_height), resample=Image.Resampling.LANCZOS)


def format_seconds(seconds):
    seconds = int(max(0, seconds))
    minutes, seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(text, max_chars):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def progress_image(title, lines, width=1024, height=1024):
    image = Image.new("RGB", (width, height), color=(18, 29, 44))
    draw = ImageDraw.Draw(image)
    accent = (99, 91, 255)
    text = (238, 244, 255)
    muted = (180, 194, 214)
    title_font = font(34)
    body_font = font(24)

    draw.rectangle((0, 0, width, 104), fill=(31, 45, 65))
    draw.rectangle((40, 30, 58, 72), fill=accent)
    draw.text((82, 30), title, fill=text, font=title_font)

    y = 150
    for index, line in enumerate(lines):
        fill = text if index == 0 else muted
        for wrapped_line in wrap_text(line, 72):
            draw.text((60, y), wrapped_line, fill=fill, font=body_font)
            y += 36
        y += 10

    return image


def image_looks_blank(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.mean[0] < 3 and stat.extrema[0][1] < 12


def normalize_oom_message(message):
    """Clean up PyTorch OOM text that sometimes labels byte counts as GiB."""
    normalized = message
    notes = []

    def replace_large_gib(match):
        raw_value = float(match.group(1))
        if raw_value < 1024:
            return match.group(0)

        gib_value = raw_value / 1024**3
        notes.append(
            f"PyTorch reported {raw_value:.0f} GiB, but that value looks like bytes "
            f"({gib_value:.2f} GiB)."
        )
        return f"this process has about {gib_value:.2f} GiB memory in use"

    normalized = re.sub(
        r"this process has ([0-9]+(?:\.[0-9]+)?) GiB memory in use",
        replace_large_gib,
        normalized,
    )
    if notes:
        normalized += "\n" + "\n".join(notes)
    return normalized


def model_gpu_memory_percent(config):
    return clamp(int(config.get("gpu_memory_percent", MAX_GPU_MEMORY_PERCENT)), 40, MAX_GPU_MEMORY_PERCENT)


def gpu_limit_gb(limit_percent):
    if not CUDA_AVAILABLE:
        return 0
    props = torch.cuda.get_device_properties(0)
    return (props.total_memory / 1024**3) * (clamp(int(limit_percent), 40, MAX_GPU_MEMORY_PERCENT) / 100)


def lighter_model_choices(config):
    alternatives = []
    for choice, option in MODEL_OPTIONS.items():
        if option["pipeline_type"] not in SUPPORTED_PIPELINES:
            continue
        if option["recommended_vram_gb"] < config["recommended_vram_gb"]:
            alternatives.append(choice)
    return alternatives[:3]


def model_memory_tip(model_choice, config, generation_width, generation_height):
    percent = model_gpu_memory_percent(config)
    limit_gb = gpu_limit_gb(percent)
    tips = [
        (
            f"The selected model uses an automatic GPU memory cap of {percent}% "
            f"({limit_gb:.2f} GiB on this GPU)."
        )
    ]

    alternatives = [choice for choice in lighter_model_choices(config) if choice != model_choice]
    if alternatives:
        tips.append("If this model keeps failing, select a lighter model: " + ", ".join(alternatives) + ".")

    if config.get("model_family") == "sdxl":
        tips.append(
            f"For {generation_width}x{generation_height} internal generation, lower the "
            "Internal Generation Limit or output size before retrying this SDXL model."
        )

    return "\n" + "\n".join(tips)


def cuda_runtime_tip(error):
    error_text = str(error).lower()
    if "device not ready" not in error_text and "cuda driver error" not in error_text:
        return ""

    unload_pipelines(release_cuda_cache=True)
    return (
        "\n\nCUDA recovery note: the GPU driver reported a device-level error. "
        "This can happen after an earlier out-of-memory failure or during a heavy "
        "SDXL VAE decode. The app unloaded its pipelines, but if the next run fails "
        "again, restart the container with `docker compose restart ai-image-generator-gpu`."
    )


def configure_gpu_memory_limit(percent):
    """Set the PyTorch CUDA allocator limit for this process."""
    global current_gpu_memory_fraction

    if not CUDA_AVAILABLE:
        return "CUDA is not available."

    percent = clamp(int(percent), 40, MAX_GPU_MEMORY_PERCENT)
    fraction = percent / 100

    if current_gpu_memory_fraction != fraction:
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        current_gpu_memory_fraction = fraction
        clear_memory(release_cuda_cache=True)
        print(f"GPU memory limit set to {percent}% of visible GPU memory")

    return gpu_memory_status(percent)


def gpu_memory_status(limit_percent=None):
    if not CUDA_AVAILABLE:
        return "GPU memory: CUDA is not available."

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3

    if limit_percent is None:
        limit_percent = int((current_gpu_memory_fraction or 1.0) * 100)

    limit_gb = total * (int(limit_percent) / 100)
    return (
        "GPU memory: "
        f"{allocated:.2f} GiB allocated, "
        f"{reserved:.2f} GiB reserved, "
        f"{limit_gb:.2f} GiB limit ({int(limit_percent)}% of {total:.2f} GiB visible)."
    )


def gpu_inventory_status():
    if not CUDA_AVAILABLE:
        return "GPU: CUDA is not available in this container."

    lines = ["Detected GPU configuration:"]
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        if idx == torch.cuda.current_device():
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            free_gb = free_bytes / 1024**3
            total_gb = total_bytes / 1024**3
            free_detail = f", {free_gb:.2f} GiB currently free"
        else:
            free_detail = ""
        limit_gb = total_gb * (MAX_GPU_MEMORY_PERCENT / 100)
        capability = ".".join(map(str, torch.cuda.get_device_capability(idx)))
        lines.append(
            f"- GPU {idx}: {props.name}, {total_gb:.2f} GiB visible VRAM"
            f"{free_detail}, max automatic app cap {limit_gb:.2f} GiB "
            f"({MAX_GPU_MEMORY_PERCENT}%), sm_{capability.replace('.', '_')}"
        )
    return "\n".join(lines)


def model_config(model_choice):
    try:
        return MODEL_OPTIONS[model_choice]
    except KeyError as exc:
        raise ValueError(f"Unknown model choice: {model_choice}") from exc


def preferred_model_for_visible_vram(visible_vram_gb):
    supported = [
        (choice, config)
        for choice, config in MODEL_OPTIONS.items()
        if config["pipeline_type"] in SUPPORTED_PIPELINES
        and visible_vram_gb * (model_gpu_memory_percent(config) / 100) >= config["min_vram_gb"]
    ]
    if not supported:
        return "Stable Diffusion 1.5 (legacy, low VRAM)"

    recommended = [
        (choice, config)
        for choice, config in supported
        if visible_vram_gb * (model_gpu_memory_percent(config) / 100) >= config["recommended_vram_gb"]
    ]
    return (recommended or supported)[0][0]


def resolve_default_model_choice():
    if REQUESTED_DEFAULT_MODEL_CHOICE in MODEL_OPTIONS:
        return REQUESTED_DEFAULT_MODEL_CHOICE

    if CUDA_AVAILABLE:
        props = torch.cuda.get_device_properties(0)
        return preferred_model_for_visible_vram(props.total_memory / 1024**3)

    return "Stable Diffusion 1.5 (legacy, low VRAM)"


DEFAULT_MODEL_CHOICE = resolve_default_model_choice()


def describe_model_options(device_choice):
    if device_choice != "GPU":
        preferred_choice = "Stable Diffusion 1.5 (legacy, low VRAM)"
        return (
            gr.update(
                label="CPU And Model Recommendations",
                value=(
                    "CPU selected. VRAM is not used in CPU mode, so GPU memory "
                    "availability does not apply.\n\n"
                    "Generation will use system RAM and will be much slower. "
                    "Use Stable Diffusion 1.5 for CPU tests."
                ),
            ),
            gr.update(value=preferred_choice),
            gr.update(value=selected_model_status(preferred_choice)),
        )

    if not CUDA_AVAILABLE:
        preferred_choice = "Stable Diffusion 1.5 (legacy, low VRAM)"
        return (
            gr.update(
                label="GPU And Model Recommendations",
                value=(
                    "GPU selected, but CUDA is not available in this container. "
                    "Check Docker GPU access."
                ),
            ),
            gr.update(value=preferred_choice),
            gr.update(value=selected_model_status(preferred_choice)),
        )

    props = torch.cuda.get_device_properties(0)
    visible_vram_gb = props.total_memory / 1024**3
    preferred_choice = preferred_model_for_visible_vram(visible_vram_gb)

    lines = [
        gpu_inventory_status(),
        "",
        "Supported runtime model choices:",
    ]
    for choice, config in MODEL_OPTIONS.items():
        if config["pipeline_type"] not in SUPPORTED_PIPELINES:
            continue

        model_percent = model_gpu_memory_percent(config)
        model_limit_gb = visible_vram_gb * (model_percent / 100)
        if model_limit_gb >= config["recommended_vram_gb"]:
            status = "recommended"
        elif model_limit_gb >= config["min_vram_gb"]:
            status = "available, but keep sizes conservative"
        else:
            status = "not recommended for this GPU"

        marker = " *" if choice == preferred_choice else ""
        lines.append(
            f"- {choice}{marker}: {status}. "
            f"Auto GPU cap {model_percent}% ({model_limit_gb:.2f} GiB). "
            f"{config['description']} Model: {config['model_id']}"
        )

    lines.extend(
        [
            "",
            "Selecting a supported model downloads it into the cache volume on first use. "
            "A Docker rebuild is only needed after app code or dependency changes.",
        ]
    )
    return (
        gr.update(label="GPU And Model Recommendations", value="\n".join(lines)),
        gr.update(value=preferred_choice),
        gr.update(value=selected_model_status(preferred_choice)),
    )


def initial_model_recommendation_text(device_choice):
    recommendation_update = describe_model_options(device_choice)[0]
    if isinstance(recommendation_update, dict):
        return recommendation_update.get("value", "")
    return str(recommendation_update)


def selected_model_status(model_choice):
    config = model_config(model_choice)
    gpu_cap = model_gpu_memory_percent(config)
    return (
        f"Selected model: {config['model_id']}\n"
        f"Pipeline: {config['pipeline_type']}\n"
        f"Automatic GPU memory cap: {gpu_cap}%\n"
        f"{config['description']}"
    )


def model_generation_defaults(model_choice):
    config = model_config(model_choice)
    size = min(config["default_size"], MAX_OUTPUT_DIMENSION)
    generation_limit = min(config["default_generation_limit"], MAX_GENERATION_DIMENSION)
    return (
        selected_model_status(model_choice),
        gr.update(value=config["default_steps"]),
        gr.update(value=config["default_guidance"]),
        gr.update(value=size),
        gr.update(value=size),
        gr.update(value=generation_limit),
    )


def cuda_version_tuple():
    if not torch.version.cuda:
        return (0, 0)
    parts = torch.version.cuda.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return (0, 0)


def log_system_info():
    """Print useful runtime details for Docker logs."""
    print("AI Image Generator starting")
    print(f"Configured MODEL_ID: {MODEL_ID}")
    print(f"Default model choice: {DEFAULT_MODEL_CHOICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"HF_HOME: {CACHE_DIR}")
    print(f"Offline model loading: {OFFLINE_MODE}")

    if CUDA_AVAILABLE:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            capability = torch.cuda.get_device_capability(idx)
            print(
                "GPU "
                f"{idx}: {props.name}, "
                f"{props.total_memory / 1024**3:.1f} GB, "
                f"compute capability {capability[0]}.{capability[1]}"
            )

            if capability[0] >= 12 and cuda_version_tuple() < (12, 8):
                print(
                    "WARNING: This GPU is Blackwell/RTX 50-series class, but "
                    f"PyTorch reports CUDA {torch.version.cuda}. Use a CUDA 12.8+ "
                    "or CUDA 13+ PyTorch build for reliable GPU generation."
                )
    else:
        print("Running without CUDA. Select CPU mode or fix GPU access.")


def check_cache_status():
    """Log cache directory status to make first-run downloads obvious."""
    for cache_dir in {CACHE_DIR, os.getenv("HF_HUB_CACHE", "/app/cache/hub")}:
        print(f"Checking cache: {cache_dir}")
        if not os.path.exists(cache_dir):
            print("  Missing")
            continue

        try:
            items = os.listdir(cache_dir)
            total_size = 0
            for root, _, files in os.walk(cache_dir):
                for filename in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, filename))
                    except OSError:
                        pass
            print(f"  Items: {len(items)}")
            print(f"  Size: {total_size / 1024**3:.2f} GB")
        except OSError as exc:
            print(f"  Could not read cache: {exc}")


def clear_memory(release_cuda_cache=False):
    gc.collect()
    if release_cuda_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            print(f"Could not clear CUDA cache: {exc}")


def unload_pipelines(release_cuda_cache=True):
    global gpu_pipeline, cpu_pipeline, gpu_pipeline_choice, cpu_pipeline_choice

    gpu_pipeline = None
    cpu_pipeline = None
    gpu_pipeline_choice = None
    cpu_pipeline_choice = None
    clear_memory(release_cuda_cache=release_cuda_cache)


def set_offline_mode(enabled):
    """Switch Hugging Face model loading between online and cached-only modes."""
    global OFFLINE_MODE, gpu_pipeline, cpu_pipeline, gpu_pipeline_choice, cpu_pipeline_choice

    enabled = bool(enabled)
    if OFFLINE_MODE == enabled:
        return

    OFFLINE_MODE = enabled
    os.environ["HF_HUB_OFFLINE"] = "1" if enabled else "0"
    print(f"Offline model loading set to: {OFFLINE_MODE}")

    if gpu_pipeline is not None or cpu_pipeline is not None:
        print("Offline mode changed. Unloading pipelines so the next load uses the selected mode.")
        gpu_pipeline = None
        cpu_pipeline = None
        gpu_pipeline_choice = None
        cpu_pipeline_choice = None
        clear_memory(release_cuda_cache=True)


def build_pipeline(model_choice, device):
    """Load Stable Diffusion for the requested device."""
    config = model_config(model_choice)
    pipeline_type = config["pipeline_type"]
    pipeline_class = SUPPORTED_PIPELINES.get(pipeline_type)
    if pipeline_class is None:
        raise ValueError(
            f"Pipeline type '{pipeline_type}' is not supported by this app image. "
            "Supported built-in types: sd15, sdxl, auto_text2image."
        )

    is_gpu = device == "cuda"
    dtype = torch.float16 if is_gpu else torch.float32

    print(f"Loading {config['model_id']} ({pipeline_type}) on {device}")
    load_kwargs = {
        "pretrained_model_name_or_path": config["model_id"],
        "dtype": dtype,
        "use_safetensors": True,
        "cache_dir": CACHE_DIR,
        "token": HF_TOKEN,
        "local_files_only": OFFLINE_MODE,
    }
    if pipeline_type == "sd15":
        load_kwargs.update(
            {
                "safety_checker": None,
                "requires_safety_checker": False,
            }
        )

    pipeline = pipeline_class.from_pretrained(**load_kwargs)

    if is_gpu and config.get("use_dpm_scheduler", True) and hasattr(pipeline, "scheduler"):
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

    pipeline = pipeline.to(device)

    if is_gpu and config.get("model_family") == "sdxl" and hasattr(pipeline, "vae"):
        pipeline.vae.to(dtype=torch.float32)
        pipeline.vae.config.force_upcast = True
        print("SDXL VAE decode set to float32 to avoid blank/NaN images")

    if is_gpu and ENABLE_VAE_SLICING and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print("VAE slicing enabled")

    if is_gpu and ENABLE_VAE_TILING and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        print("VAE tiling enabled")

    if is_gpu and hasattr(pipeline, "unet"):
        pipeline.unet.to(memory_format=torch.channels_last)

    if ENABLE_ATTENTION_SLICING:
        pipeline.enable_attention_slicing()
        print("Attention slicing enabled")

    if is_gpu and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("xFormers attention enabled")
        except Exception as exc:
            print(f"xFormers attention unavailable, continuing without it: {exc}")

    if is_gpu and ENABLE_TORCH_COMPILE and hasattr(torch, "compile") and hasattr(pipeline, "unet"):
        print("Compiling UNet. First generation will be slower; later generations may be faster.")
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=False)

    return pipeline


def load_gpu_pipeline(model_choice):
    global gpu_pipeline, gpu_pipeline_choice
    if not CUDA_AVAILABLE:
        return False
    if gpu_pipeline is not None and gpu_pipeline_choice != model_choice:
        print(f"Unloading GPU pipeline for {gpu_pipeline_choice}")
        gpu_pipeline = None
        gpu_pipeline_choice = None
        clear_memory(release_cuda_cache=True)

    if gpu_pipeline is None:
        clear_memory(release_cuda_cache=True)
        try:
            gpu_pipeline = build_pipeline(model_choice, "cuda")
            gpu_pipeline_choice = model_choice
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU pipeline ready. Allocated memory: {allocated:.2f} GB")
        except Exception:
            gpu_pipeline = None
            gpu_pipeline_choice = None
            raise
    return True


def load_cpu_pipeline(model_choice):
    global cpu_pipeline, cpu_pipeline_choice
    if cpu_pipeline is not None and cpu_pipeline_choice != model_choice:
        print(f"Unloading CPU pipeline for {cpu_pipeline_choice}")
        cpu_pipeline = None
        cpu_pipeline_choice = None
        clear_memory()

    if cpu_pipeline is None:
        clear_memory()
        cpu_pipeline = build_pipeline(model_choice, "cpu")
        cpu_pipeline_choice = model_choice
        print("CPU pipeline ready")
    return True


def generate_with_pipeline(
    pipeline,
    prompt,
    device,
    steps,
    guidance,
    width,
    height,
    progress_state=None,
):
    generator_device = "cuda" if device == "GPU" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(42)

    def update_progress(_, step, timestep, callback_kwargs):
        if progress_state is not None:
            progress_state["step"] = int(step) + 1
            progress_state["timestep"] = int(timestep) if hasattr(timestep, "item") else timestep
        return callback_kwargs

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "generator": generator,
    }
    if progress_state is not None:
        kwargs.update(
            {
                "callback_on_step_end": update_progress,
                "callback_on_step_end_tensor_inputs": ["latents"],
            }
        )

    try:
        if device == "GPU":
            with torch.inference_mode():
                return pipeline(**kwargs).images[0]

        with torch.inference_mode():
            return pipeline(**kwargs).images[0]
    except TypeError as exc:
        if "callback_on_step_end" not in str(exc):
            raise

        kwargs.pop("callback_on_step_end", None)
        kwargs.pop("callback_on_step_end_tensor_inputs", None)
        if progress_state is not None:
            progress_state["callback_supported"] = False

        if device == "GPU":
            with torch.inference_mode():
                return pipeline(**kwargs).images[0]

        with torch.inference_mode():
            return pipeline(**kwargs).images[0]


def run_generation_job(job, pipeline, prompt, device, steps, guidance, width, height, progress_state):
    try:
        job["image"] = generate_with_pipeline(
            pipeline,
            prompt,
            device,
            steps,
            guidance,
            width,
            height,
            progress_state=progress_state,
        )
    except Exception as exc:
        job["error"] = exc
        job["traceback"] = traceback.format_exc()
    finally:
        job["done"] = True


def generate_image(
    prompt,
    device_choice,
    model_choice,
    offline_mode,
    num_inference_steps,
    guidance_scale,
    width,
    height,
    generation_limit,
):
    """Generate an image using the selected device."""
    set_offline_mode(offline_mode)

    prompt = (prompt or "").strip()
    if not prompt:
        yield None, "Please enter a prompt."
        return

    try:
        selected_config = model_config(model_choice)
    except ValueError as exc:
        yield None, str(exc)
        return

    width = int(width)
    height = int(height)
    if width % 8 != 0 or height % 8 != 0:
        yield None, "Width and height must be divisible by 8."
        return
    if width > MAX_OUTPUT_DIMENSION or height > MAX_OUTPUT_DIMENSION:
        yield None, f"Width and height must be {MAX_OUTPUT_DIMENSION}px or smaller."
        return

    generation_limit = int(generation_limit)
    generation_width, generation_height, will_upscale = calculate_generation_size(
        width,
        height,
        generation_limit,
    )
    if generation_width < 256 or generation_height < 256:
        yield None, (
            "The selected aspect ratio is too wide or too tall for the configured "
            "generation limit. Try a less extreme width/height ratio."
        )
        return

    total_start_time = time.time()
    clear_memory()
    gpu_memory_percent = model_gpu_memory_percent(selected_config)

    try:
        if device_choice == "GPU":
            if not CUDA_AVAILABLE:
                yield None, "GPU was selected, but CUDA is not available in this container."
                return

            yield progress_image(
                "Preparing GPU Generation",
                [
                    f"Model: {selected_config['model_id']}",
                    f"Output: {width}x{height}",
                    f"Internal generation: {generation_width}x{generation_height}",
                    "Checking GPU memory and preparing the pipeline...",
                ],
                generation_width,
                generation_height,
            ), "Checking GPU memory and preparing the pipeline..."
            memory_status = configure_gpu_memory_limit(gpu_memory_percent)
            pipeline_was_loaded = gpu_pipeline is not None and gpu_pipeline_choice == model_choice
            load_start_time = time.time()
            load_message = (
                "Reusing model already loaded in GPU memory."
                if pipeline_was_loaded
                else "Loading model into GPU memory. First run may also download cached files."
            )
            yield progress_image(
                "Loading Model",
                [
                    f"Model: {selected_config['model_id']}",
                    load_message,
                    "This is usually the slowest part on the first run.",
                    memory_status,
                ],
                generation_width,
                generation_height,
            ), load_message
            load_gpu_pipeline(model_choice)
            load_elapsed = time.time() - load_start_time
            pipeline = gpu_pipeline
            device_used = "GPU"
        elif device_choice == "CPU":
            memory_status = "Automatic GPU memory caps are ignored for CPU generation."
            yield progress_image(
                "Preparing CPU Generation",
                [
                    f"Model: {selected_config['model_id']}",
                    "CPU mode uses system RAM, not VRAM.",
                    "This can be much slower than GPU generation.",
                    f"Internal generation: {generation_width}x{generation_height}",
                ],
                generation_width,
                generation_height,
            ), "Preparing CPU generation..."
            pipeline_was_loaded = cpu_pipeline is not None and cpu_pipeline_choice == model_choice
            load_start_time = time.time()
            load_message = (
                "Reusing model already loaded in system RAM."
                if pipeline_was_loaded
                else "Loading model into system RAM. First run may also download cached files."
            )
            yield progress_image(
                "Loading Model",
                [
                    f"Model: {selected_config['model_id']}",
                    load_message,
                    "CPU generation can take a long time.",
                    memory_status,
                ],
                generation_width,
                generation_height,
            ), load_message
            load_cpu_pipeline(model_choice)
            load_elapsed = time.time() - load_start_time
            pipeline = cpu_pipeline
            device_used = "CPU"
        else:
            yield None, f"Unknown device choice: {device_choice}"
            return

        generation_start_time = time.time()
        steps = int(num_inference_steps)
        progress_state = {
            "step": 0,
            "total": steps,
            "callback_supported": True,
        }
        job = {"done": False, "image": None, "error": None, "traceback": None}
        thread = threading.Thread(
            target=run_generation_job,
            args=(
                job,
                pipeline,
                prompt,
                device_used,
                steps,
                float(guidance_scale),
                generation_width,
                generation_height,
                progress_state,
            ),
            daemon=True,
        )
        thread.start()

        while not job["done"]:
            generation_elapsed = time.time() - generation_start_time
            step = progress_state["step"]
            if step:
                detail = f"Diffusion step {step}/{steps}."
            elif progress_state["callback_supported"]:
                detail = "Generation is running. Waiting for the first diffusion step..."
            else:
                detail = "Generation is running. This pipeline does not expose step progress."

            yield progress_image(
                "Generating Image",
                [
                    detail,
                    f"Elapsed generation time: {format_seconds(generation_elapsed)}",
                    f"Load time before generation: {format_seconds(load_elapsed)}",
                    f"Internal generation: {generation_width}x{generation_height}",
                    f"Prompt: {prompt[:90]}",
                ],
                generation_width,
                generation_height,
            ), (
                f"{detail}\n"
                f"Generation elapsed: {format_seconds(generation_elapsed)}\n"
                f"Model load/cache time: {format_seconds(load_elapsed)}"
            )
            time.sleep(1)

        thread.join()
        if job["error"] is not None:
            raise job["error"]

        image = job["image"]

        image = upscale_image(image, width, height)
        total_elapsed = time.time() - total_start_time
        generation_elapsed = time.time() - generation_start_time
        clear_memory()
        if device_used == "GPU":
            memory_status = gpu_memory_status(gpu_memory_percent)
        size_status = f"Generated at {generation_width}x{generation_height}."
        if will_upscale:
            size_status += f" Upscaled to {width}x{height} output pixels."
        offline_status = "Offline model loading: on." if OFFLINE_MODE else "Offline model loading: off."
        blank_warning = ""
        if image_looks_blank(image):
            blank_warning = (
                "\nWarning: the returned image appears almost blank. "
                "Try a different prompt, fewer steps, lower guidance, or restart the container if this repeats."
            )
        yield image, (
            f"Generated using {device_used} in {total_elapsed:.1f}s.\n"
            f"Model: {selected_config['model_id']} ({selected_config['pipeline_type']}).\n"
            f"Model load/cache time: {load_elapsed:.1f}s.\n"
            f"Diffusion generation time: {generation_elapsed:.1f}s.\n"
            f"{size_status}\n"
            f"{memory_status}\n"
            f"{offline_status}"
            f"{blank_warning}"
        )

    except torch.cuda.OutOfMemoryError as exc:
        elapsed = time.time() - total_start_time
        unload_pipelines(release_cuda_cache=True)
        cleaned_error = normalize_oom_message(str(exc))
        tip = model_memory_tip(model_choice, selected_config, generation_width, generation_height)
        detail = (
            f"GPU ran out of memory after {elapsed:.1f}s.\n"
            f"Error: {cleaned_error}\n\n"
            f"{gpu_memory_status(gpu_memory_percent)}\n\n"
            "Try a lighter model, a smaller output size, a lower Internal Generation Limit, "
            "fewer steps, or restart the container to clear GPU state. "
            "The app unloaded the current pipelines after this OOM."
            f"{tip}"
        )
        print(detail)
        yield progress_image("GPU Out Of Memory", detail.splitlines(), generation_width, generation_height), detail
    except Exception as exc:
        elapsed = time.time() - total_start_time
        clear_memory(release_cuda_cache=True)
        details = job.get("traceback") if "job" in locals() and job.get("traceback") else traceback.format_exc()
        device_label = device_choice if device_choice in {"GPU", "CPU"} else "selected device"
        recovery_tip = cuda_runtime_tip(exc)
        message = (
            f"{device_label} generation failed after {elapsed:.1f}s.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error: {exc}\n\n"
            f"Details:\n{details}"
            f"{recovery_tip}"
        )
        print(message)
        yield progress_image("Generation Failed", message.splitlines()[:8], generation_width, generation_height), message


def create_interface():
    """Create the Gradio web interface."""
    default_config = model_config(DEFAULT_MODEL_CHOICE)
    default_output_size = min(default_config["default_size"], MAX_OUTPUT_DIMENSION)
    default_generation_limit = min(default_config["default_generation_limit"], MAX_GENERATION_DIMENSION)

    with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# AI Image Generator")
        gr.Markdown("Generate images using Stable Diffusion with GPU or CPU.")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image description...",
                    lines=3,
                )

                device_choice = gr.Radio(
                    choices=["GPU", "CPU"],
                    value="GPU" if CUDA_AVAILABLE else "CPU",
                    label="Device",
                    info="GPU is preferred when available.",
                )

                model_choice = gr.Radio(
                    choices=list(MODEL_OPTIONS.keys()),
                    value=DEFAULT_MODEL_CHOICE,
                    label="Model",
                    info="Supported choices switch at runtime and download into the cache volume on first use.",
                )

                gpu_recommendations = gr.Textbox(
                    label=(
                        "GPU And Model Recommendations"
                        if CUDA_AVAILABLE
                        else "CPU And Model Recommendations"
                    ),
                    value=initial_model_recommendation_text("GPU" if CUDA_AVAILABLE else "CPU"),
                    lines=12,
                    interactive=False,
                )

                model_status = gr.Textbox(
                    label="Selected Model",
                    value=selected_model_status(DEFAULT_MODEL_CHOICE),
                    lines=4,
                    interactive=False,
                )

                offline_toggle = gr.Checkbox(
                    label="Offline Mode",
                    value=OFFLINE_MODE,
                    info="Use cached Hugging Face model files only.",
                )

                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=default_config["default_steps"],
                        step=1,
                        label="Inference Steps",
                        info=(
                            "How many denoising passes to run. More steps can add detail, "
                            "but generation gets slower almost linearly."
                        ),
                    )

                    guidance_slider = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=default_config["default_guidance"],
                        step=0.5,
                        label="Guidance Scale",
                        info=(
                            "How strongly the image follows the prompt. Lower values are more "
                            "creative; higher values are stricter but can look harsh."
                        ),
                    )

                width_slider = gr.Slider(
                    minimum=256,
                    maximum=MAX_OUTPUT_DIMENSION,
                    value=default_output_size,
                    step=64,
                    label="Output Width (px)",
                    info="Final image width. Large values are generated smaller and upscaled.",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=MAX_OUTPUT_DIMENSION,
                    value=default_output_size,
                    step=64,
                    label="Output Height (px)",
                    info="Final image height. 3840x2160 is 4K UHD.",
                )
                generation_limit_slider = gr.Slider(
                    minimum=512,
                    maximum=MAX_GENERATION_DIMENSION,
                    value=default_generation_limit,
                    step=64,
                    label="Internal Generation Limit (px)",
                    info="Largest side used for the GPU diffusion pass before upscaling.",
                )
                with gr.Accordion("Generation Settings", open=False):
                    gr.Markdown(
                        f"""
                        **Inference Steps:** More steps give the model more chances to refine the image. `16-20` is a good fast range for most models. `30-50` can improve some prompts, but it is much slower.

                        **Guidance Scale:** Controls prompt strength. `5-8` is usually balanced. Lower values allow more variation. Very high values can over-sharpen, distort colors, or create artifacts.

                        **Output Width / Height:** Sets the final image pixels. For large sizes, the app generates at a safer internal size and upscales to the requested output. Use `3840x2160` for 4K UHD, or `4096x4096` for a large square image.

                        **Internal Generation Limit:** Caps the largest side used for the actual GPU diffusion pass. For large final outputs such as `2200x1800`, use `768px` first on 16 GB-class GPUs, then raise it only if the selected model has enough VRAM headroom.

                        **GPU Memory:** The app selects a per-model GPU memory cap automatically from the selected model and visible VRAM, leaving driver headroom.

                        **Model Selection:** Supported models are switched at runtime. The first generation with a new model downloads it into the Docker cache volume, then later runs reuse the cached files. A Docker rebuild is only needed after app code or dependency changes.
                        """
                    )
                generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", height=512)
                status_text = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("System Information", open=False):
            gpu_name = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
            cuda_runtime = torch.version.cuda or "N/A"
            info_text = f"""
            **GPU Available:** {CUDA_AVAILABLE}
            **GPU Name:** {gpu_name}
            **PyTorch Version:** {torch.__version__}
            **PyTorch CUDA Runtime:** {cuda_runtime}
            **Configured MODEL_ID:** {MODEL_ID}
            **Default Model Choice:** {DEFAULT_MODEL_CHOICE}
            **Cache Directory:** {CACHE_DIR}
            **Offline Model Loading:** {OFFLINE_MODE}
            **GPU Preload:** {PRELOAD_GPU}
            **Attention Slicing:** {ENABLE_ATTENTION_SLICING}
            **VAE Slicing:** {ENABLE_VAE_SLICING}
            **VAE Tiling:** {ENABLE_VAE_TILING}
            **Torch Compile:** {ENABLE_TORCH_COMPILE}
            **Max Automatic GPU Memory Cap:** {MAX_GPU_MEMORY_PERCENT}%
            **Max Generation Dimension:** {MAX_GENERATION_DIMENSION}px
            **Max Output Dimension:** {MAX_OUTPUT_DIMENSION}px
            """
            gr.Markdown(info_text)

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                device_choice,
                model_choice,
                offline_toggle,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
                generation_limit_slider,
            ],
            outputs=[output_image, status_text],
        )

        device_choice.change(
            fn=describe_model_options,
            inputs=[device_choice],
            outputs=[gpu_recommendations, model_choice, model_status],
        )

        model_choice.change(
            fn=model_generation_defaults,
            inputs=[model_choice],
            outputs=[
                model_status,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
                generation_limit_slider,
            ],
        )

    return interface


if __name__ == "__main__":
    log_system_info()
    check_cache_status()
    if CUDA_AVAILABLE:
        configure_gpu_memory_limit(model_gpu_memory_percent(model_config(DEFAULT_MODEL_CHOICE)))
    if PRELOAD_GPU and CUDA_AVAILABLE:
        print("Preloading GPU pipeline at startup.")
        load_gpu_pipeline(DEFAULT_MODEL_CHOICE)
    else:
        print("Pipelines will be loaded on first use.")

    try:
        app = create_interface()
        print("Open in your browser: http://localhost:7860")
        print("Docker binds the app inside the container on 0.0.0.0:7860.")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=False,
            inbrowser=False,
            prevent_thread_lock=False,
        )
    except Exception as exc:
        print(f"Failed to start application: {exc}")
        raise
