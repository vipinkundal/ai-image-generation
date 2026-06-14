"""
Simple AI Image Generator.

GPU is preferred when CUDA is available. CPU runs only when explicitly selected.
"""

import gc
import os
import time
import traceback

# Set Hugging Face cache directories before importing diffusers/transformers.
os.environ.setdefault("HF_HOME", "/app/cache")
os.environ.setdefault("HF_HUB_CACHE", "/app/cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/cache/hub")

import gradio as gr
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image


MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
CACHE_DIR = os.getenv("HF_HOME", "/app/cache")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}
PRELOAD_GPU = os.getenv("PRELOAD_GPU", "0").lower() in {"1", "true", "yes"}
ENABLE_ATTENTION_SLICING = os.getenv("ENABLE_ATTENTION_SLICING", "0").lower() in {
    "1",
    "true",
    "yes",
}
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "0").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_GPU_MEMORY_PERCENT = int(os.getenv("GPU_MEMORY_PERCENT", "80"))
MAX_GENERATION_DIMENSION = int(os.getenv("MAX_GENERATION_DIMENSION", "1024"))
MAX_OUTPUT_DIMENSION = int(os.getenv("MAX_OUTPUT_DIMENSION", "4096"))

CUDA_AVAILABLE = torch.cuda.is_available()
gpu_pipeline = None
cpu_pipeline = None
current_gpu_memory_fraction = None

if CUDA_AVAILABLE:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def round_down_to_multiple(value, multiple):
    return max(multiple, int(value) // multiple * multiple)


def calculate_generation_size(output_width, output_height):
    """Pick a diffusion size that preserves aspect ratio and can be upscaled."""
    max_generation_dimension = clamp(MAX_GENERATION_DIMENSION, 256, MAX_OUTPUT_DIMENSION)
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


def configure_gpu_memory_limit(percent):
    """Set the PyTorch CUDA allocator limit for this process."""
    global current_gpu_memory_fraction

    if not CUDA_AVAILABLE:
        return "CUDA is not available."

    percent = clamp(int(percent), 40, 100)
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
        f"{allocated:.2f} GB allocated, "
        f"{reserved:.2f} GB reserved, "
        f"{limit_gb:.2f} GB limit ({int(limit_percent)}% of {total:.2f} GB visible)."
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
    print(f"Model: {MODEL_ID}")
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
        torch.cuda.empty_cache()


def set_offline_mode(enabled):
    """Switch Hugging Face model loading between online and cached-only modes."""
    global OFFLINE_MODE, gpu_pipeline, cpu_pipeline

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
        clear_memory(release_cuda_cache=True)


def build_pipeline(device):
    """Load Stable Diffusion for the requested device."""
    is_gpu = device == "cuda"
    dtype = torch.float16 if is_gpu else torch.float32

    print(f"Loading {MODEL_ID} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        local_files_only=OFFLINE_MODE,
    )

    if is_gpu:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )

    pipeline = pipeline.to(device)

    if is_gpu:
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

    if is_gpu and ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
        print("Compiling UNet. First generation will be slower; later generations may be faster.")
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=False)

    return pipeline


def load_gpu_pipeline():
    global gpu_pipeline
    if not CUDA_AVAILABLE:
        return False
    if gpu_pipeline is None:
        clear_memory(release_cuda_cache=True)
        try:
            gpu_pipeline = build_pipeline("cuda")
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU pipeline ready. Allocated memory: {allocated:.2f} GB")
        except Exception:
            gpu_pipeline = None
            raise
    return True


def load_cpu_pipeline():
    global cpu_pipeline
    if cpu_pipeline is None:
        clear_memory()
        cpu_pipeline = build_pipeline("cpu")
        print("CPU pipeline ready")
    return True


def generate_with_pipeline(pipeline, prompt, device, steps, guidance, width, height):
    generator_device = "cuda" if device == "GPU" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(42)

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "generator": generator,
    }

    if device == "GPU":
        with torch.inference_mode(), torch.autocast("cuda"):
            return pipeline(**kwargs).images[0]

    with torch.inference_mode():
        return pipeline(**kwargs).images[0]


def generate_image(
    prompt,
    device_choice,
    offline_mode,
    gpu_memory_percent,
    num_inference_steps,
    guidance_scale,
    width,
    height,
):
    """Generate an image using the selected device."""
    set_offline_mode(offline_mode)

    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Please enter a prompt."

    width = int(width)
    height = int(height)
    if width % 8 != 0 or height % 8 != 0:
        return None, "Width and height must be divisible by 8."
    if width > MAX_OUTPUT_DIMENSION or height > MAX_OUTPUT_DIMENSION:
        return None, f"Width and height must be {MAX_OUTPUT_DIMENSION}px or smaller."

    generation_width, generation_height, will_upscale = calculate_generation_size(width, height)
    if generation_width < 256 or generation_height < 256:
        return None, (
            "The selected aspect ratio is too wide or too tall for the configured "
            "generation limit. Try a less extreme width/height ratio."
        )

    start_time = time.time()
    clear_memory()

    try:
        if device_choice == "GPU":
            if not CUDA_AVAILABLE:
                return None, "GPU was selected, but CUDA is not available in this container."

            memory_status = configure_gpu_memory_limit(gpu_memory_percent)
            load_gpu_pipeline()
            image = generate_with_pipeline(
                gpu_pipeline,
                prompt,
                "GPU",
                int(num_inference_steps),
                float(guidance_scale),
                generation_width,
                generation_height,
            )
            device_used = "GPU"
        elif device_choice == "CPU":
            memory_status = "GPU memory limit is ignored for CPU generation."
            load_cpu_pipeline()
            image = generate_with_pipeline(
                cpu_pipeline,
                prompt,
                "CPU",
                int(num_inference_steps),
                float(guidance_scale),
                generation_width,
                generation_height,
            )
            device_used = "CPU"
        else:
            return None, f"Unknown device choice: {device_choice}"

        image = upscale_image(image, width, height)
        elapsed = time.time() - start_time
        clear_memory()
        if device_used == "GPU":
            memory_status = gpu_memory_status(gpu_memory_percent)
        size_status = f"Generated at {generation_width}x{generation_height}."
        if will_upscale:
            size_status += f" Upscaled to {width}x{height} output pixels."
        offline_status = "Offline model loading: on." if OFFLINE_MODE else "Offline model loading: off."
        return image, (
            f"Generated using {device_used} in {elapsed:.1f}s.\n"
            f"{size_status}\n"
            f"{memory_status}\n"
            f"{offline_status}"
        )

    except torch.cuda.OutOfMemoryError as exc:
        elapsed = time.time() - start_time
        clear_memory(release_cuda_cache=True)
        detail = (
            f"GPU ran out of memory after {elapsed:.1f}s.\n"
            f"Error: {exc}\n\n"
            f"{gpu_memory_status(gpu_memory_percent)}\n\n"
            "Try raising the GPU memory limit, using a smaller output size, fewer steps, "
            "or restarting the container to clear GPU state."
        )
        print(detail)
        return None, detail
    except Exception as exc:
        elapsed = time.time() - start_time
        clear_memory(release_cuda_cache=True)
        details = traceback.format_exc()
        device_label = device_choice if device_choice in {"GPU", "CPU"} else "selected device"
        message = (
            f"{device_label} generation failed after {elapsed:.1f}s.\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error: {exc}\n\n"
            f"Details:\n{details}"
        )
        print(message)
        return None, message


def create_interface():
    """Create the Gradio web interface."""
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

                offline_toggle = gr.Checkbox(
                    label="Offline Mode",
                    value=OFFLINE_MODE,
                    info="Use cached Hugging Face model files only.",
                )

                gpu_memory_slider = gr.Slider(
                    minimum=40,
                    maximum=100,
                    value=clamp(DEFAULT_GPU_MEMORY_PERCENT, 40, 100),
                    step=5,
                    label="GPU Memory Limit (%)",
                    info="Limits how much visible GPU memory this app may use. Higher is not automatically faster.",
                    interactive=CUDA_AVAILABLE,
                )

                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=16,
                        step=1,
                        label="Inference Steps",
                        info=(
                            "How many denoising passes to run. More steps can add detail, "
                            "but generation gets slower almost linearly."
                        ),
                    )

                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
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
                    value=512,
                    step=64,
                    label="Output Width (px)",
                    info="Final image width. Large values are generated smaller and upscaled.",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=MAX_OUTPUT_DIMENSION,
                    value=512,
                    step=64,
                    label="Output Height (px)",
                    info="Final image height. 3840x2160 is 4K UHD.",
                )
                with gr.Accordion("Generation Settings", open=False):
                    gr.Markdown(
                        f"""
                        **Inference Steps:** More steps give the model more chances to refine the image. `16-20` is a good fast range with the DPM scheduler. `30-50` can improve some prompts, but it is much slower.

                        **Guidance Scale:** Controls prompt strength. `5-8` is usually balanced. Lower values allow more variation. Very high values can over-sharpen, distort colors, or create artifacts.

                        **Output Width / Height:** Sets the final image pixels. For large sizes, the app generates at a safer internal size and upscales to the requested output. Use `3840x2160` for 4K UHD, or `4096x4096` for a large square image.

                        **Internal Generation Size:** Stable Diffusion v1.5 works best around `512-1024px`. This app caps the largest generated side at `{MAX_GENERATION_DIMENSION}px` by default, then upscales when the requested output is larger.

                        **GPU Memory Limit:** Limits how much visible VRAM this app may use. It helps prevent out-of-memory errors, but raising it does not automatically make normal `512x512` generation faster.
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
            **Model:** {MODEL_ID}
            **Cache Directory:** {CACHE_DIR}
            **Offline Model Loading:** {OFFLINE_MODE}
            **GPU Preload:** {PRELOAD_GPU}
            **Attention Slicing:** {ENABLE_ATTENTION_SLICING}
            **Torch Compile:** {ENABLE_TORCH_COMPILE}
            **Default GPU Memory Limit:** {clamp(DEFAULT_GPU_MEMORY_PERCENT, 40, 100)}%
            **Max Generation Dimension:** {MAX_GENERATION_DIMENSION}px
            **Max Output Dimension:** {MAX_OUTPUT_DIMENSION}px
            """
            gr.Markdown(info_text)

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                device_choice,
                offline_toggle,
                gpu_memory_slider,
                steps_slider,
                guidance_slider,
                width_slider,
                height_slider,
            ],
            outputs=[output_image, status_text],
        )

    return interface


if __name__ == "__main__":
    log_system_info()
    check_cache_status()
    if CUDA_AVAILABLE:
        configure_gpu_memory_limit(DEFAULT_GPU_MEMORY_PERCENT)
    if PRELOAD_GPU and CUDA_AVAILABLE:
        print("Preloading GPU pipeline at startup.")
        load_gpu_pipeline()
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
