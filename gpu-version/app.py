"""
Simple AI Image Generator.

GPU is preferred when CUDA is available, with CPU kept as a fallback.
"""

import gc
import os
import time

# Set Hugging Face cache directories before importing diffusers/transformers.
os.environ.setdefault("HF_HOME", "/app/cache")
os.environ.setdefault("HF_HUB_CACHE", "/app/cache/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/app/cache/hub")

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
CACHE_DIR = os.getenv("HF_HOME", "/app/cache")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}

CUDA_AVAILABLE = torch.cuda.is_available()
gpu_pipeline = None
cpu_pipeline = None


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
        print("Running without CUDA. GPU generation will fall back to CPU.")


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


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_pipeline(device):
    """Load Stable Diffusion for the requested device."""
    is_gpu = device == "cuda"
    dtype = torch.float16 if is_gpu else torch.float32

    print(f"Loading {MODEL_ID} on {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        local_files_only=OFFLINE_MODE,
    )

    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()

    if is_gpu and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("xFormers attention enabled")
        except Exception as exc:
            print(f"xFormers attention unavailable, continuing without it: {exc}")

    return pipeline


def load_gpu_pipeline():
    global gpu_pipeline
    if not CUDA_AVAILABLE:
        return False
    if gpu_pipeline is None:
        clear_memory()
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


def generate_image(prompt, device_choice, num_inference_steps, guidance_scale, width, height):
    """Generate an image using the selected device."""
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Please enter a prompt."

    width = int(width)
    height = int(height)
    if width % 8 != 0 or height % 8 != 0:
        return None, "Width and height must be divisible by 8."

    start_time = time.time()
    clear_memory()

    try:
        if device_choice == "GPU" and CUDA_AVAILABLE:
            try:
                load_gpu_pipeline()
                image = generate_with_pipeline(
                    gpu_pipeline,
                    prompt,
                    "GPU",
                    int(num_inference_steps),
                    float(guidance_scale),
                    width,
                    height,
                )
                device_used = "GPU"
            except torch.cuda.OutOfMemoryError:
                clear_memory()
                return (
                    None,
                    "GPU ran out of memory. Try 512x512, fewer steps, or CPU mode.",
                )
            except Exception as exc:
                print(f"GPU generation failed: {exc}")
                if os.getenv("DISABLE_CPU_FALLBACK", "0").lower() in {"1", "true", "yes"}:
                    raise
                print("Falling back to CPU generation")
                load_cpu_pipeline()
                image = generate_with_pipeline(
                    cpu_pipeline,
                    prompt,
                    "CPU",
                    int(num_inference_steps),
                    float(guidance_scale),
                    width,
                    height,
                )
                device_used = "CPU fallback"
        else:
            load_cpu_pipeline()
            image = generate_with_pipeline(
                cpu_pipeline,
                prompt,
                "CPU",
                int(num_inference_steps),
                float(guidance_scale),
                width,
                height,
            )
            device_used = "CPU"

        elapsed = time.time() - start_time
        clear_memory()
        return image, f"Generated using {device_used} in {elapsed:.1f}s."

    except Exception as exc:
        elapsed = time.time() - start_time
        clear_memory()
        print(f"Generation failed after {elapsed:.1f}s: {exc}")
        return None, f"Generation failed after {elapsed:.1f}s: {exc}"


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

                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps",
                    )

                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )

                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Width (px)",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Height (px)",
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
            """
            gr.Markdown(info_text)

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                device_choice,
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
    print("Pipelines will be loaded on first use.")

    try:
        app = create_interface()
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
