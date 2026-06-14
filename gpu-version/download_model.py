"""Download the configured Diffusers model into the Hugging Face cache."""

import os

from huggingface_hub import snapshot_download


MODEL_ID = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")
CACHE_DIR = os.getenv("HF_HOME", "/app/cache")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def main():
    print(f"Downloading model: {MODEL_ID}")
    print(f"Cache directory: {CACHE_DIR}")
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        local_files_only=False,
    )
    print(f"Model cached at: {snapshot_path}")


if __name__ == "__main__":
    main()
