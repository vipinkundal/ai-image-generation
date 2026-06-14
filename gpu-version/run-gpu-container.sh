#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

image_name="ai-image-gpu:latest"
container_name="ai-image-gpu"
cache_volume="ai-image-model-cache"

docker volume create "$cache_volume" >/dev/null

if ! docker image inspect "$image_name" >/dev/null 2>&1; then
  echo "Image $image_name does not exist. Building it now..."
  docker build -t "$image_name" .
fi

echo "Removing any existing $container_name container..."
docker rm -f "$container_name" >/dev/null 2>&1 || true

echo "Starting $container_name from $image_name..."
docker run -d \
  --name "$container_name" \
  --gpus all \
  -p 7860:7860 \
  -v "$cache_volume":/app/cache \
  "$image_name"

docker ps --filter "name=^/${container_name}$" \
  --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

echo "Open http://localhost:7860"
