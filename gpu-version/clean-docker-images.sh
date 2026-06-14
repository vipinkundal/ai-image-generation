#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Stopping the stable GPU container if it exists..."
docker compose down --remove-orphans

echo "Removing old auto-named GPU container if it exists..."
old_image_id="$(docker inspect -f '{{.Image}}' gpu-version-ai-image-generator-gpu-1 2>/dev/null || true)"
docker rm -f gpu-version-ai-image-generator-gpu-1 >/dev/null 2>&1 || true
if [[ -n "$old_image_id" ]]; then
  docker rmi "$old_image_id" >/dev/null 2>&1 || true
fi

echo "Removing dangling images created by this project..."
docker image prune -f --filter "label=org.opencontainers.image.title=ai-image-gpu"

echo "Current AI image generator images:"
docker images --filter "reference=ai-image-gpu" \
  --filter "reference=gpu-version-ai-image-generator-gpu" \
  --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}"
