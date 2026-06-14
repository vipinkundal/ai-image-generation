# Docker Volume Management for AI Image Generator

## Overview
This project uses Docker named volumes to persist downloaded models between container runs, eliminating the need to re-download the ~4GB Stable Diffusion model each time you start the container.

## Volume Configuration

### GPU Version
- **model_cache**: Stores application-specific model cache in `/app/cache`
- **huggingface_cache**: Stores Hugging Face Hub cache in `/app/cache/hub`
- **Port**: 7860

### CPU Version  
- **model_cache_cpu**: Stores application-specific model cache in `/app/cache`
- **huggingface_cache_cpu**: Stores HuggingFace models in `/root/.cache/huggingface`
- **Port**: 7861 (different port to avoid conflicts)

## Usage

### First Run
```powershell
# GPU Version
cd gpu-version
docker-compose up -d

# CPU Version  
cd cpu-version
docker-compose up -d
```

On the first run, the model (~4GB) will be downloaded and cached in the Docker volumes. Subsequent runs will use the cached model.

### Checking Volume Status
```powershell
# List all Docker volumes
docker volume ls

# Inspect a specific volume
docker volume inspect gpu-version_model_cache
docker volume inspect cpu-version_model_cache_cpu
```

### Volume Management Commands

#### View Volume Location
```powershell
# Find where Docker stores the volume data
docker volume inspect gpu-version_model_cache --format "{{.Mountpoint}}"
```

#### Clear Model Cache (if needed)
```powershell
# Stop containers first
docker-compose down

# Remove specific volumes to clear cache
docker volume rm gpu-version_model_cache
docker volume rm gpu-version_huggingface_cache

# Or remove all unused volumes
docker volume prune
```

#### Backup Volume Data
```powershell
# Create a backup container to copy volume data
docker run --rm -v gpu-version_model_cache:/data -v ${PWD}:/backup alpine tar czf /backup/model_cache_backup.tar.gz -C /data .
```

#### Restore Volume Data
```powershell
# Restore from backup
docker run --rm -v gpu-version_model_cache:/data -v ${PWD}:/backup alpine tar xzf /backup/model_cache_backup.tar.gz -C /data
```

## Benefits

1. **Faster Startup**: No model re-download after the first run
2. **Bandwidth Savings**: Models are downloaded once and reused
3. **Offline Usage**: Once cached, works without internet for model loading
4. **Persistence**: Models survive container rebuilds and restarts
5. **Isolation**: GPU and CPU versions have separate caches

## Storage Requirements

- **Stable Diffusion v1.5**: ~4GB
- **Additional HuggingFace caches**: ~1-2GB
- **Total per version**: ~5-6GB

## Troubleshooting

### Volume Not Persisting
- Ensure you're using `docker-compose up` instead of `docker run`
- Check that volumes are properly defined in `docker-compose.yml`

### Disk Space Issues
```powershell
# Check Docker system usage
docker system df

# Clean up unused volumes
docker volume prune

# Clean up everything unused
docker system prune -a --volumes
```

### Permission Issues
If you encounter permission issues, the volumes are managed by Docker and should work automatically with the container's user permissions.

## Architecture Notes

The volume configuration uses:
- **Named volumes**: Better performance and management than bind mounts
- **Dual cache paths**: GPU uses `/app/cache` and `/app/cache/hub`; CPU keeps its existing `/app/cache` and `/root/.cache/huggingface` paths
- **Separate volumes**: GPU and CPU versions have isolated caches to prevent conflicts
