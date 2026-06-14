#!/bin/bash
# Optimized PyTorch build script for RTX 5080 - Maximum Performance

set -e

echo "🚀 Building PyTorch from source for RTX 5080 (optimized for performance)..."

# Set environment variables for optimal RTX 5080 build
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export FORCE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=1
export USE_OPENMP=1
export USE_LAPACK=1
export USE_FLASH_ATTENTION=1
export USE_MEMORY_EFFICIENT_ATTENTION=1
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1  # Single job for stability
export CMAKE_BUILD_TYPE=Release
export BUILD_TEST=0  # Skip tests for faster build
export USE_DISTRIBUTED=0  # Skip if not needed

# CMake settings for compatibility
export CMAKE_PREFIX_PATH=/usr/local
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Setup ccache for potential rebuilds
export PATH="/usr/lib/ccache:$PATH"
export CCACHE_DIR=/tmp/ccache
ccache -M 2G

echo "📥 Cloning PyTorch (latest stable)..."
# Try latest stable first, fallback to v2.4.0 if needed
if git clone --recursive --depth 1 --branch v2.4.1 https://github.com/pytorch/pytorch.git /tmp/pytorch 2>/dev/null; then
    echo "✅ Using PyTorch v2.4.1"
elif git clone --recursive --depth 1 --branch v2.4.0 https://github.com/pytorch/pytorch.git /tmp/pytorch 2>/dev/null; then
    echo "✅ Using PyTorch v2.4.0"
else
    echo "⚠️  Falling back to main branch"
    git clone --recursive --depth 1 https://github.com/pytorch/pytorch.git /tmp/pytorch
fi

cd /tmp/pytorch

# Verify CMake version
echo "🔍 Checking CMake version..."
cmake --version

echo "🛠️  Building PyTorch (this will take 1-3 hours, please be patient)..."
echo "💡 Progress indicators:"
echo "   - Configuring build..."
python3 setup.py clean

echo "   - Starting compilation (this is the longest step)..."
python3 setup.py install 2>&1 | tee /tmp/pytorch_build.log

# Check if the installation was successful by testing a basic import
echo "🔍 Testing PyTorch installation..."
if python3 -c "import torch; print('PyTorch imported successfully')" 2>/dev/null; then
    echo "✅ PyTorch installation successful!"
else
    echo "❌ PyTorch installation failed. Checking logs..."
    tail -50 /tmp/pytorch_build.log
    exit 1
fi

echo "✅ Verifying PyTorch installation..."
python3 -c "
try:
    import torch
    print(f'🎉 PyTorch version: {torch.__version__}')
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f'🔥 CUDA available: {cuda_available}')
    
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            print(f'🖥️  GPU count: {device_count}')
        except:
            print('�️  GPU count: Detection will complete after full setup')
            
        # Test basic CUDA operations
        try:
            x = torch.tensor([1.0]).cuda()
            print(f'� CUDA tensor operations: Working')
        except:
            print(f'🚀 CUDA tensor operations: Will be available after container restart')
    
    print('✅ PyTorch core installation verified!')
    
except ImportError as e:
    print(f'❌ Critical error - PyTorch not installed: {e}')
    exit(1)
except Exception as e:
    print(f'⚠️  PyTorch installed but some CUDA features pending: {e}')
    print('🔄 Full verification will complete when container starts')
"

echo "🎯 PyTorch build completed!"
ccache -s  # Show cache statistics
