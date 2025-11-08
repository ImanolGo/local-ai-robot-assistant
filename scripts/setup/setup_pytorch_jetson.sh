#!/bin/bash

set -e

echo "========================================="
echo " PyTorch Setup for Jetson Orin Nano     "
echo "========================================="

# Check if we're on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "❌ Error: This script is for NVIDIA Jetson devices only"
    exit 1
fi

# Activate virtual environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Error: Virtual environment not found. Run setup.sh first."
    exit 1
fi

source .venv/bin/activate

# Get JetPack version
JETPACK_VERSION=$(apt-cache show nvidia-jetpack | grep Version | head -1 | cut -d' ' -f2)
JETPACK_MAJOR=$(echo "$JETPACK_VERSION" | cut -d'.' -f1)

echo "🔍 Detected JetPack version: $JETPACK_VERSION"

# Function to build torchvision from source
build_torchvision_from_source() {
    local pytorch_version="$1"

    echo "   💡 Building torchvision from source..."

    # Determine compatible torchvision version based on PyTorch version
    if [[ $pytorch_version == 2.8.* ]]; then
        TORCHVISION_BRANCH="release/0.19"
        TORCHVISION_BUILD_VERSION="0.19.0"
    elif [[ $pytorch_version == 2.5.* ]]; then
        TORCHVISION_BRANCH="release/0.20"
        TORCHVISION_BUILD_VERSION="0.20.0"
    elif [[ $pytorch_version == 2.4.* ]]; then
        TORCHVISION_BRANCH="release/0.19"
        TORCHVISION_BUILD_VERSION="0.19.0"
    elif [[ $pytorch_version == 2.3.* ]]; then
        TORCHVISION_BRANCH="release/0.18"
        TORCHVISION_BUILD_VERSION="0.18.0"
    else
        # Default fallback
        TORCHVISION_BRANCH="release/0.18"
        TORCHVISION_BUILD_VERSION="0.18.0"
        echo "   ⚠️  Unknown PyTorch version ($pytorch_version), using torchvision 0.18.0 as fallback"
    fi

    # Install build dependencies
    echo "   Installing build dependencies..."
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y \
        libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        build-essential cmake git > /dev/null 2>&1

    # Build torchvision from source
    cd /tmp
    echo "   Cloning torchvision repository (branch: $TORCHVISION_BRANCH)..."
    rm -rf torchvision  # Remove any existing clone

    git clone --branch "$TORCHVISION_BRANCH" https://github.com/pytorch/vision torchvision > /dev/null 2>&1
    cd torchvision

    export BUILD_VERSION="$TORCHVISION_BUILD_VERSION"
    export TORCH_CUDA_ARCH_LIST="8.7"  # Orin Nano compute capability
    export FORCE_CUDA=1

    echo "   Building torchvision $TORCHVISION_BUILD_VERSION from source (this may take 10-15 minutes)..."
    python3 setup.py install --user > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "   ✅ Built and installed torchvision $TORCHVISION_BUILD_VERSION from source"
    else
        echo "   ❌ Failed to build torchvision from source"
        return 1
    fi

    cd - > /dev/null
    rm -rf /tmp/torchvision
    return 0
}

# Install system dependencies for PyTorch compilation
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libopenblas-dev \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libffi-dev \
    libssl-dev

# Remove existing PyTorch installations
echo "🧹 Removing existing PyTorch installations..."
uv pip uninstall torch torchvision torchaudio || true

# Install PyTorch based on JetPack version
if [ "$JETPACK_MAJOR" = "6" ]; then
    echo "🔥 Installing PyTorch for JetPack 6.x..."

    # First install cuSPARSELt (required for PyTorch 24.06+)
    echo "📦 Installing cuSPARSELt..."
    if [ ! -f /usr/local/cuda/include/cusparseLt.h ]; then
        echo "   Installing cuSPARSELt 0.7.1.0..."
        mkdir -p /tmp/tmp_cusparselt && cd /tmp/tmp_cusparselt
        CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64"
        CUSPARSELT_VERSION="0.7.1.0"
        CUSPARSELT_NAME="libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive"

        if curl --retry 3 -OLs "${CUSPARSELT_URL}/${CUSPARSELT_NAME}.tar.xz"; then
            tar xf "${CUSPARSELT_NAME}.tar.xz"
            sudo cp -a "${CUSPARSELT_NAME}/include/"* /usr/local/cuda/include/
            sudo cp -a "${CUSPARSELT_NAME}/lib/"* /usr/local/cuda/lib64/
            sudo ldconfig
            echo "   ✅ cuSPARSELt installed"
        else
            echo "   ⚠️  cuSPARSELt download failed, continuing without it"
        fi
        cd - > /dev/null
        rm -rf /tmp/tmp_cusparselt
    else
        echo "   ✅ cuSPARSELt already installed"
    fi

    # For JetPack 6.x - use pre-compiled wheels optimized for JetPack 6.2.1
    echo "   Using pre-compiled wheels for JetPack 6.2.1..."

    # Primary wheels from Shattered217's JetPack 6.2.1 collection
    TORCH_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/torch-2.3.0a0+git97ff6cf-cp310-cp310-linux_aarch64.whl"
    TORCH_WHEEL_FILE="torch-2.3.0a0+git97ff6cf-cp310-cp310-linux_aarch64.whl"
    TORCHVISION_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl"
    TORCHVISION_WHEEL_FILE="torchvision-0.18.0-cp310-cp310-linux_aarch64.whl"
    ONNX_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"
    ONNX_WHEEL_FILE="onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"

    # Backup wheels from Jetson AI Lab
    BACKUP_TORCH_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
    BACKUP_TORCH_WHEEL_FILE="torch-2.8.0-cp310-cp310-linux_aarch64.whl"

    # Final fallback wheels
    NVIDIA_TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
    FALLBACK_TORCH_URL="http://jetson.webredirect.org/jp6/cu126/+f/5cf/9ed17e35cb752/torch-2.5.0-cp310-cp310-linux_aarch64.whl#sha256=5cf9ed17e35cb7523812aeda9e7d6353c437048c5a6df1dc6617650333049092"
    FALLBACK_TORCHVISION_URL="http://jetson.webredirect.org/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl#sha256=5f967f920de3953f2a39d95154b1feffd5ccc06b4589e51540dc070021a9adb9"

elif [ "$JETPACK_MAJOR" = "5" ]; then
    echo "🔥 Installing PyTorch for JetPack 5.x..."

    # PyTorch 2.1.0 for JetPack 5.1.2
    TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"
    TORCHVISION_VERSION="0.16.0"
    TORCHVISION_SOURCE_VERSION="16"
else
    echo "❌ Error: Unsupported JetPack version: $JETPACK_VERSION"
    echo "Please check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 for compatible wheels"
    exit 1
fi

# Download and install PyTorch
echo "⬇️  Downloading PyTorch and torchvision wheels..."

if [ "$JETPACK_MAJOR" = "6" ]; then
    # Try JetPack 6.2.1 specific wheels first (most compatible)
    echo "   Trying JetPack 6.2.1 specific wheels..."

    # Download PyTorch wheel
    if wget --quiet "$TORCH_WHEEL_URL" -O "/tmp/$TORCH_WHEEL_FILE"; then
        echo "   ✅ Downloaded PyTorch 2.3.0 wheel for JetPack 6.2.1"

        # Download torchvision wheel
        if wget --quiet "$TORCHVISION_WHEEL_URL" -O "/tmp/$TORCHVISION_WHEEL_FILE"; then
            echo "   ✅ Downloaded torchvision 0.18.0 wheel for JetPack 6.2.1"

            # Install both wheels
            echo "📦 Installing PyTorch 2.3.0 and torchvision 0.18.0..."
            uv pip install --no-cache-dir "/tmp/$TORCH_WHEEL_FILE"
            uv pip install --no-cache-dir "/tmp/$TORCHVISION_WHEEL_FILE"

            # Clean up
            rm -f "/tmp/$TORCH_WHEEL_FILE" "/tmp/$TORCHVISION_WHEEL_FILE"
            echo "   ✅ Successfully installed PyTorch 2.3.0 and torchvision 0.18.0 for JetPack 6.2.1"

        else
            echo "   ⚠️  JetPack 6.2.1 torchvision download failed"
            # Install PyTorch and try fallback for torchvision
            uv pip install --no-cache-dir "/tmp/$TORCH_WHEEL_FILE"
            rm -f "/tmp/$TORCH_WHEEL_FILE"

            # Try backup wheels or build from source
            echo "   💡 Trying backup wheels or building from source..."
            if wget --quiet "$BACKUP_TORCH_WHEEL_URL" -O "/tmp/$BACKUP_TORCH_WHEEL_FILE"; then
                echo "   Installing backup PyTorch wheel..."
                uv pip uninstall torch || true
                uv pip install --no-cache-dir "/tmp/$BACKUP_TORCH_WHEEL_FILE"
                rm -f "/tmp/$BACKUP_TORCH_WHEEL_FILE"
            fi

            # Build torchvision from source as last resort
            PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
            echo "   Building torchvision from source for PyTorch $PYTORCH_VERSION..."
            build_torchvision_from_source "$PYTORCH_VERSION"
        fi
    else
        echo "   ❌ JetPack 6.2.1 PyTorch download failed, trying backup wheels..."

        # Try backup Jetson AI Lab wheels
        if wget --quiet "$BACKUP_TORCH_WHEEL_URL" -O "/tmp/$BACKUP_TORCH_WHEEL_FILE"; then
            echo "   ✅ Downloaded backup PyTorch wheel"
            uv pip install --no-cache-dir "/tmp/$BACKUP_TORCH_WHEEL_FILE"
            rm -f "/tmp/$BACKUP_TORCH_WHEEL_FILE"

            # Build compatible torchvision from source
            PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
            echo "   Building torchvision from source for PyTorch $PYTORCH_VERSION..."
            build_torchvision_from_source "$PYTORCH_VERSION"
        else
            echo "   💡 Trying NVIDIA official wheels..."
            if uv pip install --no-cache-dir "$NVIDIA_TORCH_URL"; then
                echo "   ✅ Installed PyTorch from NVIDIA wheel"

                # Build compatible torchvision
                PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
                build_torchvision_from_source "$PYTORCH_VERSION"
            else
                echo "   ❌ All PyTorch installation attempts failed"
                exit 1
            fi
        fi
    fi

else
    # JetPack 5.x - use NVIDIA wheels
    WHEEL_FILENAME=$(basename "$TORCH_WHEEL_URL" | cut -d'#' -f1)
    if curl --fail -L -o "/tmp/$WHEEL_FILENAME" "$TORCH_WHEEL_URL"; then
        echo "📦 Installing PyTorch wheel..."
        uv pip install --no-cache-dir "/tmp/$WHEEL_FILENAME"
        rm -f "/tmp/$WHEEL_FILENAME"

        echo "📦 Installing torchvision..."
        uv pip install --no-cache-dir "torchvision==$TORCHVISION_VERSION"
    else
        echo "❌ Error: Failed to download PyTorch wheel for JetPack 5.x"
        echo "Please check https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
        echo "for the latest PyTorch wheels for your JetPack version."
        exit 1
    fi
fi

# Install ONNX Runtime for Jetson
echo "🔧 Installing ONNX Runtime GPU for Jetson..."
uv pip uninstall onnxruntime onnxruntime-gpu || true

# Fix numpy version compatibility issue
echo "🔧 Fixing numpy compatibility..."
uv pip install 'numpy<2'

# For JetPack 6, use the specific wheel mentioned in documentation
if [ "$JETPACK_MAJOR" = "6" ]; then
    echo "   Installing ONNX Runtime GPU wheel for JetPack 6.2.1..."

    # Try the JetPack 6.2.1 specific wheel first
    if wget --quiet "$ONNX_WHEEL_URL" -O "$ONNX_WHEEL_FILE"; then
        uv pip install "$ONNX_WHEEL_FILE"
        rm -f "$ONNX_WHEEL_FILE"
        echo "   ✅ Installed ONNX Runtime GPU 1.24.0 for JetPack 6.2.1"
    else
        echo "   ⚠️  JetPack 6.2.1 ONNX wheel failed, trying backup wheel..."
        # Fallback to the original wheel from NVIDIA box
        BACKUP_ONNX_WHEEL_URL="https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl"
        if wget --quiet "$BACKUP_ONNX_WHEEL_URL" -O "onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl"; then
            uv pip install onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
            rm -f onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
            echo "   ✅ Installed backup ONNX Runtime GPU 1.17.0"
        else
            echo "   ⚠️  Failed to download ONNX Runtime wheels, using standard version"
            uv pip install onnxruntime
        fi
    fi
else
    # Try Jetson-optimized ONNX Runtime for JetPack 5
    if uv pip install onnxruntime-gpu --extra-index-url https://pypi.nvidia.com; then
        echo "✅ Installed ONNX Runtime GPU for Jetson"
    else
        echo "⚠️  Jetson-optimized ONNX Runtime not available, using standard version"
        uv pip install onnxruntime
    fi
fi

# Verify installation
echo "🧪 Verifying PyTorch CUDA installation..."
python3 -c "
import sys
import torch
import torchvision

print('=' * 50)
print('PyTorch Installation Verification')
print('=' * 50)
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability(0)}')
    print(f'Current device: {torch.cuda.current_device()}')

    # Test tensor operations
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.mm(x, y)
        print('✅ CUDA tensor operations working correctly')
    except Exception as e:
        print(f'❌ CUDA tensor operations failed: {e}')
else:
    print('❌ CUDA not available - PyTorch installation may have issues')
    print('   Environment variables:')
    import os
    print(f'   CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')
    print(f'   PATH: {\":\" in os.environ.get(\"PATH\", \"\") and \"/usr/local/cuda/bin\" in os.environ.get(\"PATH\", \"\")}')
print('=' * 50)
"

# Test ONNX Runtime
echo "🧪 Testing ONNX Runtime..."
python3 -c "
try:
    import onnxruntime as ort
    print(f'ONNX Runtime version: {ort.__version__}')
    providers = ort.get_available_providers()
    print(f'Available providers: {providers}')
    if 'TensorrtExecutionProvider' in providers:
        print('✅ TensorRT provider available')
    elif 'CUDAExecutionProvider' in providers:
        print('✅ CUDA provider available')
    else:
        print('⚠️  Only CPU providers available')
except Exception as e:
    print(f'❌ ONNX Runtime test failed: {e}')
"

echo ""
echo "========================================="
echo "✅ PyTorch setup complete!"
echo "========================================="
echo ""
echo "If CUDA is still not available, try:"
echo "1. Reload environment: direnv reload"
echo "2. Check CUDA installation: ls /usr/local/cuda"
echo "3. Verify device access: ls -la /dev/nvidia*"
echo "4. Check latest wheels at: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
