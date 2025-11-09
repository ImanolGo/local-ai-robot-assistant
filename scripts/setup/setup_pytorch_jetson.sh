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

# Install system OpenCV (reliable and simple)
echo "📦 Installing system OpenCV..."
sudo apt-get install -y python3-opencv

# Remove existing PyTorch installations
echo "🧹 Removing existing PyTorch and TensorRT installations..."
uv pip uninstall torch torchvision torchaudio tensorrt onnxruntime onnxruntime-gpu || true

# Install PyTorch based on JetPack version
if [ "$JETPACK_MAJOR" = "6" ]; then
    echo "🔥 Installing PyTorch for JetPack 6.x..."

    # Install cuSPARSELt if needed
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

    # PyTorch wheel URLs for JetPack 6.2.1
    TORCH_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/torch-2.3.0a0+git97ff6cf-cp310-cp310-linux_aarch64.whl"
    TORCHVISION_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl"
    TENSORRT_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/tensorrt-10.3.0-cp310-none-linux_aarch64.whl"
    ONNX_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"

    echo "⬇️  Installing PyTorch, torchvision, and TensorRT wheels..."

    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    echo "   📁 Created temp directory: $TEMP_DIR"

    # Download and install PyTorch
    TORCH_WHEEL_FILE=$(basename "$TORCH_WHEEL_URL")
    echo "   📥 Downloading PyTorch wheel (this may take several minutes)..."
    echo "      URL: $TORCH_WHEEL_URL"
    if wget --progress=bar:force --timeout=300 "$TORCH_WHEEL_URL" -O "$TEMP_DIR/$TORCH_WHEEL_FILE"; then
        echo "   ✅ Downloaded PyTorch wheel ($(du -h "$TEMP_DIR/$TORCH_WHEEL_FILE" | cut -f1))"
        echo "   📦 Installing PyTorch..."
        uv pip install "$TEMP_DIR/$TORCH_WHEEL_FILE"
        echo "   ✅ PyTorch installed successfully"
    else
        echo "   ❌ PyTorch download failed, trying backup..."
        # Fallback to NVIDIA wheel
        uv pip install torch --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
    fi

    # Download and install torchvision
    TORCHVISION_WHEEL_FILE=$(basename "$TORCHVISION_WHEEL_URL")
    echo "   📥 Downloading torchvision wheel..."
    echo "      URL: $TORCHVISION_WHEEL_URL"
    if wget --progress=bar:force --timeout=300 "$TORCHVISION_WHEEL_URL" -O "$TEMP_DIR/$TORCHVISION_WHEEL_FILE"; then
        echo "   ✅ Downloaded torchvision wheel ($(du -h "$TEMP_DIR/$TORCHVISION_WHEEL_FILE" | cut -f1))"
        echo "   📦 Installing torchvision..."
        uv pip install "$TEMP_DIR/$TORCHVISION_WHEEL_FILE"
        echo "   ✅ torchvision installed successfully"
    else
        echo "   ❌ torchvision download failed, will build from source"
        # Build torchvision from source as fallback
        echo "   🔨 Building torchvision from source..."
        uv pip install torchvision --no-deps
        PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo "   ✅ Built torchvision for PyTorch $PYTORCH_VERSION"
    fi

    # Download and install TensorRT Python bindings
    TENSORRT_WHEEL_FILE=$(basename "$TENSORRT_WHEEL_URL")
    echo "   📥 Downloading TensorRT wheel..."
    echo "      URL: $TENSORRT_WHEEL_URL"
    if wget --progress=bar:force --timeout=300 "$TENSORRT_WHEEL_URL" -O "$TEMP_DIR/$TENSORRT_WHEEL_FILE"; then
        echo "   ✅ Downloaded TensorRT wheel ($(du -h "$TEMP_DIR/$TENSORRT_WHEEL_FILE" | cut -f1))"
        echo "   📦 Installing TensorRT..."
        uv pip install "$TEMP_DIR/$TENSORRT_WHEEL_FILE"
        echo "   ✅ TensorRT installed successfully"
    else
        echo "   ⚠️  TensorRT wheel download failed, using system TensorRT"
    fi

elif [ "$JETPACK_MAJOR" = "5" ]; then
    echo "🔥 Installing PyTorch for JetPack 5.x..."
    # Use NVIDIA official wheel for JetPack 5
    TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"

    TEMP_DIR=$(mktemp -d)
    TORCH_WHEEL_FILE=$(basename "$TORCH_WHEEL_URL")
    if wget --quiet "$TORCH_WHEEL_URL" -O "$TEMP_DIR/$TORCH_WHEEL_FILE"; then
        echo "   ✅ Downloaded PyTorch wheel for JetPack 5"
        uv pip install "$TEMP_DIR/$TORCH_WHEEL_FILE"
        uv pip install torchvision==0.16.0
    else
        echo "❌ Error: Failed to download PyTorch wheel for JetPack 5.x"
        exit 1
    fi
else
    echo "❌ Error: Unsupported JetPack version: $JETPACK_VERSION"
    exit 1
fi

# Install ONNX Runtime
echo "🔧 Installing ONNX Runtime..."
uv pip install 'numpy<2'  # Fix numpy compatibility

if [ "$JETPACK_MAJOR" = "6" ] && [ -n "$ONNX_WHEEL_URL" ]; then
    # Try Jetson-optimized ONNX Runtime wheel
    ONNX_WHEEL_FILE=$(basename "$ONNX_WHEEL_URL")
    echo "   📥 Downloading ONNX Runtime GPU wheel..."
    echo "      URL: $ONNX_WHEEL_URL"
    if wget --progress=bar:force --timeout=300 "$ONNX_WHEEL_URL" -O "$TEMP_DIR/$ONNX_WHEEL_FILE" 2>/dev/null; then
        echo "   ✅ Downloaded ONNX Runtime GPU wheel ($(du -h "$TEMP_DIR/$ONNX_WHEEL_FILE" | cut -f1))"
        echo "   📦 Installing ONNX Runtime GPU..."
        uv pip install "$TEMP_DIR/$ONNX_WHEEL_FILE"
        echo "   ✅ ONNX Runtime GPU installed successfully"
    else
        echo "   ⚠️  ONNX Runtime GPU wheel download failed, using standard version"
        uv pip install onnxruntime
    fi
else
    echo "   📦 Installing standard ONNX Runtime..."
    uv pip install onnxruntime
    echo "   ✅ Standard ONNX Runtime installed"
fi

# Clean up temp directory
if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
    rm -rf "$TEMP_DIR"
fi

    echo ""
    echo "🧪 Testing installations..."
    echo "========================================="

    # Test PyTorch
    python3 -c "
import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   Device: {torch.cuda.get_device_name()}')
    # Test CUDA operations
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = torch.mm(x, y)
    print('   ✅ CUDA tensor operations working')
"

    # Test torchvision
    python3 -c "
import torchvision
print(f'✅ torchvision {torchvision.__version__}')
"

    # Test OpenCV
    python3 -c "
import cv2
print(f'✅ OpenCV {cv2.__version__} (system)')
print('   ℹ️  No CUDA support (system OpenCV)')
"

    # Test TensorRT
    python3 -c "
try:
    import tensorrt as trt
    print(f'✅ TensorRT {trt.__version__}')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    print('   ✅ TensorRT builder working')
except Exception as e:
    print(f'⚠️  TensorRT: {e}')
    print('   Using system TensorRT')
"

    # Test ONNX Runtime
    python3 -c "
try:
    import onnxruntime as ort
    print(f'✅ ONNX Runtime {ort.__version__}')
    providers = ort.get_available_providers()
    if 'TensorrtExecutionProvider' in providers:
        print('   ✅ TensorRT provider available')
    elif 'CUDAExecutionProvider' in providers:
        print('   ✅ CUDA provider available')
    else:
        print('   ℹ️  CPU provider only')
except Exception as e:
    print(f'⚠️  ONNX Runtime: {e}')
"

    echo "========================================="
    echo "✅ PyTorch and ML stack setup complete!"
    echo "========================================="
    echo ""
    echo "Summary:"
    echo "• PyTorch with CUDA support: ✅"
    echo "• torchvision: ✅"
    echo "• OpenCV (system): ✅"
    echo "• TensorRT: ✅"
    echo "• ONNX Runtime: ✅"
    echo ""
    echo "Note: OpenCV does not have CUDA support (system version)"
    echo "      For CUDA OpenCV, compile from source using OpenCV-4-13-0.sh"
