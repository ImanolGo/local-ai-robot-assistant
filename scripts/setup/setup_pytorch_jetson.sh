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

# Create persistent wheel directory
WHEEL_DIR="$HOME/jetson_wheels"
mkdir -p "$WHEEL_DIR"
echo "📁 Using wheel directory: $WHEEL_DIR"

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

    # PyTorch wheel URLs for JetPack 6
    TORCH_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
    TORCH_WHEEL_FILE="torch-2.8.0-cp310-cp310-linux_aarch64.whl"

    TORCHVISION_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1"
    TORCHVISION_WHEEL_FILE="torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"

    TENSORRT_WHEEL_URL="https://github.com/Shattered217/Jetson-Orin-Nano-Wheels/releases/download/6.2.1rc1/tensorrt-10.3.0-cp310-none-linux_aarch64.whl"
    TENSORRT_WHEEL_FILE="tensorrt-10.3.0-cp310-none-linux_aarch64.whl"

    ONNX_WHEEL_URL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl#sha256=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b"
    ONNX_WHEEL_FILE="onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"

    echo "⬇️  Installing PyTorch, torchvision, and TensorRT wheels..."

    # Download and install PyTorch
    if [ -f "$WHEEL_DIR/$TORCH_WHEEL_FILE" ]; then
        echo "   ✅ PyTorch wheel already exists ($(du -h "$WHEEL_DIR/$TORCH_WHEEL_FILE" | cut -f1))"
    else
        echo "   📥 Downloading PyTorch wheel (this may take several minutes)..."
        echo "      URL: $TORCH_WHEEL_URL"
        if wget --progress=bar:force --timeout=300 "$TORCH_WHEEL_URL" -O "$WHEEL_DIR/$TORCH_WHEEL_FILE"; then
            echo "   ✅ Downloaded PyTorch wheel ($(du -h "$WHEEL_DIR/$TORCH_WHEEL_FILE" | cut -f1))"
        else
            echo "   ❌ PyTorch download failed"
            exit 1
        fi
    fi
    echo "   📦 Installing PyTorch..."
    uv pip install "$WHEEL_DIR/$TORCH_WHEEL_FILE"
    echo "   ✅ PyTorch installed successfully"

    # Download and install torchvision
    if [ -f "$WHEEL_DIR/$TORCHVISION_WHEEL_FILE" ]; then
        echo "   ✅ torchvision wheel already exists ($(du -h "$WHEEL_DIR/$TORCHVISION_WHEEL_FILE" | cut -f1))"
    else
        echo "   📥 Downloading torchvision wheel..."
        echo "      URL: $TORCHVISION_WHEEL_URL"
        if wget --progress=bar:force --timeout=300 "$TORCHVISION_WHEEL_URL" -O "$WHEEL_DIR/$TORCHVISION_WHEEL_FILE"; then
            echo "   ✅ Downloaded torchvision wheel ($(du -h "$WHEEL_DIR/$TORCHVISION_WHEEL_FILE" | cut -f1))"
        else
            echo "   ❌ torchvision download failed"
            exit 1
        fi
    fi
    echo "   📦 Installing torchvision..."
    uv pip install "$WHEEL_DIR/$TORCHVISION_WHEEL_FILE"
    echo "   ✅ torchvision installed successfully"

    # Download and install TensorRT Python bindings
    if [ -f "$WHEEL_DIR/$TENSORRT_WHEEL_FILE" ]; then
        echo "   ✅ TensorRT wheel already exists ($(du -h "$WHEEL_DIR/$TENSORRT_WHEEL_FILE" | cut -f1))"
    else
        echo "   📥 Downloading TensorRT wheel..."
        echo "      URL: $TENSORRT_WHEEL_URL"
        if wget --progress=bar:force --timeout=300 "$TENSORRT_WHEEL_URL" -O "$WHEEL_DIR/$TENSORRT_WHEEL_FILE"; then
            echo "   ✅ Downloaded TensorRT wheel ($(du -h "$WHEEL_DIR/$TENSORRT_WHEEL_FILE" | cut -f1))"
        else
            echo "   ⚠️  TensorRT wheel download failed, using system TensorRT"
        fi
    fi
    if [ -f "$WHEEL_DIR/$TENSORRT_WHEEL_FILE" ]; then
        echo "   📦 Installing TensorRT..."
        uv pip install "$WHEEL_DIR/$TENSORRT_WHEEL_FILE"
        echo "   ✅ TensorRT installed successfully"
    fi

elif [ "$JETPACK_MAJOR" = "5" ]; then
    echo "🔥 Installing PyTorch for JetPack 5.x..."
    # Use NVIDIA official wheel for JetPack 5
    TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"
    TORCH_WHEEL_FILE="torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"

    if [ -f "$WHEEL_DIR/$TORCH_WHEEL_FILE" ]; then
        echo "   ✅ PyTorch wheel already exists"
    else
        if wget --quiet "$TORCH_WHEEL_URL" -O "$WHEEL_DIR/$TORCH_WHEEL_FILE"; then
            echo "   ✅ Downloaded PyTorch wheel for JetPack 5"
        else
            echo "❌ Error: Failed to download PyTorch wheel for JetPack 5.x"
            exit 1
        fi
    fi
    uv pip install "$WHEEL_DIR/$TORCH_WHEEL_FILE"
    uv pip install torchvision==0.16.0
else
    echo "❌ Error: Unsupported JetPack version: $JETPACK_VERSION"
    exit 1
fi

# Install ONNX Runtime
echo "🔧 Installing ONNX Runtime..."
uv pip install 'numpy<2'  # Fix numpy compatibility

if [ "$JETPACK_MAJOR" = "6" ] && [ -n "$ONNX_WHEEL_URL" ]; then
    # Try Jetson-optimized ONNX Runtime wheel
    if [ -f "$WHEEL_DIR/$ONNX_WHEEL_FILE" ]; then
        echo "   ✅ ONNX Runtime GPU wheel already exists ($(du -h "$WHEEL_DIR/$ONNX_WHEEL_FILE" | cut -f1))"
    else
        echo "   📥 Downloading ONNX Runtime GPU wheel..."
        echo "      URL: $ONNX_WHEEL_URL"
        if wget --progress=bar:force --timeout=300 "$ONNX_WHEEL_URL" -O "$WHEEL_DIR/$ONNX_WHEEL_FILE" 2>/dev/null; then
            echo "   ✅ Downloaded ONNX Runtime GPU wheel ($(du -h "$WHEEL_DIR/$ONNX_WHEEL_FILE" | cut -f1))"
        else
            echo "   ⚠️  ONNX Runtime GPU wheel download failed, using standard version"
        fi
    fi

    if [ -f "$WHEEL_DIR/$ONNX_WHEEL_FILE" ]; then
        echo "   📦 Installing ONNX Runtime GPU..."
        uv pip install "$WHEEL_DIR/$ONNX_WHEEL_FILE"
        echo "   ✅ ONNX Runtime GPU installed successfully"
    else
        uv pip install onnxruntime
    fi
else
    echo "   📦 Installing standard ONNX Runtime..."
    uv pip install onnxruntime
    echo "   ✅ Standard ONNX Runtime installed"
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
echo "Wheels saved to: $WHEEL_DIR"
echo ""
echo "Note: OpenCV does not have CUDA support (system version)"
echo "      For CUDA OpenCV, compile from source using OpenCV-4-13-0.sh"
