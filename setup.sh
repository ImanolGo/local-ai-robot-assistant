#!/bin/bash
set -e

echo "========================================="
echo " Local AI Robot Assistant - Setup Script "
echo "========================================="

# --- Check Jetson ---
if [ ! -f /etc/nv_tegra_release ]; then
    echo "⚠️  Warning: Not running on NVIDIA Jetson"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# --- System update ---
echo "🔄 Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# --- Install essential tools ---
echo "🧰 Installing base tools..."
sudo apt-get install -y \
    python3 python3-venv python3-dev \
    curl git build-essential \
    direnv

# --- Install ROS2 deps ---
echo "🤖 Installing ROS2 dependencies..."
sudo apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

# --- Install audio + vision deps ---
echo "🎧 Installing audio + vision dependencies..."
sudo apt-get install -y \
    portaudio19-dev alsa-utils libopencv-dev python3-opencv

# Optional: Install audio feedback tools
sudo apt-get install -y espeak beep

# --- Install GStreamer and PyGObject for DeepStream ---
echo "🎬 Installing GStreamer and PyGObject dependencies..."
sudo apt-get install -y \
    python3-gi python3-gi-cairo \
    gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# --- Install DeepStream ---
echo "🔍 Installing DeepStream SDK..."
sudo apt-get install -y deepstream-7.1

# --- Install TensorRT and Model Conversion Dependencies ---
echo "🚀 Installing TensorRT and model conversion tools..."

# Install TensorRT system packages
sudo apt-get install -y \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    python3-libnvinfer-dev \
    tensorrt

# Install additional build dependencies for model conversion
sudo apt-get install -y \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config

# Install audio dependencies for Whisper
sudo apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    libasound2-dev

echo "   ✅ TensorRT and conversion dependencies installed"

# Verify TensorRT installation
if command -v trtexec &> /dev/null; then
    echo "   ✅ trtexec found and ready"
else
    echo "   ⚠️  trtexec not found - may need to add to PATH"
fi

# --- Fix IMX219 red tint issue ---
echo "🎨 Applying IMX219 camera ISP tuning fix for red tint..."
if [ ! -f /var/nvidia/nvcam/settings/camera_overrides.isp ]; then
    echo "   Downloading ArduCam ISP tuning parameters..."
    cd /tmp
    wget -q https://www.arducam.com/downloads/Jetson/Camera_overrides.tar.gz
    if [ $? -eq 0 ]; then
        tar zxvf Camera_overrides.tar.gz
        if [ -f camera_overrides.isp ]; then
            sudo mkdir -p /var/nvidia/nvcam/settings/
            sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
            sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
            sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
            echo "   ✅ ISP tuning file installed successfully"
        else
            echo "   ⚠️  ISP tuning file not found in archive"
        fi
        rm -f Camera_overrides.tar.gz camera_overrides.isp
    else
        echo "   ⚠️  Failed to download ISP tuning file - will use software color correction"
    fi
    cd - > /dev/null
else
    echo "   ✅ ISP tuning file already exists"
fi

# --- Setup direnv ---
echo "⚙️  Configuring direnv..."
if ! grep -q 'direnv hook bash' ~/.bashrc; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
fi

# --- Setup uv ---
if ! command -v uv &>/dev/null; then
    echo "🚀 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Create venv using uv ---
echo "🐍 Setting up Python virtual environment (via uv)..."
if [ -d ".venv" ]; then
    echo "ℹ️  Virtual environment .venv already exists — removing and recreating with system site packages..."
    rm -rf .venv
fi
# Create new venv with system site packages access
uv venv .venv --system-site-packages

# --- Setup .envrc correctly ---
echo "⚙️  Setting up .envrc..."
cat > .envrc << 'EOF'
#!/usr/bin/env bash
# Activate the virtual environment
source .venv/bin/activate

# Ensure GStreamer and GObject introspection work properly
export GI_TYPELIB_PATH=/usr/lib/aarch64-linux-gnu/girepository-1.0:/usr/lib/girepository-1.0
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

# TensorRT and CUDA paths
export PATH=$PATH:/usr/src/tensorrt/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64

# Python path for local modules
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/tools

# Model conversion tools
export MODEL_CACHE_DIR=$(pwd)/models
export TENSORRT_WORKSPACE_SIZE=256

# Set optimal CUDA settings for Jetson
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_PATH=/tmp/.nv/ComputeCache
EOF
direnv allow || true

# --- Install Python project deps ---
if [ -f pyproject.toml ]; then
    echo "📦 Installing project dependencies from pyproject.toml..."

    # Install PyTorch for Jetson first (if available)
    echo "🔥 Installing PyTorch for Jetson..."

    # Check if we're on Jetson and try to install optimized PyTorch
    if [ -f /etc/nv_tegra_release ]; then
        echo "   Detected Jetson - attempting to install optimized PyTorch..."

        # Try to install PyTorch for Jetson (user should check latest wheels at forums)
        # Note: These URLs may need to be updated for latest versions
        TORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"

        if curl --output /dev/null --silent --head --fail "$TORCH_WHEEL_URL"; then
            echo "   Installing PyTorch wheel for Jetson..."
            uv pip install "$TORCH_WHEEL_URL"
        else
            echo "   ⚠️  Jetson PyTorch wheel not available, falling back to pip install"
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi

        # Install ONNX Runtime for Jetson
        echo "   Installing ONNX Runtime GPU for Jetson..."
        uv pip install onnxruntime-gpu || uv pip install onnxruntime
    else
        echo "   Installing standard PyTorch..."
        uv pip install torch torchvision
    fi

    # Install remaining dependencies
    echo "   Installing remaining project dependencies..."
    uv sync

    # Install optional model conversion dependencies
    echo "   Installing model conversion tools..."
    uv sync --extra conversion --extra audio

    # Test TensorRT installation
    echo "🧪 Testing TensorRT installation..."
    if [ -f tools/test_tensorrt.py ]; then
        .venv/bin/python tools/test_tensorrt.py || echo "⚠️  TensorRT test failed - some dependencies may be missing"
    fi

else
    echo "⚠️  No pyproject.toml found. Creating a minimal one..."
    uv init --app
    uv add numpy opencv-python rospkg
fi

# --- Initialize rosdep ---
if [ ! -d /etc/ros/rosdep ]; then
    echo "🔧 Initializing rosdep..."
    sudo rosdep init || true
fi
rosdep update

# --- Build ROS2 workspace ---
echo "🏗️  Building ROS2 workspace..."
cd src
colcon build --symlink-install
cd ..

# --- Source workspace ---
echo "🌐 Sourcing workspace..."
source install/setup.bash

# --- Create directories ---
echo "📁 Creating required directories..."
mkdir -p models/{wake_word,whisper_tiny_trt,piper_voice,yolo_trt,depth_trt,nanollm_quantized}
mkdir -p logs maps calibration_images

# --- Permissions ---
echo "🔒 Setting permissions..."
sudo usermod -a -G dialout "$USER"
sudo chmod 666 /dev/ttyTHS* /dev/ttyUSB* 2>/dev/null || true

echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo
echo "Next steps:"
echo "1. Test model conversion tools: python3 tools/overview.py"
echo "2. Convert models: python3 tools/conversion/convert_yolo.py --help"
echo "3. Download models: ./scripts/setup/download_models.sh"
echo "4. Calibrate camera: python3 hardware_tests/calibrate_camera.py"
echo "5. Test hardware: python3 hardware_tests/test_*.py"
echo "6. Launch system: ros2 launch launch/full_system_launch.py"
echo
echo "Model conversion commands:"
echo "• YOLO: python3 tools/conversion/convert_yolo.py --model YOLOv11n --output-dir ./models/yolo_trt"
echo "• Depth: python3 tools/conversion/convert_depth.py --output-dir ./models/depth_trt"
echo "• Whisper: python3 tools/conversion/convert_whisper.py --model-size tiny"
echo
echo "Documentation:"
echo "• Model conversion guide: docs/guides/model_conversion_best_practices.md"
echo "• Architecture: docs/architecture.md"
echo
echo "Note: You may need to log out and back in for group changes to take effect"
