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
    echo "ℹ️  Virtual environment .venv already exists — skipping creation."
else
    uv venv .venv
fi

# --- Setup .envrc correctly ---
echo "⚙️  Setting up .envrc..."
cat > .envrc << 'EOF'
#!/usr/bin/env bash
# Activate the virtual environment
source .venv/bin/activate
EOF
direnv allow || true

# --- Install Python project deps ---
if [ -f pyproject.toml ]; then
    echo "📦 Installing project dependencies from pyproject.toml..."
    uv sync
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
echo "1. Download models: ./scripts/setup/download_models.sh"
echo "2. Calibrate camera: python3 hardware_tests/calibrate_camera.py"
echo "3. Test hardware: python3 hardware_tests/test_*.py"
echo "4. Launch system: ros2 launch launch/full_system_launch.py"
echo
echo "Note: You may need to log out and back in for group changes to take effect"
