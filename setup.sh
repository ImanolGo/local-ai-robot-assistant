#!/bin/bash

# Setup script for Local AI Robot Assistant
# This script installs all dependencies and configures the environment

set -e  # Exit on error

echo "========================================="
echo "Local AI Robot Assistant - Setup Script"
echo "========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: Not running on NVIDIA Jetson"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install ROS2 dependencies
echo "Installing ROS2 dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install audio dependencies
echo "Installing audio dependencies..."
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    alsa-utils

# Install vision dependencies
echo "Installing vision dependencies..."
sudo apt-get install -y \
    python3-opencv \
    libopencv-dev

# Initialize rosdep
if [ ! -d /etc/ros/rosdep ]; then
    echo "Initializing rosdep..."
    sudo rosdep init
fi
rosdep update

# Build workspace
echo "Building ROS2 workspace..."
cd src
colcon build --symlink-install

# Source workspace
echo "Sourcing workspace..."
source install/setup.bash

# Create necessary directories
echo "Creating directories..."
mkdir -p models/{wake_word,whisper_tiny_trt,piper_voice,yolo_trt,depth_trt,nanollm_quantized}
mkdir -p logs
mkdir -p maps
mkdir -p calibration_images

# Set permissions
echo "Setting permissions..."
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyTHS* 2>/dev/null || true
sudo chmod 666 /dev/ttyUSB* 2>/dev/null || true

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download models: ./scripts/setup/download_models.sh"
echo "2. Calibrate camera: python3 hardware_tests/calibrate_camera.py"
echo "3. Test hardware: python3 hardware_tests/test_*.py"
echo "4. Launch system: ros2 launch launch/full_system_launch.py"
echo ""
echo "Note: You may need to log out and back in for group changes to take effect"