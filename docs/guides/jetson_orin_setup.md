# Jetson Orin Nano — Setup & ROS2 Quick Start

This guide collects the essential steps to flash, configure and prepare a Jetson Orin Nano for development with ROS2 (Humble). It is intended as a reproducible, copy-pastable checklist you can follow when provisioning a new device.

Requirements

- A host computer (Ubuntu recommended) with NVIDIA SDK Manager installed: [SDK Manager](https://developer.nvidia.com/sdk-manager)
- Jetson Orin Nano and required cables (USB-C for recovery, power, NVMe if applicable)
- NVMe SSD (if using external storage) and Internet access

## Step 1: Initialize Jetson Orin Nano (Est. 2-3 hours)

### 1.1 Flash JetPack (on host)

1. Download and install SDK Manager on the host machine from NVIDIA:

   ```bash
   # SDK Manager: https://developer.nvidia.com/sdk-manager
   ```

2. Put the Jetson into recovery mode and connect it to the host via USB-C.

3. Open SDK Manager, select the Jetson Orin Nano target and choose the desired JetPack
   version (pick the latest stable release compatible with your CUDA/TensorRT needs).
   Select the NVMe SSD if you want the OS installed there. Flash the device (this can
   take 30–60 minutes depending on host and network).

4. After flashing complete, follow on-screen prompts for initial setup on the Jetson
   (create user, set timezone, connect to network). Then update packages:

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### 1.2 Configure Jetson for Development (on device)

1. Enable SSH so you can work headless:

   ```bash
   sudo apt install openssh-server -y
   sudo systemctl enable ssh
   sudo systemctl start ssh
   # Get Jetson's IP
   ip addr show
   ```

2. Optional: configure headless (text-only) mode to reduce memory used by GUI:

   ```bash
   sudo systemctl set-default multi-user.target
   sudo reboot
   ```

3. Create a 16GB swap file on NVMe for build & model conversion steps (adjust path
   if you're using a different disk):

   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

4. Install development tools commonly needed:

   ```bash
   sudo apt install -y \
      git \
      vim \
      tmux \
      htop \
      python3-pip \
      build-essential \
      cmake
   ```

5. Optional useful tools:

- `nvtop` (GPU monitoring), `nvidia-smi` (on Jetson newer BSPs) or `tegrastats` for monitoring
- `docker` if you plan to use containers for reproducibility

---

## Step 2: Set Up ROS2 Environment (Est. 1-2 hours)

### 2.1 Install ROS2 Humble

Follow the official ROS2 installation steps. Example (Ubuntu):

```bash
# Locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS2 apt repository
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble (desktop for full tools)
sudo apt update
sudo apt install ros-humble-desktop -y

# Developer tooling
sudo apt install python3-colcon-common-extensions python3-rosdep -y

# Initialize rosdep
sudo rosdep init || true
rosdep update
```

### 2.2 Configure ROS2 environment

Add sourcing to your shell startup so ROS2 tools are available:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify
ros2 --version
```

## Quick verification checklist

- SSH access works from host
- Enough disk space and swap configured
- `ros2` command is available
- GPU drivers and CUDA (if needed) are present

## Troubleshooting & notes

- If SDK Manager fails to see the Jetson, ensure the device is in recovery mode and USB cable is good.
- For GPU/CUDA/TensorRT concerns, check NVIDIA JetPack / L4T compatibility matrix for the Jetson Orin Nano and the models you plan to run.
- If using NVMe as the boot/install target, confirm BIOS/boot config and NVMe health.

## Where to record changes

- Add any machine-specific notes (IP address, user, serial numbers) to `docs/images` or an internal secure inventory — do not commit secrets.

---

If you want, I can also create a single checklist script under `scripts/setup/jetson_quick_setup.sh` that automates the user-side commands (non-destructive ones like swap creation, package installs and shell updates). Let me know and I will add it.
