# Jetson Orin Nano — Setup & ROS2 Quick Start

This guide collects the essential steps to flash, configure and prepare a Jetson Orin Nano for development with ROS2 (Humble). It's a concise, copy-pastable checklist you can follow when provisioning a new device.

## Requirements

1. A host computer (Ubuntu recommended) with NVIDIA SDK Manager installed — see the NVIDIA developer site.
2. Jetson Orin Nano and required cables (USB-C for recovery, power, NVMe if applicable).
3. NVMe SSD (if using external storage) and Internet access.

## Step 1 — Initialize Jetson Orin Nano (Est. 2–3 hours)

### 1.1 Flash JetPack (on host)

1. Download and install SDK Manager on the host machine.
1. Put the Jetson into recovery mode and connect it to the host via USB-C.
1. Open SDK Manager, select the Jetson Orin Nano target and choose the desired JetPack version; select NVMe as install target if appropriate. Flash the device (30–60 minutes).
1. After flashing completes, perform the initial device setup and update packages:

```bash
sudo apt update && sudo apt upgrade -y
```


## 2. Install ROS2 Humble (Est. 1–2 hours)

Follow the official ROS2 instructions. Example (Ubuntu):

```bash
# locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# add ROS2 apt repository and install
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop -y
sudo apt install python3-colcon-common-extensions python3-rosdep -y
sudo rosdep init || true
rosdep update
```

Configure your shell to source ROS2 on login:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Verify with:

```bash
ros2 --version
```

---

## 3. Configuring the CSI connector (camera)

Use `jetson-io` to configure the 24-pin CSI connector for camera modules.

1. Check currently-visible video devices:

```bash
ls /dev/video*
```

1. Run the `jetson-io` utility (requires sudo):

```bash
cd /opt/nvidia/jetson-io/
sudo python jetson-io.py
```

1. In the interactive menu choose `Configure Jetson 24 pin CSI Connector` → `Configure for compatible hardware` and select your camera:

- IMX219 = Raspberry Pi V2 camera
- IMX477 = Raspberry Pi V3 camera (use dual-lane / 4-lane CSI for full bandwidth)

1. Choose `Save pin changes` then `Save and reboot to reconfigure pins`. Confirm to reboot.

1. After reboot re-check `/dev/video*` to confirm the camera is available:

```bash
ls /dev/video*
```

---

## Troubleshooting & notes

- If SDK Manager does not detect the Jetson, ensure the device is in recovery mode and the USB cable is connected properly.
- For GPU/CUDA/TensorRT compatibility, consult the JetPack / L4T documentation for your JetPack version.
- If using NVMe as the boot/install target, verify NVMe health and partitioning.

### How to Fix UART Permission Denied

If you get a "Permission denied" (Errno 13) when opening the Jetson serial device (for example `/dev/ttyTHS1`), grant your user access to the standard serial group and re-login.

Add your user to the `dialout` group (standard Unix group for serial devices):

```bash
sudo usermod -aG dialout $USER
```

The change takes effect after you log out and log back in. To apply the group change immediately in the current shell you can run:

```bash
newgrp dialout
```

If you prefer a udev-based rule or need more fine-grained control (for CI or automated setups), create a udev rule under `/etc/udev/rules.d/99-wave-rover-tty.rules` with appropriate ownership/permissions or use `setfacl` to grant access to a specific user. Reboot or reload udev rules after adding a rule.


## Recording machine-specific notes

- Add machine-specific details to an internal inventory (do not commit secrets into the repo).


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

2. **RAM Optimization (Critical for 8GB Jetson Orin Nano)**

   Running LLMs requires significant RAM. On Jetson Orin Nano with only 8GB RAM,
   it's crucial to optimize system memory usage.

   **Disable Desktop GUI (saves ~800MB):**

   If using SSH for development, disable the Ubuntu desktop GUI:

   ```bash
   # Temporary disable (can restart with 'sudo init 5')
   sudo init 3

   # Permanent disable (persistent across reboots)
   sudo systemctl set-default multi-user.target
   sudo reboot

   # To re-enable desktop later:
   # sudo systemctl set-default graphical.target
   ```

   **Disable unnecessary services:**

   ```bash
   sudo systemctl disable nvargus-daemon.service
   ```

3. **Create optimized swap file:**

   First, check your current storage and swap configuration:

   ```bash
   # Check current swap status
   swapon --show

   # Check storage layout to see if you have NVMe
   lsblk
   df -h
   ```

   If you see ZRAM devices, disable them first:
   ```bash
   sudo systemctl disable nvzramconfig
   sudo reboot
   ```

   **For systems with NVMe SSD (mounted as root `/`):**
   ```bash
   # Create swap file on NVMe (via root filesystem)
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

   **For systems with separate NVMe mount (rare):**
   ```bash
   # Only if you have /ssd/ or similar separate mount point
   sudo fallocate -l 16G /ssd/16GB.swap
   sudo chmod 600 /ssd/16GB.swap
   sudo mkswap /ssd/16GB.swap
   sudo swapon /ssd/16GB.swap
   echo '/ssd/16GB.swap none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

   **Verify optimal configuration:**
   ```bash
   swapon --show
   free -h
   ```
   Target: Only one swap file, no ZRAM devices, maximum available RAM.

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

## Configuring the CSI Connector

The `jetson-io` tool can configure the GPIO header, the CSI connectors and the M.2 Key E connector for use with the Jetson. Here we'll work with the CSI connector. To start, check that the camera is available — it should report as `/dev/video*` where `*` is a number:

```bash
ls /dev/video*
```

Next, start the `jetson-io` utility:

```bash
cd /opt/nvidia/jetson-io/
sudo python jetson-io.py
```

Once the script starts select `Configure Jetson 24 pin CSI Connector` then `Configure for compatible hardware`. Remember that the IMX219 is the RPi V2 camera and the IMX477 is the RPi V3. Dual lane uses the CSI 4 lane for the IMX477. You can mix and match cameras. Make your selection then choose `Save pin changes` and `Save and reboot to reconfigure pins`. Confirm and the Jetson will reboot.

After reboot the camera(s) will appear as `/dev/video*` again. Re-run the `ls /dev/video*` check to confirm.
