# Hardware Tests

This directory contains hardware validation tests for the Local AI Robot Assistant project.

## Test Scripts

### test_deepstream_install.py
Quick verification that DeepStream SDK and pyds are properly installed.

```bash
python3 hardware_tests/test_deepstream_install.py
```

### test_camera_capture.py
Comprehensive camera testing using NVIDIA DeepStream SDK for hardware-accelerated processing.

**Features:**
- Tests multiple resolutions (1280x720, 1920x1080, 1640x1232)
- Hardware-accelerated capture using nvarguscamerasrc
- GPU memory optimization with NVMM buffers
- Continuous capture testing
- FPS measurement and stability analysis
- Sample image capture for validation
- Balanced camera settings to reduce noise

**Usage:**
```bash
# Run full test suite (takes ~6 minutes)
python3 hardware_tests/test_camera_capture.py

# Quick test mode (faster, takes ~1 minute)
python3 hardware_tests/test_camera_capture.py --quick

# Custom output directory
python3 hardware_tests/test_camera_capture.py --output-dir /path/to/save/images

# Custom camera device
python3 hardware_tests/test_camera_capture.py --device 1

# Custom continuous test duration
python3 hardware_tests/test_camera_capture.py --continuous-duration 120
```

**Requirements:**
- NVIDIA Jetson with DeepStream SDK 7.1+
- IMX219 camera connected to CSI port
- pyds Python bindings installed

### test_waveroever_uart.py
Comprehensive UART communication testing with the Wave Rover robot platform.

**Features:**
- JSON command transmission and response parsing
- Interactive mode for manual testing
- Automated test sequences for validation
- Motor control testing (speed, PWM, ROS-style)
- IMU data retrieval testing
- Continuous feedback mode testing
- OLED display command testing
- PID parameter setting testing
- Real-time response monitoring

**Usage:**
```bash
# Interactive mode (type JSON commands manually)
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1

# Run full automated test suite
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --auto

# Run specific test only
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test motor
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test imu
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test feedback
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test oled

# Custom port and baud rate
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyUSB0 --baud 9600
```

**Supported Commands:**
- **Motor Control**: `{"T":1,"L":0.2,"R":0.2}` - Speed control (-0.5 to +0.5)
- **PWM Input**: `{"T":11,"L":100,"R":100}` - Direct PWM control
- **ROS Control**: `{"T":13,"X":0.05,"Z":0.1}` - Linear/angular velocity
- **IMU Data**: `{"T":126}` - Request IMU readings
- **Continuous Feedback**: `{"T":131,"cmd":1}` - Enable periodic data
- **OLED Display**: `{"T":3,"lineNum":0,"Text":"Hello"}` - Display text
- **PID Settings**: `{"T":2,"P":200,"I":2500,"D":0,"L":255}` - Motor PID

**Requirements:**
- Wave Rover connected via UART (typically `/dev/ttyTHS1` on Jetson)
- Python serial library (`pyserial`)
- Proper UART permissions

### analyze_camera_images.py
Analysis tool for captured camera images to compare quality and settings.

**Usage:**
```bash
python3 hardware_tests/analyze_camera_images.py --dir test_images
```

### correct_color_balance.py
Post-processing tool to correct color balance and reduce red tint in camera images.

**Usage:**
```bash
# Process single image
python3 hardware_tests/correct_color_balance.py image.jpg

# Process entire directory
python3 hardware_tests/correct_color_balance.py test_images/ --output corrected/

# Custom color correction
python3 hardware_tests/correct_color_balance.py image.jpg --red-gain 0.8 --blue-gain 1.2
```

## Prerequisites

Before running camera tests, ensure:

1. DeepStream SDK is installed:
   ```bash
   sudo apt-get install deepstream-7.1
   ```

2. Python dependencies are installed:
   ```bash
   pip install pyds opencv-python numpy
   ```

3. Camera is connected and detected:
   ```bash
   ls /dev/video*
   ```

4. Test with GStreamer directly:
   ```bash
   nvgstcapture-1.0 --camsrc=0 --cap-dev-node=0
   ```

## Test Results

Test results and sample images are saved to the specified output directory (default: `test_images/`).

### Expected Performance

On NVIDIA Jetson Orin Nano with optimized settings:
- 1280x720: 28+ FPS (CROPPED FOV, high frame rate mode)
- 1920x1080: 28+ FPS (CROPPED FOV, standard HD)
- 1640x1232: 28+ FPS (FULL FOV, best for computer vision)

**Note**: Resolutions 640x480 and 3280x2464 are disabled in current implementation due to stability issues.

### Troubleshooting

**Camera not detected:**
- Check physical connections
- Verify with `dmesg | grep imx219`
- Test with `nvgstcapture-1.0`

**DeepStream import errors:**
- Ensure DeepStream SDK is installed
- Check that pyds is available: `python3 -c "import pyds"`
- Verify GStreamer installation: `gst-inspect-1.0 nvarguscamerasrc`

**Low FPS or dropped frames:**
- Check system load: `htop`
- Monitor GPU usage: `tegrastats`
- Ensure adequate cooling
- Check memory usage: `free -h`

**Pipeline errors:**
- Check GStreamer logs for detailed error messages
- Verify camera permissions: `sudo usermod -a -G video $USER`
- Test with different resolutions

**Red tint issues (IMX219 specific):**

The red tint issue on the IMX219 camera when used with NVIDIA Jetson platforms is a known problem often related to improper ISP (Image Signal Processor) tuning and lens shading correction.

*Common Causes:*
- ISP tuning parameters incompatible or missing
- Incorrect white balance
- Lens shading or vignetting effect not calibrated
- IR sensitivity or filter mismatches

*Recommended Fixes:*

Apply ISP tuning override file - Download and install the official ISP tuning parameter file for IMX219 to adjust color correction and lens shading automatically:

```bash
wget https://www.arducam.com/downloads/Jetson/Camera_overrides.tar.gz
tar zxvf Camera_overrides.tar.gz
sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
```

After applying the ISP override file, restart your camera application or reboot the system for changes to take effect.

Alternative software-based color correction is available using the `correct_color_balance.py` tool included in this directory.

## Performance Monitoring

During tests, monitor system resources:

```bash
# GPU and system stats
tegrastats

# Check running processes
htop

# Monitor disk space
df -h
```
