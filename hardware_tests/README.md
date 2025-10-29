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
- Tests all 6 sensor modes (0-5) with different resolutions and FOV settings
- Hardware-accelerated capture using nvarguscamerasrc and NVMM buffers
- Performance measurement with latency and FPS analysis
- Sample image capture for quality validation
- Configurable device ID, output directory, and frame count
- Individual sensor mode testing or complete test suite
- Verbose logging for debugging

**Usage:**
```bash
# Run full test suite (all sensor modes)
python3 hardware_tests/test_camera_capture.py

# Test specific sensor mode only
python3 hardware_tests/test_camera_capture.py --mode 3

# Custom output directory
python3 hardware_tests/test_camera_capture.py --output-dir /path/to/save/images

# Use different camera device
python3 hardware_tests/test_camera_capture.py --device 1

# Capture more frames per test
python3 hardware_tests/test_camera_capture.py --frames 10

# Enable verbose debug logging
python3 hardware_tests/test_camera_capture.py --verbose
```

**Sensor Modes:**
- **Mode 0**: 3280x2464 @ 21fps (8MP full resolution, native)
- **Mode 1**: 3280x1848 @ 28fps (6MP wide, native aspect)
- **Mode 2**: 1920x1080 @ 30fps (2MP HD cropped center)
- **Mode 3**: 1640x1232 @ 30fps (2MP full FOV) - **Recommended for CV**
- **Mode 4**: 1280x720 @ 60fps (1MP HD cropped)
- **Mode 5**: 820x616 @ 60fps (0.25x downscaled)

**Requirements:**
- NVIDIA Jetson with DeepStream SDK 7.1+
- IMX219 camera connected to CSI port
- Python packages: gi, numpy, Pillow

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

## Camera Calibration (Critical for Fisheye Camera)

### calibrate_camera.py

**CRITICAL for fisheye camera setup** - DeepStream-based camera calibration for IMX219 fisheye distortion correction.

**Features:**

- Hardware-accelerated capture using NVIDIA DeepStream
- Interactive calibration with audio guidance for headless operation
- Automatic USB speaker detection for audio feedback
- Checkerboard detection with subpixel accuracy
- Real-time preview and capture guidance
- Comprehensive calibration quality validation
- Saves calibration parameters to `config/camera_calibration.yaml`

**Step-by-Step Calibration Process:**

**1. Download and Print Checkerboard Pattern:**

```bash
# Download official OpenCV calibration pattern
mkdir -p hardware_tests/calibration_pattern
cd hardware_tests/calibration_pattern
wget https://github.com/opencv/opencv/raw/master/doc/pattern.png
```

**Print the pattern:**

- Print `pattern.png` on A4 paper at **actual size** (100% scale, no fit-to-page)
- Use high-quality laser printer for sharp edges
- Mount on rigid surface (cardboard/clipboard) to prevent bending
- Measure actual square size with ruler (should be ~25mm for default settings)

**2. Run Camera Calibration:**

```bash
# Basic calibration with default settings (9x6 checkerboard, 25 images)
python3 hardware_tests/calibrate_camera.py

# Custom checkerboard configuration (if using different pattern)
python3 hardware_tests/calibrate_camera.py --cols 7 --rows 5 --square-size 30.0

# Capture more images for better accuracy
python3 hardware_tests/calibrate_camera.py --num-images 30

# Use different camera device
python3 hardware_tests/calibrate_camera.py --device 1

# Verbose output for debugging
python3 hardware_tests/calibrate_camera.py --verbose
```

**3. Calibration Capture Guidelines:**

The script will guide you through capturing calibration images. For best results:

- **Cover all image areas**: center, corners, edges
- **Vary distances**: close (30cm), medium (60cm), far (100cm+)
- **Try different angles**:
  - Straight on (perpendicular)
  - Tilted left/right (±30°)
  - Tilted up/down (±30°)
  - Rotated views (±45°)
- **Ensure good lighting**: avoid shadows on checkerboard
- **Keep checkerboard flat**: no bending or warping
- **Wait for "DETECTED ✓"** before pressing ENTER

**Audio feedback** (if USB speakers connected):

- Beep when checkerboard detected
- Voice prompts for capture progress
- Quality notifications

### test_undistortion.py

Test and validate camera calibration by comparing original vs. undistorted images.

**Features:**

- Hardware-accelerated undistortion using DeepStream
- Live camera testing mode
- Batch processing of calibration images
- Side-by-side visual comparisons
- Grid overlay for distortion assessment
- Comprehensive quality analysis and reporting

**Usage:**

```bash
# Test with live camera (recommended first test)
python3 hardware_tests/test_undistortion.py

# Test on existing calibration images
python3 hardware_tests/test_undistortion.py --mode existing

# Test both existing and live images
python3 hardware_tests/test_undistortion.py --mode both

# Use custom calibration file
python3 hardware_tests/test_undistortion.py --calibration config/custom_calibration.yaml

# Custom input/output directories
python3 hardware_tests/test_undistortion.py --input-dir my_images/ --output-dir results/
```

**Validation Guidelines:**

After running undistortion tests, review the generated images:

- **`*_comparison.jpg`**: Side-by-side original vs. corrected
- **`*_grid_comparison.jpg`**: Grid overlays showing line straightness
- **Look for**:
  - ✅ Straight lines appear straighter in undistorted images
  - ✅ Reduced barrel/pincushion distortion at edges
  - ✅ Grid lines more parallel and perpendicular
  - ✅ Better geometric accuracy overall

**Quality Assessment:**

- **Good calibration**: Reprojection error < 0.5 pixels
- **Acceptable**: Reprojection error < 1.0 pixels
- **Poor**: Reprojection error > 1.0 pixels (recalibrate recommended)

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

## Camera Performance Benchmarks

### Expected Performance

On NVIDIA Jetson Orin Nano with DeepStream SDK 7.1:

- **Mode 0**: 3280x2464 @ 21fps → ~5500fps, 0.18ms avg (full FOV, 8MP native)
- **Mode 1**: 3280x1848 @ 28fps → ~14750fps, 0.07ms avg (full FOV, 6MP wide)
- **Mode 2**: 1920x1080 @ 30fps → ~14500fps, 0.07ms avg (cropped FOV, 2MP HD)
- **Mode 3**: 1640x1232 @ 30fps → ~15275fps, 0.07ms avg (full FOV, 2MP) - **Recommended**
- **Mode 4**: 1280x720 @ 60fps → ~15300fps, 0.07ms avg (cropped FOV, 1MP HD)
- **Mode 5**: 820x616 @ 60fps → ~16500fps, 0.06ms avg (scaled FOV, quarter-res) - **Optimal**

**Note**: The extremely high measured FPS values indicate hardware-accelerated frame capture with minimal latency. Mode 3 (1640x1232) provides the best balance of full FOV and high performance for computer vision applications, while Mode 5 achieves optimal performance for high-speed processing scenarios.

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
