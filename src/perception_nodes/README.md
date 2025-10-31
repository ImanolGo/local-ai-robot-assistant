# Camera Pipeline - DeepStream Accelerated

This directory contains the DeepStream-accelerated camera pipeline for the Local AI Robot Assistant project, optimized for NVIDIA Jetson Orin Nano.

## Overview

The camera pipeline consists of two main components:

1. **Camera Driver** (`camera_driver.py`) - Hardware-accelerated camera capture using DeepStream
2. **Image Undistortion Node** (`image_undistort_node.py`) - GPU-accelerated lens distortion correction

## Features

### Camera Driver
- DeepStream pipeline with nvarguscamerasrc for IMX219 camera
- NVMM memory buffers for zero-copy operations
- Hardware-accelerated frame rate control
- Camera info publisher with calibration data
- GPU memory optimization
- Real-time performance monitoring

### Image Undistortion Node
- GPU-accelerated undistortion using OpenCV CUDA
- CPU fallback when GPU not available
- Cached distortion maps for optimal performance
- Configurable interpolation methods
- Real-time performance monitoring
- Memory usage optimization

## Dependencies

### System Dependencies
```bash
# GStreamer with DeepStream support
sudo apt install gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
sudo apt install gstreamer1.0-libav

# Python dependencies
pip install opencv-python opencv-contrib-python
pip install pyyaml numpy psutil
```

### Optional Dependencies
```bash
# For GPU monitoring
pip install pynvml

# For OpenCV CUDA support (if not included in opencv-contrib-python)
# This may require building OpenCV from source with CUDA support
```

## Configuration

### Camera Configuration (`config/camera_config.yaml`)

The camera configuration file contains all settings for the camera pipeline:

```yaml
# Camera hardware settings
camera:
  device_id: 0                # Camera device ID
  sensor_mode: 0              # IMX219 sensor mode
  width: 1640                 # Image width
  height: 1232                # Image height
  framerate: 30               # Target frame rate
  flip_method: 0              # Image rotation

# DeepStream pipeline settings
deepstream:
  source_element: "nvarguscamerasrc"
  nvmm_memory: true           # Use NVMM buffers
  format: "NV12"              # Native format
  buffer_pool_size: 4         # Buffer pool size
  max_buffers: 8              # Maximum buffers

# Undistortion settings
undistortion:
  use_gpu_acceleration: true  # Enable GPU undistortion
  interpolation_method: "linear"
  border_mode: "constant"
  cache_maps: true            # Cache distortion maps
  alpha: 1.0                  # Free scaling parameter
```

### Camera Calibration (`config/camera_calibration.yaml`)

Camera calibration data is required for undistortion:

```yaml
camera_matrix:
  - [fx, 0, cx]
  - [0, fy, cy]
  - [0, 0, 1]
distortion_coefficients:
  - [k1, k2, p1, p2, k3]
image_width: 1640
image_height: 1232
```

## Usage

### Running Individual Nodes

#### Camera Driver
```bash
# Source the workspace
source /home/imanolgo/repos/local-ai-robot-assistant/src/install/setup.bash

# Run camera driver
ros2 run perception_nodes camera_driver
```

#### Image Undistortion Node
```bash
# Run undistortion node
ros2 run perception_nodes image_undistort_node
```

### Running Complete Pipeline

#### Using Launch File
```bash
# Launch complete camera pipeline
ros2 launch perception_nodes camera_pipeline_launch.py

# With custom parameters
ros2 launch perception_nodes camera_pipeline_launch.py use_gpu:=true camera_device:=0
```

#### Manual Launch
```bash
# Terminal 1: Camera driver
ros2 run perception_nodes camera_driver

# Terminal 2: Undistortion node
ros2 run perception_nodes image_undistort_node
```

## Topics

### Published Topics
- `/camera/raw` - Raw camera images (sensor_msgs/Image)
- `/camera/camera_info` - Camera calibration info (sensor_msgs/CameraInfo)
- `/camera/undistorted` - Undistorted images (sensor_msgs/Image)

### Subscribed Topics
- `/camera/raw` - Raw camera images (image_undistort_node)

## Performance Monitoring

Both nodes include built-in performance monitoring:

### Metrics Available
- Frame rate (FPS)
- Processing latency (average, min, max)
- Memory usage
- GPU memory usage (when available)

### Viewing Performance Stats
Performance statistics are logged to the console when `log_performance_stats` is enabled in the configuration.

## Testing

### Unit Tests
```bash
# Run camera driver tests
python -m pytest src/perception_nodes/test/test_camera_driver.py -v

# Run undistortion tests
python -m pytest src/perception_nodes/test/test_image_undistort.py -v
```

### Integration Tests
```bash
# Run complete pipeline integration test
python -m pytest src/perception_nodes/test/test_camera_pipeline_integration.py -v
```

### Performance Benchmarks
```bash
# Run comprehensive benchmark
python scripts/benchmark_camera_pipeline.py

# Save results to file
python scripts/benchmark_camera_pipeline.py --output benchmark_results.json

# Skip GPU benchmarks
python scripts/benchmark_camera_pipeline.py --no-gpu
```

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check if camera is detected
ls /dev/video*

# Test camera with gstreamer
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```

#### GPU Acceleration Not Working
```bash
# Check OpenCV CUDA support
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

# Check NVIDIA driver
nvidia-smi
```

#### High Memory Usage
- Enable map caching (`cache_maps: true`)
- Reduce buffer pool size
- Lower image resolution
- Monitor with benchmark script

#### Low Frame Rate
- Check GPU utilization with `nvidia-smi`
- Reduce image resolution
- Disable unnecessary processing
- Check thermal throttling

### Performance Optimization

#### For High Frame Rate (>30 FPS)
- Use GPU acceleration
- Enable NVMM memory buffers
- Reduce image resolution
- Cache distortion maps

#### For Low Memory Usage
- Disable map caching if memory constrained
- Reduce buffer pool size
- Lower image resolution

#### For Low Latency
- Reduce buffer pool size
- Use linear interpolation
- Enable GPU acceleration

## Hardware Requirements

### Minimum Requirements
- NVIDIA Jetson Orin Nano 8GB
- IMX219 camera module
- Ubuntu 22.04 with JetPack SDK
- 4GB available RAM

### Recommended Requirements
- NVIDIA Jetson Orin Nano 8GB
- IMX219 camera with good lens calibration
- Active cooling solution
- 6GB available RAM
- Fast microSD card or NVMe SSD

## Development

### Adding New Features

When adding new features to the camera pipeline:

1. Update configuration schema in `camera_config.yaml`
2. Add parameters to relevant node class
3. Implement feature with error handling
4. Add unit tests
5. Update integration tests
6. Update documentation

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all functions
- Add comprehensive docstrings
- Include error handling
- Add performance monitoring where applicable

### Testing Guidelines

- Write unit tests for all new functions
- Include integration tests for multi-node features
- Add performance benchmarks for critical paths
- Test error conditions and recovery
- Verify memory usage patterns

## License

Apache-2.0 License - see LICENSE file for details.
