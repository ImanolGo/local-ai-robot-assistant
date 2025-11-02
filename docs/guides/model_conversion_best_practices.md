# Model Conversion Best Practices
## NVIDIA Jetson Orin Nano Deployment Guide

This document provides comprehensive guidance for converting AI models for deployment on the NVIDIA Jetson Orin Nano, following the Local AI Robot Assistant architecture specifications.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Conversion Pipeline](#conversion-pipeline)
4. [Model-Specific Guidelines](#model-specific-guidelines)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

### Architecture Requirements

According to the project architecture, all AI models must follow this optimization path:

```
PyTorch/HuggingFace → ONNX (intermediate) → TensorRT Engine (deployment)
```

### Performance Targets

| Model Type | Target FPS | Resolution | Memory Budget |
|------------|------------|------------|---------------|
| YOLO Object Detection | 20+ FPS | 640x480 | 600 MB |
| FastDepth Estimation | 15+ FPS | 320x240 | 400 MB |
| Whisper ASR | RTF < 0.3x | Audio | 500 MB |
| NanoLLM | < 3s latency | Text | 2.5 GB |

### System Constraints

- **Total RAM**: 8GB (7.5GB usable)
- **GPU Memory**: Shared with system RAM
- **Storage**: NVMe SSD recommended
- **Precision**: FP16 preferred, INT8 for extreme optimization

## Prerequisites

### Hardware Requirements

- NVIDIA Jetson Orin Nano Developer Kit (8GB RAM)
- NVMe SSD with at least 256GB storage
- 16GB swap file configured on NVMe

### Software Dependencies

#### 1. NVIDIA JetPack SDK
```bash
# Should be pre-installed, verify version
cat /etc/nv_tegra_release
# Recommended: JetPack 5.x or 6.x
```

#### 2. CUDA and TensorRT
```bash
# Verify CUDA installation
nvcc --version

# Test TensorRT
python3 tools/test_tensorrt.py
```

#### 3. PyTorch for Jetson
Install PyTorch wheel specifically built for Jetson:

```bash
# Check available PyTorch wheels at:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson

# Example for PyTorch 2.0.0 (adjust URL for latest)
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.0.0a0+fe05266f-cp310-cp310-linux_aarch64.whl
pip3 install torch-2.0.0a0+fe05266f-cp310-cp310-linux_aarch64.whl

# Install torchvision
pip3 install torchvision
```

#### 4. ONNX and ONNX Runtime
```bash
# Install ONNX
pip3 install onnx

# Install ONNX Runtime GPU for Jetson
pip3 install onnxruntime-gpu
```

#### 5. Model-Specific Dependencies
```bash
# For YOLO conversion
pip3 install ultralytics

# For Whisper conversion
pip3 install openai-whisper faster-whisper

# For profiling and utilities
pip3 install psutil nvidia-ml-py3 matplotlib seaborn tabulate
```

## Conversion Pipeline

### Step 1: Prepare PyTorch Model

#### From HuggingFace
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model_name")
model.eval()
torch.save(model.state_dict(), "model.pt")
```

#### From Ultralytics (YOLO)
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# Model automatically downloaded and ready for export
```

### Step 2: Convert to ONNX

#### Standard PyTorch Export
```python
import torch

# Load model
model = load_your_model()
model.eval()

# Create dummy input matching your target shape
dummy_input = torch.randn(1, 3, 640, 480)  # Adjust for your model

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,          # Compatible with TensorRT
    do_constant_folding=True,  # Optimize constants
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={             # Optional: dynamic batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

#### Verify ONNX Model
```python
import onnx
import onnxruntime as ort

# Load and verify
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Test inference
session = ort.InferenceSession("model.onnx")
dummy_input = np.random.randn(1, 3, 640, 480).astype(np.float32)
output = session.run(None, {'input': dummy_input})
print(f"ONNX output shape: {output[0].shape}")
```

### Step 3: Convert to TensorRT

#### Using trtexec (Command Line)
```bash
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_fp16.trt \
    --fp16 \
    --workspace=256 \
    --verbose \
    --shapes=input:1x3x640x480
```

#### Using Python API (Recommended)
```python
import tensorrt as trt

# Create logger and builder
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# Create network and parser
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("model.onnx", 'rb') as model_file:
    if not parser.parse(model_file.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))

# Configure builder
config = builder.create_builder_config()
config.max_workspace_size = 256 * 1024 * 1024  # 256MB
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16

# Build and save engine
engine = builder.build_engine(network, config)
with open("model_fp16.trt", 'wb') as f:
    f.write(engine.serialize())
```

## Model-Specific Guidelines

### YOLO Object Detection

#### Conversion Command
```bash
python3 tools/conversion/convert_yolo.py \
    --model yolov8n \
    --output-dir ./models/yolo_trt \
    --input-size 640 480 \
    --precision fp16
```

#### Optimization Tips
- Use YOLOv8n (nano) for best performance on Jetson
- Input size 640x480 balances accuracy and speed
- Consider 416x416 for higher FPS if needed
- Enable NMS (Non-Maximum Suppression) optimization in TensorRT

#### Integration in ROS2
```python
# In perception node
class YOLODetector:
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()

    def detect(self, image):
        # Preprocess image to 640x480
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.infer(input_tensor)

        # Post-process detections
        detections = self.postprocess(outputs)
        return detections
```

### FastDepth Monocular Depth

#### Conversion Command
```bash
python3 tools/conversion/convert_depth.py \
    --output-dir ./models/depth_trt \
    --input-size 320 240 \
    --precision fp16
```

#### Optimization Tips
- Target resolution 320x240 for 15+ FPS
- Use FP16 precision for 2x memory reduction
- Consider depth range clipping (0.1m to 10m)
- Implement bilateral filtering for noise reduction

#### Integration in ROS2
```python
class DepthEstimator:
    def __init__(self, engine_path, camera_matrix):
        self.engine = self.load_engine(engine_path)
        self.camera_matrix = camera_matrix

    def estimate_depth(self, rgb_image):
        # Resize to 320x240
        resized = cv2.resize(rgb_image, (320, 240))

        # Normalize
        normalized = self.normalize_image(resized)

        # Inference
        depth_map = self.infer(normalized)

        # Convert to point cloud if needed
        points = self.depth_to_pointcloud(depth_map)
        return depth_map, points
```

### Whisper Speech Recognition

#### Conversion Command (faster-whisper recommended)
```bash
python3 tools/conversion/convert_whisper.py \
    --model-size tiny \
    --conversion-type faster-whisper \
    --quantization int8 \
    --output-dir ./models/whisper
```

#### Optimization Tips
- Use faster-whisper over custom TensorRT for Whisper
- INT8 quantization provides best speed/accuracy balance
- Implement voice activity detection (VAD) preprocessing
- Consider streaming inference for real-time processing

#### Integration in ROS2
```python
from faster_whisper import WhisperModel

class STTNode:
    def __init__(self, model_path):
        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="int8"
        )

    def transcribe(self, audio_data):
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=1,  # Faster inference
            language="en"
        )

        text = " ".join([segment.text for segment in segments])
        return text
```

## Performance Optimization

### Memory Management

#### Dynamic Model Loading
```python
class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.memory_threshold = 0.85  # 85% RAM usage

    def load_model(self, model_name, model_path):
        # Check memory usage
        if psutil.virtual_memory().percent > self.memory_threshold * 100:
            self.unload_least_used_model()

        # Load model
        self.loaded_models[model_name] = self.load_engine(model_path)

    def unload_model(self, model_name):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()  # Clear GPU cache
```

#### Model Switching Strategy
1. **Perception Mode** (default): SLAM + YOLO + Depth active
2. **Reasoning Mode**: Load LLM, optionally unload vision models
3. **Emergency Mode**: Keep only essential models

### GPU Optimization

#### CUDA Streams for Parallel Processing
```python
import pycuda.driver as cuda

class ParallelInference:
    def __init__(self):
        self.stream1 = cuda.Stream()
        self.stream2 = cuda.Stream()

    def parallel_inference(self, yolo_input, depth_input):
        # Run YOLO and depth estimation in parallel
        yolo_future = self.async_inference(self.yolo_engine, yolo_input, self.stream1)
        depth_future = self.async_inference(self.depth_engine, depth_input, self.stream2)

        # Synchronize and get results
        yolo_result = yolo_future.result()
        depth_result = depth_future.result()

        return yolo_result, depth_result
```

#### Memory Pool Management
```python
# Pre-allocate GPU memory pools
def setup_memory_pools():
    import pycuda.tools

    # Create memory pool to avoid frequent allocations
    mem_pool = pycuda.tools.MemoryPool(pycuda.tools.DeviceMemoryPool())

    # Pre-allocate common buffer sizes
    common_sizes = [
        640 * 480 * 3 * 4,  # YOLO input
        320 * 240 * 3 * 4,  # Depth input
        1024 * 1024 * 4     # General buffer
    ]

    for size in common_sizes:
        mem_pool.allocate(size)
```

### Thermal Management

#### Monitoring and Throttling
```python
class ThermalManager:
    def __init__(self):
        self.temp_warning = 75  # °C
        self.temp_throttle = 80  # °C
        self.temp_emergency = 85  # °C

    def check_thermal_state(self):
        gpu_temp = self.get_gpu_temperature()

        if gpu_temp > self.temp_emergency:
            return "emergency"  # Disable all AI models
        elif gpu_temp > self.temp_throttle:
            return "throttle"   # Reduce inference frequency
        elif gpu_temp > self.temp_warning:
            return "warning"    # Log warning
        else:
            return "normal"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ONNX Parsing Errors
```
Error: Failed to parse ONNX model
```

**Solutions:**
- Check ONNX opset version (use 11 for TensorRT compatibility)
- Validate ONNX model with `onnx.checker.check_model()`
- Try simplifying ONNX model with `onnxsim`

```bash
pip3 install onnxsim
onnxsim model.onnx model_simplified.onnx
```

#### 2. TensorRT Build Failures
```
Error: Failed to build TensorRT engine
```

**Solutions:**
- Increase workspace size (try 512MB or 1GB)
- Check input shapes match ONNX model
- Disable FP16 if causing issues (use FP32)
- Verify TensorRT version compatibility

#### 3. Memory Allocation Errors
```
Error: CUDA out of memory
```

**Solutions:**
- Reduce batch size to 1
- Use dynamic model loading
- Clear GPU cache: `torch.cuda.empty_cache()`
- Check swap file configuration

#### 4. Performance Below Target
**Diagnosis:**
```python
# Profile model performance
python3 tools/benchmarking/profile_model.py --models-dir ./models
```

**Solutions:**
- Use INT8 quantization for critical models
- Optimize input resolution
- Enable TensorRT graph optimization
- Check for CPU bottlenecks in preprocessing

### Debug Tools

#### TensorRT Verbose Logging
```python
# Enable detailed TensorRT logs
logger = trt.Logger(trt.Logger.VERBOSE)
```

#### NVIDIA System Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor power consumption
tegrastats

# Check thermal status
cat /sys/class/thermal/thermal_zone*/temp
```

#### Memory Profiling
```python
import psutil
import nvidia_ml_py3 as nvml

def print_memory_usage():
    # System memory
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB ({mem.percent:.1f}%)")

    # GPU memory
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    gpu_mem = nvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU RAM: {gpu_mem.used/1024**3:.1f}GB / {gpu_mem.total/1024**3:.1f}GB")
```

## Best Practices

### Development Workflow

1. **Start with Largest Model**: Convert the largest model (usually LLM) first to understand memory constraints
2. **Profile Early**: Benchmark each model individually before integration
3. **Incremental Testing**: Test each conversion step (PyTorch → ONNX → TensorRT)
4. **Version Control**: Keep conversion scripts and model metadata in git

### Production Deployment

1. **Model Registry**: Maintain a registry of converted models with versions
2. **Health Checks**: Implement model health monitoring in ROS2 nodes
3. **Fallback Models**: Have simpler fallback models for emergency situations
4. **Automated Testing**: Set up CI/CD for model conversion and validation

### Optimization Checklist

- [ ] Use FP16 precision for all vision models
- [ ] Implement dynamic model loading based on RAM usage
- [ ] Profile thermal performance under sustained load
- [ ] Test with realistic input data distributions
- [ ] Validate accuracy after quantization
- [ ] Document performance characteristics
- [ ] Set up monitoring and alerting

### File Organization

```
models/
├── yolo_trt/
│   ├── yolov8n_fp16.trt
│   ├── yolov8n_metadata.json
│   └── yolov8n_benchmark.json
├── depth_trt/
│   ├── fastdepth_fp16.trt
│   ├── fastdepth_metadata.json
│   └── fastdepth_benchmark.json
└── whisper/
    ├── faster_whisper_tiny_int8/
    │   ├── config.json
    │   └── model.bin
    └── whisper_tiny_benchmark.json
```

## Conclusion

Following these best practices ensures optimal model performance on the Jetson Orin Nano while maintaining the architecture requirements of the Local AI Robot Assistant project. Regular profiling and monitoring help maintain performance targets as the system evolves.

For specific model conversion, use the provided scripts in the `tools/conversion/` directory. For performance analysis, use the profiling utilities in `tools/benchmarking/`.

## References

- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Jetson AI Lab Documentation](https://jetson-ai-lab.com/)
- [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [ONNX Documentation](https://onnx.ai/onnx/)

---

**Last Updated**: November 2025
**Version**: 1.0
**Target Platform**: NVIDIA Jetson Orin Nano (8GB)
