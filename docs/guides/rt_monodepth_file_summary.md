# Complete RT-MonoDepth Integration - File Summary

## Overview
This document provides a complete overview of all files created for RT-MonoDepth integration into your robot assistant project.

---

## 📁 File Structure

```
src/perception_nodes/perception_nodes/depth/
├── __init__.py                         ← Update this
├── rt_monodepth_model.py              ← NEW (Artifact 1)
├── rt_monodepth_preprocessing.py      ← NEW (Artifact 2)
├── rt_monodepth_inference.py          ← NEW (Artifact 3 - Updated)
├── convert_to_tensorrt.py             ← NEW (Artifact 4 - Updated)
├── networks/                          ← Copy from RT-MonoDepth repo
├── layers.py                          ← Copy from RT-MonoDepth repo
└── options.py                         ← Copy from RT-MonoDepth repo

scripts/utils/
└── test_rt_monodepth_standalone.py    ← NEW (Artifact 5)

hardware_tests/
└── test_depth_tensorrt.py             ← NEW (Artifact 6)

models/depth_trt/
├── rt_monodepth_s.pth                 ← Download
└── rt_monodepth_s.engine              ← Generate via conversion
```

---

## 📄 File Descriptions

### 1. `rt_monodepth_model.py` (NEW)
**Purpose**: Model wrapper class that handles architecture loading and weight management.

**Key Features**:
- Clean separation of model concerns from inference logic
- Handles different checkpoint formats automatically
- Supports both 'small' and 'full' model variants
- Provides model information and statistics
- Factory methods for easy model creation

**Key Classes**:
- `RTMonoDepthModel` - Main model wrapper
- `RTMonoDepthModelFactory` - Factory for creating models

**When to use**:
- When you need direct access to the model for training or fine-tuning
- When you want to inspect model architecture
- As a dependency for `rt_monodepth_inference.py`

**Example**:
```python
from perception_nodes.depth.rt_monodepth_model import RTMonoDepthModel

model = RTMonoDepthModel(
    model_variant='small',
    device='cuda',
    pretrained_path='models/depth_trt/rt_monodepth_s.pth'
)
model.print_model_info()
```

---

### 2. `rt_monodepth_preprocessing.py`
**Purpose**: Handles all image preprocessing and postprocessing operations.

**Key Features**:
- Flexible input (file path or numpy array)
- Proper resizing and normalization
- Batch dimension handling
- Depth map postprocessing
- Colormap visualization utilities

**Key Classes**:
- `RTMonoDepthPreprocessor` - Main preprocessing class

**Key Methods**:
- `preprocess_image()` - Prepare image for model input
- `postprocess_depth()` - Convert output to original dimensions
- `depth_to_colormap()` - Create visual depth maps

**When to use**:
- Used internally by `rt_monodepth_inference.py`
- Can be used standalone for preprocessing testing
- Useful for custom pipeline implementations

**Example**:
```python
from perception_nodes.depth.rt_monodepth_preprocessing import RTMonoDepthPreprocessor

preprocessor = RTMonoDepthPreprocessor(input_height=192, input_width=640)
input_tensor, original_shape = preprocessor.preprocess_image(image_path='test.jpg')
```

---

### 3. `rt_monodepth_inference.py` (UPDATED)
**Purpose**: Main inference class supporting both PyTorch and TensorRT backends.

**Key Features**:
- Unified interface for PyTorch and TensorRT
- Automatic backend selection
- Built-in benchmarking
- Batch processing support
- Easy integration with ROS2

**Key Classes**:
- `RTMonoDepthInference` - Main inference engine

**Key Methods**:
- `predict()` - Run inference on single image
- `predict_batch()` - Process multiple images
- `benchmark()` - Performance testing
- `get_model_info()` - Model information

**When to use**:
- This is your main entry point for depth estimation
- Use in ROS2 nodes
- Use in standalone applications
- Use for performance testing

**Example**:
```python
from perception_nodes.depth.rt_monodepth_inference import RTMonoDepthInference

# For production (TensorRT)
inference = RTMonoDepthInference(
    model_path='models/depth_trt/rt_monodepth_s.engine',
    use_tensorrt=True
)

# For development (PyTorch)
inference = RTMonoDepthInference(
    model_path='models/depth_trt/rt_monodepth_s.pth',
    use_tensorrt=False,
    model_variant='small'
)

depth_map, original_shape = inference.predict(image_path='test.jpg')
```

---

### 4. `convert_to_tensorrt.py` (UPDATED)
**Purpose**: Convert PyTorch models to TensorRT engines for optimized inference.

**Key Features**:
- Two conversion methods: torch2trt (simple) and native TensorRT (advanced)
- FP16 optimization support
- Automatic validation after conversion
- Output comparison between PyTorch and TensorRT

**When to use**:
- After downloading PyTorch weights
- Before deploying to Jetson for production
- When you need maximum inference speed

**Usage**:
```bash
# Method 1: torch2trt (recommended)
python convert_to_tensorrt.py \
    --weight_path models/depth_trt/rt_monodepth_s.pth \
    --output_path models/depth_trt/rt_monodepth_s.engine \
    --model_variant small \
    --method torch2trt \
    --fp16 \
    --test_inference

# Method 2: Native TensorRT (more control)
python convert_to_tensorrt.py \
    --weight_path models/depth_trt/rt_monodepth_s.pth \
    --output_path models/depth_trt/rt_monodepth_s.engine \
    --model_variant small \
    --method tensorrt \
    --fp16
```

---

### 5. `test_rt_monodepth_standalone.py`
**Purpose**: Comprehensive standalone testing script.

**Key Features**:
- Single image inference with visualization
- Video stream processing (webcam or file)
- Performance benchmarking
- Side-by-side comparison displays

**Modes**:
- `single` - Test on one image
- `video` - Real-time video processing
- `benchmark` - Performance testing

**When to use**:
- Testing model after conversion
- Validating depth estimation quality
- Benchmarking performance
- Debugging issues

**Usage**:
```bash
# Test single image
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode single \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --image_path test.jpg \
    --output_path output/depth.png \
    --visualize

# Test video stream
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode video \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --video_source 0

# Benchmark
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode benchmark \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --num_iterations 100
```

---

### 6. `test_depth_tensorrt.py`
**Purpose**: Hardware validation script for Jetson testing.

**Key Features**:
- Camera integration testing
- Real-time performance monitoring
- Thermal stability testing
- Complete validation suite

**Test Modes**:
- `all` - Complete validation suite
- `quick` - Camera and inference test only
- `performance` - Extended performance testing
- `thermal` - Thermal stability under load

**When to use**:
- After deploying to Jetson hardware
- Validating camera integration
- Checking thermal performance
- Pre-deployment validation

**Usage**:
```bash
# Complete validation
python hardware_tests/test_depth_tensorrt.py \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --test all

# Quick test
python hardware_tests/test_depth_tensorrt.py \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --test quick

# Performance test (30 seconds)
python hardware_tests/test_depth_tensorrt.py \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --test performance \
    --duration 30
```

---

## 🔄 Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
│         (ROS2 Node / Standalone Script / Test)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  RTMonoDepthInference      │ ← Main entry point
        │  (inference engine)        │
        └──────┬──────────────┬──────┘
               │              │
         PyTorch│              │TensorRT
               │              │
               ▼              ▼
    ┌──────────────┐   ┌──────────────┐
    │RTMonoDepth   │   │  TensorRT    │
    │Model         │   │  Engine      │
    │(wrapper)     │   │              │
    └──────┬───────┘   └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   networks/  │ ← From RT-MonoDepth repo
    │   layers.py  │
    └──────────────┘

    Data flow handled by:
    RTMonoDepthPreprocessor (input/output processing)
```

---

## 🎯 Architecture Benefits

### 1. **Separation of Concerns**
- `rt_monodepth_model.py` - Model architecture and weights
- `rt_monodepth_preprocessing.py` - Data transformation
- `rt_monodepth_inference.py` - Inference orchestration

### 2. **Flexibility**
- Easy switching between PyTorch (dev) and TensorRT (prod)
- Support for different model variants
- Extensible for future model types

### 3. **Maintainability**
- Clear responsibility for each module
- Easy to test individual components
- Simple to update or replace parts

### 4. **ROS2 Integration**
- Clean interface for ROS2 nodes
- Handles ROS image messages natively
- Minimal dependencies in node code

---

## 🚀 Quick Start Workflow

### 1. Setup Files
```bash
# Copy files from artifacts to project
cp rt_monodepth_model.py src/perception_nodes/perception_nodes/depth/
cp rt_monodepth_preprocessing.py src/perception_nodes/perception_nodes/depth/
cp rt_monodepth_inference.py src/perception_nodes/perception_nodes/depth/
cp convert_to_tensorrt.py src/perception_nodes/perception_nodes/depth/
cp test_rt_monodepth_standalone.py scripts/utils/
cp test_depth_tensorrt.py hardware_tests/

# Copy from RT-MonoDepth repo
cp -r ~/RT-MonoDepth/networks src/perception_nodes/perception_nodes/depth/
cp ~/RT-MonoDepth/layers.py src/perception_nodes/perception_nodes/depth/
cp ~/RT-MonoDepth/options.py src/perception_nodes/perception_nodes/depth/
```

### 2. Download Model
```bash
# Download pretrained weights (check RT-MonoDepth repo for link)
cd models/depth_trt
wget [MODEL_URL] -O rt_monodepth_s.pth
```

### 3. Test PyTorch
```bash
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode single \
    --model_path models/depth_trt/rt_monodepth_s.pth \
    --image_path test.jpg \
    --visualize
```

### 4. Convert to TensorRT
```bash
cd src/perception_nodes/perception_nodes/depth
python convert_to_tensorrt.py \
    --weight_path ../../../../models/depth_trt/rt_monodepth_s.pth \
    --output_path ../../../../models/depth_trt/rt_monodepth_s.engine \
    --method torch2trt \
    --fp16 \
    --test_inference
```

### 5. Test TensorRT
```bash
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode benchmark \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt
```

### 6. Hardware Validation
```bash
python hardware_tests/test_depth_tensorrt.py \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --test all
```

---

## 📝 Update `__init__.py`

**File**: `src/perception_nodes/perception_nodes/depth/__init__.py`

```python
"""
RT-MonoDepth depth estimation module.
Provides PyTorch and TensorRT inference for monocular depth estimation.
"""

from .rt_monodepth_model import RTMonoDepthModel, RTMonoDepthModelFactory
from .rt_monodepth_preprocessing import RTMonoDepthPreprocessor
from .rt_monodepth_inference import RTMonoDepthInference

__all__ = [
    'RTMonoDepthModel',
    'RTMonoDepthModelFactory',
    'RTMonoDepthPreprocessor',
    'RTMonoDepthInference',
]

__version__ = '1.0.0'
```

---

## 🔧 ROS2 Integration Example

**Update your `depth_estimator.py` node**:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from perception_nodes.depth.rt_monodepth_inference import RTMonoDepthInference


class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator_node')

        # Declare parameters
        self.declare_parameter('model_path', 'models/depth_trt/rt_monodepth_s.engine')
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('model_variant', 'small')

        # Get parameters
        model_path = self.get_parameter('model_path').value
        use_tensorrt = self.get_parameter('use_tensorrt').value
        model_variant = self.get_parameter('model_variant').value

        # Initialize depth estimator
        self.depth_estimator = RTMonoDepthInference(
            model_path=model_path,
            use_tensorrt=use_tensorrt,
            model_variant=model_variant,
            input_height=192,
            input_width=640,
            device='cuda'
        )

        self.depth_estimator.print_info()

        # Setup ROS interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.depth_pub = self.create_publisher(Image, '/depth/image', 10)

        self.get_logger().info('Depth Estimator Node started')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run inference
            depth_map, _ = self.depth_estimator.predict(image_array=cv_image)

            # Convert depth to ROS message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
            depth_msg.header = msg.header

            # Publish
            self.depth_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## ✅ Summary

You now have a complete, modular RT-MonoDepth integration with:

1. ✅ **Model Wrapper** (`rt_monodepth_model.py`) - Clean model management
2. ✅ **Preprocessing** (`rt_monodepth_preprocessing.py`) - Data handling
3. ✅ **Inference Engine** (`rt_monodepth_inference.py`) - Main interface
4. ✅ **Conversion Tool** (`convert_to_tensorrt.py`) - TensorRT optimization
5. ✅ **Standalone Tests** (`test_rt_monodepth_standalone.py`) - Comprehensive testing
6. ✅ **Hardware Validation** (`test_depth_tensorrt.py`) - Jetson validation

All files work together cohesively while maintaining clean separation of concerns!
