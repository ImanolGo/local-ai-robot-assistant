# RT-MonoDepth Integration Setup Guide

This guide walks you through integrating RT-MonoDepth into your robot assistant project.

## Step 1: Clone RT-MonoDepth Repository

```bash
cd ~/
git clone https://github.com/Ecalpal/RT-MonoDepth.git
```

## Step 2: Copy Required Files

### Create Depth Module Structure
```bash
# Create the depth subfolder
mkdir -p src/perception_nodes/perception_nodes/depth

# Create __init__.py to make it a Python package
touch src/perception_nodes/perception_nodes/depth/__init__.py
```

### Copy Model Architecture
```bash
# Navigate to your project
cd ~/repos/local-ai-robot-assistant

# Copy the networks folder (contains model architecture)
cp -r ~/RT-MonoDepth/networks src/perception_nodes/perception_nodes/depth/

# Copy layers.py (custom PyTorch layers)
cp ~/RT-MonoDepth/layers.py src/perception_nodes/perception_nodes/depth/

# Copy options.py (configuration)
cp ~/RT-MonoDepth/options.py src/perception_nodes/perception_nodes/depth/
```



### Copy the Inference Files (from artifacts above)
Create these files in `src/perception_nodes/perception_nodes/depth/`:
- `rt_monodepth_preprocessing.py`
- `rt_monodepth_inference.py`
- `convert_to_tensorrt.py`

### Copy Test Scripts
Create these in `scripts/utils/`:
- `test_rt_monodepth_standalone.py`

## Step 3: Download Model Weights

```bash
# Download pre-trained weights (example - check RT-MonoDepth repo for actual links)
cd models/depth_trt

# Download small model weights
wget https://path-to-weights/rt_monodepth_s.pth

# Or download from Google Drive if that's where they're hosted
# Use gdown or manual download
```

## Step 4: Install Dependencies

```bash
# Install PyTorch dependencies (if not already installed)
pip install torch torchvision

# Install OpenCV for image processing
pip install opencv-python

# Install additional dependencies
pip install pillow scikit-image matplotlib

# For TensorRT conversion (Method 1 - Recommended)
pip install torch2trt

# For TensorRT conversion (Method 2 - More control)
# TensorRT is typically pre-installed on Jetson
# If needed: pip install tensorrt pycuda
```

## Step 5: Fix Import Paths

Update the `__init__.py` files to enable imports:

**src/perception_nodes/perception_nodes/depth/__init__.py:**
```python
from .rt_monodepth_inference import RTMonoDepthInference
from .rt_monodepth_preprocessing import RTMonoDepthPreprocessor

__all__ = ['RTMonoDepthInference', 'RTMonoDepthPreprocessor']
```

**Fix imports in convert_to_tensorrt.py:**
```python
# Change this line:
# from .networks import RTMonoDepth_S

# To this:
from networks.rtmonodepth import RTMonoDepth_S  # Adjust based on actual structure
```

## Step 6: Test PyTorch Inference

```bash
# Test with a sample image
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode single \
    --model_path models/depth_trt/rt_monodepth_s.pth \
    --image_path /path/to/test/image.jpg \
    --output_path outputs/depth_test.png \
    --visualize
```

## Step 7: Convert to TensorRT

### Method 1: Using torch2trt (Recommended)
```bash
cd src/perception_nodes/perception_nodes/depth

python convert_to_tensorrt.py \
    --weight_path ../../../../models/depth_trt/rt_monodepth_s.pth \
    --output_path ../../../../models/depth_trt/rt_monodepth_s.engine \
    --method torch2trt \
    --fp16 \
    --test_inference
```

### Method 2: Using Native TensorRT
```bash
python convert_to_tensorrt.py \
    --weight_path ../../../../models/depth_trt/rt_monodepth_s.pth \
    --output_path ../../../../models/depth_trt/rt_monodepth_s.engine \
    --method tensorrt \
    --fp16
```

## Step 8: Test TensorRT Inference

```bash
# Test TensorRT model
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode single \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --image_path /path/to/test/image.jpg \
    --output_path outputs/depth_test_trt.png \
    --visualize
```

## Step 9: Benchmark Performance

```bash
# Benchmark PyTorch
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode benchmark \
    --model_path models/depth_trt/rt_monodepth_s.pth \
    --num_iterations 100

# Benchmark TensorRT
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode benchmark \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --num_iterations 100
```

## Step 10: Test Video Stream

```bash
# Test with webcam
python scripts/utils/test_rt_monodepth_standalone.py \
    --mode video \
    --model_path models/depth_trt/rt_monodepth_s.engine \
    --use_tensorrt \
    --video_source 0
```

## Step 11: Integrate with ROS2 Node

Update your `depth_estimator.py` node to use the new inference class:

```python
from perception_nodes.depth.rt_monodepth_inference import RTMonoDepthInference

class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Load model
        self.depth_estimator = RTMonoDepthInference(
            model_path='/path/to/rt_monodepth_s.engine',
            use_tensorrt=True,
            input_height=192,
            input_width=640,
            device='cuda'
        )

    def image_callback(self, msg):
        # Convert ROS image to numpy
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run inference
        depth_map, original_shape = self.depth_estimator.predict(
            image_array=cv_image
        )

        # Publish depth map
        # ... your publishing logic ...
```

## Troubleshooting

### Issue: Import errors for networks
**Solution:** Make sure you've copied the entire `networks/` folder and that `layers.py` is in the same directory.

### Issue: CUDA out of memory
**Solution:**
- Reduce input resolution: `--input_height 128 --input_width 416`
- Use FP16 mode: `--fp16`
- Close other GPU applications

### Issue: TensorRT conversion fails
**Solution:**
- Try torch2trt method first (simpler)
- Check TensorRT version compatibility
- Ensure model loads correctly in PyTorch first

### Issue: Slow inference on Jetson
**Solution:**
- Ensure you're using TensorRT, not PyTorch
- Enable FP16 mode
- Set Jetson to MAX performance: `sudo nvpmodel -m 0 && sudo jetson_clocks`

### Issue: Model architecture mismatch
**Solution:**
- Verify you're using the correct model class (RTMonoDepth_S vs RTMonoDepth_Full)
- Check that checkpoint format matches expected format in code

## Performance Expectations

### Jetson Orin Nano 8GB:
- **PyTorch FP32**: ~15-20 FPS at 640x192
- **TensorRT FP16**: ~60-80 FPS at 640x192

### Jetson AGX Orin:
- **PyTorch FP32**: ~25-30 FPS at 640x192
- **TensorRT FP16**: ~120-150 FPS at 640x192

## Next Steps

1. Create hardware test: `hardware_tests/test_depth_tensorrt.py`
2. Add to launch files: `launch/perception_launch.py`
3. Configure parameters: `config/perception_config.yaml`
4. Add unit tests: `src/perception_nodes/test/test_depth_estimator.py`

## File Summary

### Files Created:
```
src/perception_nodes/perception_nodes/depth/
├── __init__.py
├── rt_monodepth_preprocessing.py
├── rt_monodepth_inference.py
├── convert_to_tensorrt.py
├── networks/                    # Copied from RT-MonoDepth
├── layers.py                    # Copied from RT-MonoDepth
└── options.py                   # Copied from RT-MonoDepth

scripts/utils/
└── test_rt_monodepth_standalone.py

models/depth_trt/
├── rt_monodepth_s.pth          # Downloaded weights
└── rt_monodepth_s.engine       # Converted TensorRT engine
```

### Reference Files from RT-MonoDepth:
- `test_simple_s.py` - Reference for how to run inference
- `utils.py` - Contains preprocessing functions (adapted into our code)
- `networks/` - Model architecture definitions
- `layers.py` - Custom layer implementations
- `options.py` - Configuration parameters
