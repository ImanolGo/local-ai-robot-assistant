#!/usr/bin/env python3
"""
Enhanced Depth Anything V2 Test Script
Tests ONNX and TensorRT models with comprehensive metrics
Fixed for TensorRT 10.x on Jetson Orin Nano Super
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with some patterns"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 300), 80, (0, 255, 0), -1)
    cv2.line(image, (0, height // 2), (width, height // 2), (0, 0, 255), 5)
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    return image


class ONNXDepthModel:
    """ONNX Runtime Depth Anything V2 model wrapper"""

    def __init__(self, onnx_path: str, config_path: str = None, use_gpu: bool = True):
        """Initialize ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime not available")

        print(f"Loading ONNX model from {onnx_path}...")

        # Setup providers - GPU by default
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Load config if available
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        provider_used = self.session.get_providers()[0]
        print(f"✓ ONNX model loaded with {provider_used}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name}")

        # Performance tracking
        self.inference_times = []

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size (518x518 for Depth Anything V2)
        target_size = (518, 518)
        image_resized = cv2.resize(image_rgb, target_size)

        # Normalize to float32
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_float - mean) / std

        # Add batch dimension and transpose to NCHW
        image_tensor = np.transpose(image_norm, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0).astype(np.float32)

        return image_batch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference and return depth map"""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference with timing
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = time.time() - start_time

        self.inference_times.append(inference_time)

        depth_map = outputs[0].squeeze()
        return depth_map

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colored visualization"""
        # Normalize to 0-255
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return depth_colored

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "std_inference_time": np.std(times),
            "avg_fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
        }

    def cleanup(self):
        """Cleanup resources"""
        pass


class TensorRTDepthModel:
    """TensorRT Depth Anything V2 model wrapper for Jetson Orin Nano

    Fixed for TensorRT 10.x with proper CUDA context management
    """

    def __init__(self, engine_path: str, config_path: str = None):
        """Initialize TensorRT model"""
        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT or PyCUDA not available. Install with: pip install pycuda")

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        print(f"Loading TensorRT engine from {engine_path}...")

        # Initialize CUDA context manually (don't use autoinit)
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get tensor names (TensorRT 10.x)
        self.num_io_tensors = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io_tensors)]

        # Identify input and output tensors
        self.input_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

        if not self.input_names or not self.output_names:
            raise RuntimeError("Could not identify input/output tensors")

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]

        # Get shapes
        input_shape = self.engine.get_tensor_shape(self.input_name)
        output_shape = self.engine.get_tensor_shape(self.output_name)

        # Calculate sizes
        self.input_size = trt.volume(input_shape)
        self.output_size = trt.volume(output_shape)

        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.input_size * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(self.output_size * np.dtype(np.float32).itemsize)

        # Create stream
        self.stream = cuda.Stream()

        # Load configuration
        self.config = {}
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)

        print("✓ TensorRT engine loaded successfully")
        print(f"  Input: {self.input_name} {input_shape}")
        print(f"  Output: {self.output_name} {output_shape}")

        # Performance tracking
        self.inference_times = []

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT model"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        target_size = (518, 518)
        image_resized = cv2.resize(image_rgb, target_size)

        # Normalize to float32
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_float - mean) / std

        # Transform to NCHW format with batch dimension
        image_tensor = np.transpose(image_norm, (2, 0, 1))
        image_batch = np.expand_dims(image_tensor, axis=0).astype(np.float32)

        return image_batch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run TensorRT inference"""
        import pycuda.driver as cuda

        # Ensure CUDA context is active
        self.cuda_ctx.push()

        try:
            # Preprocess input
            input_tensor = self.preprocess(image)
            input_tensor = np.ascontiguousarray(input_tensor)

            # Time inference
            start_time = time.time()

            # Copy input to device
            cuda.memcpy_htod_async(self.d_input, input_tensor, self.stream)

            # Set tensor addresses for TensorRT 10.x
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))

            # Execute
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Prepare output buffer
            output_array = np.empty((1, 518, 518), dtype=np.float32)

            # Copy result from device
            cuda.memcpy_dtoh_async(output_array, self.d_output, self.stream)

            # Synchronize stream
            self.stream.synchronize()

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Extract depth map
            depth_map = output_array.squeeze()

            return depth_map

        finally:
            # Pop CUDA context
            self.cuda_ctx.pop()

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colored visualization"""
        # Check for valid depth map
        if depth_map.max() == depth_map.min():
            print("⚠ Warning: Depth map has no variation (all same values)")
            # Create a placeholder visualization
            depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            # Normalize to 0-255 range
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap for visualization
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return depth_colored

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "std_inference_time": np.std(times),
            "avg_fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
        }

    def cleanup(self):
        """Cleanup TensorRT and CUDA resources"""
        try:
            if hasattr(self, "d_input"):
                self.d_input.free()
            if hasattr(self, "d_output"):
                self.d_output.free()
            if hasattr(self, "context"):
                del self.context
            if hasattr(self, "engine"):
                del self.engine
            if hasattr(self, "runtime"):
                del self.runtime
            if hasattr(self, "cuda_ctx"):
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


class HuggingFaceDepthModel:
    """HuggingFace model wrapper - tries to load efficiently"""

    def __init__(self, model_path: str, use_cpu_only: bool = True):
        """Initialize HuggingFace model"""
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            raise ImportError("PyTorch and transformers not available")

        print("Loading HuggingFace model...")

        # Force CPU to avoid GPU memory issues
        self.device = "cpu" if use_cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Try the direct hub model name
            model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map="cpu" if use_cpu_only else "auto",
            )
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            print(f"✓ Loaded from hub: {model_name}")
        except Exception as e:
            print(f"❌ Could not load HuggingFace model: {e}")
            raise

        self.model.eval()
        print(f"✓ HuggingFace model loaded on {self.device}")

        # Performance tracking
        self.inference_times = []

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference and return depth map"""
        import torch

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        # Inference with timing
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(pixel_values)
            depth = outputs.predicted_depth

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        # Post-process
        depth_np = depth.squeeze().numpy()
        return depth_np

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colored visualization"""
        # Handle edge cases
        if np.isnan(depth_map).any() or np.isinf(depth_map).any():
            depth_map = np.nan_to_num(
                depth_map,
                nan=0,
                posinf=depth_map[~np.isinf(depth_map)].max(),
                neginf=depth_map[~np.isinf(depth_map)].min(),
            )

        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max == depth_min:
            # Create a gradient for visualization when no variation
            depth_uint8 = np.full_like(depth_map, 128, dtype=np.uint8)
        else:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
        return depth_colored

    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "std_inference_time": np.std(times),
            "avg_fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
        }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor


def calculate_metrics(depth1: np.ndarray, depth2: np.ndarray) -> dict:
    """Calculate comparison metrics between two depth maps"""
    # Ensure same size
    if depth1.shape != depth2.shape:
        depth2 = cv2.resize(depth2, (depth1.shape[1], depth1.shape[0]))

    # Check for invalid depth maps
    if depth1.max() == depth1.min() or depth2.max() == depth2.min():
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "psnr": float("inf"),
            "ssim": float("nan"),
            "correlation": float("nan"),
            "error": "One or both depth maps have no variation",
        }

    # Normalize both to same range for comparison
    depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min())
    depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min())

    # Mean Absolute Error
    mae = np.mean(np.abs(depth1_norm - depth2_norm))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((depth1_norm - depth2_norm) ** 2))

    # Peak Signal-to-Noise Ratio
    mse = np.mean((depth1_norm - depth2_norm) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")

    # Correlation coefficient
    corr = np.corrcoef(depth1_norm.flatten(), depth2_norm.flatten())[0, 1]

    # Simplified SSIM
    mean1, mean2 = np.mean(depth1_norm), np.mean(depth2_norm)
    var1, var2 = np.var(depth1_norm), np.var(depth2_norm)
    cov = np.mean((depth1_norm - mean1) * (depth2_norm - mean2))

    c1, c2 = 0.01, 0.03
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / (
        (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    )

    return {"mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim, "correlation": corr}


def check_memory_status() -> dict:
    """Check system and GPU memory status"""
    print("🔍 Memory Status:")
    print("=" * 40)

    mem_info = {}

    # System memory
    mem = psutil.virtual_memory()
    mem_info["system"] = {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent_used": mem.percent,
    }

    print("System RAM:")
    print(f"  Total: {mem_info['system']['total_gb']:.1f} GB")
    print(f"  Available: {mem_info['system']['available_gb']:.1f} GB")
    print(
        f"  Used: {mem_info['system']['used_gb']:.1f} GB \
            ({mem_info['system']['percent_used']:.1f}%)"
    )

    # GPU memory - try multiple methods
    mem_info["gpu"] = {"error": "GPU memory unavailable"}

    # Try nvidia-smi first
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(", ")
            mem_info["gpu"] = {
                "total_mb": int(gpu_info[0]),
                "used_mb": int(gpu_info[1]),
                "free_mb": int(gpu_info[2]),
                "percent_used": int(gpu_info[1]) / int(gpu_info[0]) * 100,
            }

            print("\nGPU Memory:")
            print(
                f"  Total: {mem_info['gpu']['total_mb']} MB \
                    ({mem_info['gpu']['total_mb']/1024:.1f} GB)"
            )
            print(
                f"  Used: {mem_info['gpu']['used_mb']} MB \
                    ({mem_info['gpu']['used_mb']/1024:.1f} GB)"
            )
            print(
                f"  Free: {mem_info['gpu']['free_mb']} MB \
                    ({mem_info['gpu']['free_mb']/1024:.1f} GB)"
            )
            print(f"  Usage: {mem_info['gpu']['percent_used']:.1f}%")
        else:
            raise Exception("nvidia-smi returned non-zero exit code")
    except Exception:
        # Try alternative methods for GPU memory
        try:
            # Try using pynvml if available
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            mem_info["gpu"] = {
                "total_mb": info.total // 1024 // 1024,
                "used_mb": info.used // 1024 // 1024,
                "free_mb": info.free // 1024 // 1024,
                "percent_used": (info.used / info.total) * 100,
            }

            print("\nGPU Memory (via pynvml):")
            print(f"  Total: {mem_info['gpu']['total_mb']} MB")
            print(f"  Used: {mem_info['gpu']['used_mb']} MB")
            print(f"  Free: {mem_info['gpu']['free_mb']} MB")
            print(f"  Usage: {mem_info['gpu']['percent_used']:.1f}%")
        except Exception:
            print("\nGPU Memory: Not available (nvidia-smi/pynvml unavailable)")
            print("  Note: Install nvidia-utils or pynvml for GPU memory monitoring")

    # Current process memory usage
    current_process = psutil.Process()
    process_mem = current_process.memory_info()
    print("\nCurrent Process:")
    print(f"  RSS: {process_mem.rss / (1024**2):.1f} MB")
    print(f"  VMS: {process_mem.vms / (1024**2):.1f} MB")

    return mem_info


def create_three_way_comparison(
    image: np.ndarray,
    depth_hf: np.ndarray = None,
    depth_onnx: np.ndarray = None,
    depth_trt: np.ndarray = None,
    metrics_hf_onnx: dict = None,
    metrics_hf_trt: dict = None,
    metrics_onnx_trt: dict = None,
    perf_stats: dict = None,
    output_path: str = None,
) -> np.ndarray:
    """Create comprehensive three-way comparison visualization

    Generates a detailed comparison grid showing original image, all model outputs,
    difference maps, and performance/quality metrics.

    Args:
        image: Original input image
        depth_hf: HuggingFace depth map (optional)
        depth_onnx: ONNX depth map (optional)
        depth_trt: TensorRT depth map (optional)
        metrics_*: Comparison metrics between models
        perf_stats: Performance statistics for each model
        output_path: Path to save visualization

    Returns:
        Combined visualization image as numpy array
    """

    def normalize_depth(depth):
        """Normalize depth map for visualization"""
        if depth is None:
            return None

        # Handle edge case where depth has no variation
        if np.isnan(depth).any() or np.isinf(depth).any():
            depth = np.nan_to_num(
                depth,
                nan=0,
                posinf=depth[~np.isinf(depth)].max(),
                neginf=depth[~np.isinf(depth)].min(),
            )

        depth_min, depth_max = depth.min(), depth.max()
        if depth_max == depth_min:
            # Create a gradient for visualization
            normalized = np.zeros_like(depth)
            normalized[:] = 128  # Mid-gray
            return normalized.astype(np.uint8)

        # Normal normalization
        normalized = (depth - depth_min) / (depth_max - depth_min)
        result = (normalized * 255).astype(np.uint8)
        return result

    def create_placeholder(height: int, width: int, text: str) -> np.ndarray:
        """Create placeholder image with text for missing models"""
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            text,
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (128, 128, 128),
            2,
        )
        return placeholder

    # Standardize dimensions for consistent layout
    target_height = 300
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)

    # Resize original image
    image_resized = cv2.resize(image, (target_width, target_height))

    # Process depth maps with consistent visualization
    depth_maps = []
    labels = ["Original Image", "HuggingFace", "ONNX", "TensorRT"]

    # Original image
    depth_maps.append(image_resized)

    # HuggingFace model output
    if depth_hf is not None:
        hf_vis = cv2.applyColorMap(normalize_depth(depth_hf), cv2.COLORMAP_PLASMA)
        depth_maps.append(cv2.resize(hf_vis, (target_width, target_height)))
    else:
        depth_maps.append(create_placeholder(target_height, target_width, "HF N/A"))

    # ONNX model output
    if depth_onnx is not None:
        onnx_vis = cv2.applyColorMap(normalize_depth(depth_onnx), cv2.COLORMAP_PLASMA)
        depth_maps.append(cv2.resize(onnx_vis, (target_width, target_height)))
    else:
        depth_maps.append(create_placeholder(target_height, target_width, "ONNX N/A"))

    # TensorRT model output
    if depth_trt is not None:
        trt_vis = cv2.applyColorMap(normalize_depth(depth_trt), cv2.COLORMAP_PLASMA)
        depth_maps.append(cv2.resize(trt_vis, (target_width, target_height)))
    else:
        depth_maps.append(create_placeholder(target_height, target_width, "TRT N/A"))

    # Create main 2x2 grid layout
    top_row = np.hstack([depth_maps[0], depth_maps[1]])  # Original, HF
    bottom_row = np.hstack([depth_maps[2], depth_maps[3]])  # ONNX, TRT
    main_grid = np.vstack([top_row, bottom_row])

    # Create difference maps for model comparisons
    diff_maps = []
    diff_labels = []

    if depth_hf is not None and depth_onnx is not None:
        # HF vs ONNX difference visualization
        hf_norm = normalize_depth(depth_hf) / 255.0
        onnx_norm = normalize_depth(depth_onnx) / 255.0
        if hf_norm.shape != onnx_norm.shape:
            onnx_norm = cv2.resize(onnx_norm, (hf_norm.shape[1], hf_norm.shape[0]))
        diff_hf_onnx = np.abs(hf_norm - onnx_norm)
        diff_vis = cv2.applyColorMap((diff_hf_onnx * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        diff_maps.append(cv2.resize(diff_vis, (target_width, target_height)))
        diff_labels.append("HF vs ONNX")

    if depth_hf is not None and depth_trt is not None:
        # HF vs TRT difference visualization
        hf_norm = normalize_depth(depth_hf) / 255.0
        trt_norm = normalize_depth(depth_trt) / 255.0
        if hf_norm.shape != trt_norm.shape:
            trt_norm = cv2.resize(trt_norm, (hf_norm.shape[1], hf_norm.shape[0]))
        diff_hf_trt = np.abs(hf_norm - trt_norm)
        diff_vis = cv2.applyColorMap((diff_hf_trt * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        diff_maps.append(cv2.resize(diff_vis, (target_width, target_height)))
        diff_labels.append("HF vs TRT")

    if depth_onnx is not None and depth_trt is not None:
        # ONNX vs TRT difference visualization
        onnx_norm = normalize_depth(depth_onnx) / 255.0
        trt_norm = normalize_depth(depth_trt) / 255.0
        if onnx_norm.shape != trt_norm.shape:
            trt_norm = cv2.resize(trt_norm, (onnx_norm.shape[1], onnx_norm.shape[0]))
        diff_onnx_trt = np.abs(onnx_norm - trt_norm)
        diff_vis = cv2.applyColorMap((diff_onnx_trt * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        diff_maps.append(cv2.resize(diff_vis, (target_width, target_height)))
        diff_labels.append("ONNX vs TRT")

    # Combine main grid with difference maps
    if diff_maps:
        # Pad difference maps to fill row
        while len(diff_maps) < 3:
            diff_maps.append(create_placeholder(target_height, target_width, "N/A"))
            diff_labels.append("N/A")

        # Create difference comparison row
        diff_row = np.hstack(diff_maps[:3])

        # Calculate final layout dimensions
        total_height = main_grid.shape[0] + diff_row.shape[0] + 200  # Extra space for metrics
        total_width = max(main_grid.shape[1], diff_row.shape[1])

        final_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Place main grid and difference row
        final_image[: main_grid.shape[0], : main_grid.shape[1]] = main_grid
        diff_start_y = main_grid.shape[0]
        final_image[diff_start_y : diff_start_y + diff_row.shape[0], : diff_row.shape[1]] = diff_row

    else:
        # Just main grid with metrics space
        final_image = np.zeros((main_grid.shape[0] + 200, main_grid.shape[1], 3), dtype=np.uint8)
        final_image[: main_grid.shape[0], : main_grid.shape[1]] = main_grid

    # Add comprehensive text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    # Model labels for main grid
    cv2.putText(final_image, labels[0], (10, 30), font, font_scale, color, thickness)
    cv2.putText(
        final_image,
        labels[1],
        (target_width + 10, 30),
        font,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(
        final_image,
        labels[2],
        (10, target_height + 30),
        font,
        font_scale,
        color,
        thickness,
    )
    cv2.putText(
        final_image,
        labels[3],
        (target_width + 10, target_height + 30),
        font,
        font_scale,
        color,
        thickness,
    )

    # Difference map labels
    if diff_maps:
        diff_y = main_grid.shape[0] + 30
        for i, label in enumerate(diff_labels[:3]):
            x_pos = i * target_width + 10
            cv2.putText(
                final_image,
                f"Diff: {label}",
                (x_pos, diff_y),
                font,
                0.6,
                (0, 255, 255),
                2,
            )

    # Performance metrics section
    text_start_y = main_grid.shape[0] + (diff_row.shape[0] if diff_maps else 0) + 50

    if perf_stats:
        cv2.putText(
            final_image,
            "Performance Comparison:",
            (10, text_start_y),
            font,
            0.8,
            (0, 255, 255),
            2,
        )

        y_offset = text_start_y + 30
        for model_name, stats in perf_stats.items():
            if stats:  # Check if stats are available
                fps = stats.get("avg_fps", 0)
                time_ms = stats.get("avg_inference_time", 0) * 1000
                text = f"{model_name}: {time_ms:.1f}ms ({fps:.1f} FPS)"
                cv2.putText(final_image, text, (10, y_offset), font, 0.6, color, 1)
                y_offset += 25

    # Quality metrics section
    metrics_y_start = text_start_y + 120
    if metrics_hf_onnx or metrics_hf_trt or metrics_onnx_trt:
        cv2.putText(
            final_image,
            "Conversion Quality Metrics:",
            (10, metrics_y_start),
            font,
            0.8,
            (0, 255, 255),
            2,
        )

        y_offset = metrics_y_start + 30

        if metrics_hf_onnx:
            corr = metrics_hf_onnx.get("correlation", 0)
            text = f"HF vs ONNX: Corr={corr:.3f}, MAE={metrics_hf_onnx.get('mae', 0):.3f}"
            cv2.putText(final_image, text, (10, y_offset), font, 0.5, color, 1)
            y_offset += 20

        if metrics_hf_trt:
            corr = metrics_hf_trt.get("correlation", 0)
            text = f"HF vs TRT: Corr={corr:.3f}, MAE={metrics_hf_trt.get('mae', 0):.3f}"
            cv2.putText(final_image, text, (10, y_offset), font, 0.5, color, 1)
            y_offset += 20

        if metrics_onnx_trt:
            corr = metrics_onnx_trt.get("correlation", 0)
            text = f"ONNX vs TRT: Corr={corr:.3f}, MAE={metrics_onnx_trt.get('mae', 0):.3f}"
            cv2.putText(final_image, text, (10, y_offset), font, 0.5, color, 1)

    # Save comprehensive visualization
    if output_path:
        cv2.imwrite(output_path, final_image)
        print(f"✓ Three-way comparison saved to: {output_path}")

    return final_image


def test_depth_estimation(
    models_dir: str,
    test_image_path: str = None,
    output_path: str = None,
    num_iterations: int = 10,
    use_gpu: bool = True,
    test_tensorrt: bool = True,
    compare_models: bool = False,
):
    """Test depth estimation models"""

    print("🔬 Depth Anything V2 Model Testing")
    print("=" * 60)

    models_dir = Path(models_dir)

    # Default test image
    if test_image_path is None:
        test_image_path = Path(__file__).parent.parent / "assets" / "images" / "bus.jpg"

    # Load test image
    if Path(test_image_path).exists():
        image = cv2.imread(str(test_image_path))
        print(f"✓ Loaded test image: {test_image_path} {image.shape}")
    else:
        image = create_test_image()
        print(f"✓ Created synthetic test image: {image.shape}")

    # Check memory
    try:
        check_memory_status()
        print()
    except Exception as e:
        print(f"⚠ Could not check memory: {e}")

    models = {}
    depth_maps = {}

    try:
        # Load ONNX model
        onnx_path = models_dir / "depth_anything_v2_small.onnx"
        config_path = models_dir / "config.json"

        if onnx_path.exists():
            try:
                models["onnx"] = ONNXDepthModel(str(onnx_path), str(config_path), use_gpu=use_gpu)
                print("✓ ONNX model loaded")
            except Exception as e:
                print(f"⚠ Could not load ONNX: {e}")

        # Load TensorRT model
        if test_tensorrt:
            trt_path = models_dir / "depth_anything_v2_small.trt"
            if trt_path.exists():
                try:
                    models["tensorrt"] = TensorRTDepthModel(str(trt_path), str(config_path))
                    print("✓ TensorRT model loaded")
                except Exception as e:
                    print(f"⚠ Could not load TensorRT: {e}")

        # Initialize HuggingFace model if comparison requested
        if compare_models:
            try:
                # Use CPU for HuggingFace to avoid GPU memory conflicts
                models["huggingface"] = HuggingFaceDepthModel(
                    str(models_dir),
                    use_cpu_only=True,  # Always use CPU to avoid conflicts
                )
                print("✓ HuggingFace model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load HuggingFace model for comparison: {e}")
                print("  Continuing without HuggingFace comparison...")

        if not models:
            print("❌ No models could be loaded")
            return False

        print(f"\n📊 Testing {len(models)} model(s): {list(models.keys())}")

        # Warmup
        print("\nWarmup...")
        models_to_remove = []
        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ")
            try:
                warmup_iterations = 3 if model_name != "huggingface" else 1
                for _ in range(warmup_iterations):
                    _ = model.infer(image)
                print("✓")
            except Exception as e:
                print(f"❌ {e}")
                models_to_remove.append(model_name)

        # Remove failed models after iteration
        for model_name in models_to_remove:
            models.pop(model_name, None)

        # Performance testing
        print(f"\nPerformance testing ({num_iterations} iterations)...")

        for model_name, model in models.items():
            print(f"  {model_name}...")
            test_iters = num_iterations if model_name != "huggingface" else min(3, num_iterations)
            try:
                for i in range(test_iters):
                    depth_map = model.infer(image)

                    if i == 0:
                        depth_maps[model_name] = depth_map
                        print(f"    Shape: {depth_map.shape}")
                        print(f"    Range: {depth_map.min():.3f} - {depth_map.max():.3f}")

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                if model_name in depth_maps:
                    del depth_maps[model_name]

        # Calculate metrics
        print("\nCalculating comparison metrics...")
        metrics = {}

        if "huggingface" in depth_maps and "onnx" in depth_maps:
            metrics["hf_vs_onnx"] = calculate_metrics(depth_maps["huggingface"], depth_maps["onnx"])

        if "huggingface" in depth_maps and "tensorrt" in depth_maps:
            metrics["hf_vs_trt"] = calculate_metrics(
                depth_maps["huggingface"], depth_maps["tensorrt"]
            )

        if "onnx" in depth_maps and "tensorrt" in depth_maps:
            metrics["onnx_vs_trt"] = calculate_metrics(depth_maps["onnx"], depth_maps["tensorrt"])

        # Performance stats
        perf_stats = {}
        for model_name, model in models.items():
            perf_stats[model_name] = model.get_performance_stats()

        # Display results
        print("\n=== Performance Results ===")
        for model_name, stats in perf_stats.items():
            if stats:
                print(f"{model_name.upper()}:")
                print(f"  Avg time: {stats['avg_inference_time']*1000:.2f} ms")
                print(f"  Avg FPS: {stats['avg_fps']:.1f}")

        # Comparisons
        if len(perf_stats) > 1:
            print("\n=== Performance Comparisons ===")
            base_models = ["huggingface", "onnx"]
            compare_models_list = ["onnx", "tensorrt"]

            for base in base_models:
                if base in perf_stats and perf_stats[base]:
                    base_time = perf_stats[base]["avg_inference_time"]
                    for comp in compare_models_list:
                        if comp in perf_stats and comp != base and perf_stats[comp]:
                            comp_time = perf_stats[comp]["avg_inference_time"]
                            speedup = base_time / comp_time
                            print(f"  {comp.upper()} vs {base.upper()}: {speedup:.2f}x speedup")

        # Quality metrics
        if metrics:
            print("\n=== Model Quality Comparisons ===")
            for comparison, metric_data in metrics.items():
                if "error" in metric_data:
                    print(f"{comparison}: {metric_data['error']}")
                else:
                    print(f"{comparison.replace('_', ' ').title()}:")
                    print(f"  Correlation: {metric_data['correlation']:.6f}")
                    print(f"  MAE: {metric_data['mae']:.6f}")
                    print(f"  RMSE: {metric_data['rmse']:.6f}")
                    print(f"  PSNR: {metric_data['psnr']:.2f} dB")
                    print(f"  SSIM: {metric_data['ssim']:.6f}")

                    # Quality assessment
                    if metric_data["correlation"] > 0.99:
                        print("  ✓ Excellent quality - near identical")
                    elif metric_data["correlation"] > 0.95:
                        print("  ✓ Excellent quality - minimal differences")
                    elif metric_data["correlation"] > 0.90:
                        print("  ✓ Good quality - acceptable differences")
                    else:
                        print("  ⚠ Notable differences detected")

        # Save visualizations
        if output_path and depth_maps:
            print("\nSaving visualizations...")

            # Create comprehensive three-way comparison only
            comparison_output = f"{output_path}_full_comparison.jpg"
            create_three_way_comparison(
                image,
                depth_hf=depth_maps.get("huggingface"),
                depth_onnx=depth_maps.get("onnx"),
                depth_trt=depth_maps.get("tensorrt"),
                metrics_hf_onnx=metrics.get("hf_vs_onnx"),
                metrics_hf_trt=metrics.get("hf_vs_trt"),
                metrics_onnx_trt=metrics.get("onnx_vs_trt"),
                perf_stats=perf_stats,
                output_path=comparison_output,
            )

            # Save comprehensive results JSON
            results = {
                "test_image": str(test_image_path),
                "models_directory": str(models_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models_tested": list(models.keys()),
                "performance": {},
                "quality_metrics": {},
            }

            # Add performance data
            for model_name, stats in perf_stats.items():
                if stats:
                    results["performance"][model_name] = {
                        "avg_inference_time_ms": float(stats["avg_inference_time"] * 1000),
                        "fps": float(stats["avg_fps"]),
                        "iterations": int(stats["total_inferences"]),
                        "min_time_ms": float(stats["min_inference_time"] * 1000),
                        "max_time_ms": float(stats["max_inference_time"] * 1000),
                        "std_time_ms": float(stats["std_inference_time"] * 1000),
                    }

            # Add comprehensive quality metrics
            for comparison, metric_data in metrics.items():
                if "error" not in metric_data:
                    results["quality_metrics"][comparison] = {
                        "correlation": float(metric_data["correlation"]),
                        "mae": float(metric_data["mae"]),
                        "rmse": float(metric_data["rmse"]),
                        "psnr": float(metric_data["psnr"]),
                        "ssim": float(metric_data["ssim"]),
                    }
                else:
                    results["quality_metrics"][comparison] = {"error": metric_data["error"]}

            json_output = f"{output_path}_comprehensive_results.json"
            with open(json_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Comprehensive results saved: {json_output}")

        # Cleanup
        for model in models.values():
            try:
                model.cleanup()
            except Exception:
                pass

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Depth Anything V2 models (ONNX and TensorRT) with \
            comprehensive visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        default="models/depth_trt",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--image",
        help="Path to test image (default: assets/images/bus.jpg)",
    )
    parser.add_argument(
        "--output",
        default="depth_test",
        help="Output path prefix for visualizations and results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of performance test iterations",
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Skip TensorRT testing",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for ONNX/HuggingFace (default: GPU)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Include HuggingFace model comparison (default: enabled)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip HuggingFace model comparison (faster, ONNX+TensorRT only)",
    )

    args = parser.parse_args()

    # Handle compare logic
    compare_models = args.compare and not args.no_compare

    success = test_depth_estimation(
        models_dir=args.models_dir,
        test_image_path=args.image,
        output_path=args.output,
        num_iterations=args.iterations,
        use_gpu=not args.cpu,
        test_tensorrt=not args.no_tensorrt,
        compare_models=compare_models,
    )

    if success:
        print("\n🎉 Testing completed successfully!")
        print("\n📁 Generated files:")
        print(f"   - {args.output}_full_comparison.jpg: Comprehensive model comparison")
        print(
            f"   - {args.output}_comprehensive_results.json: Detailed metrics and performance data"
        )
        if compare_models:
            print("\n📊 Comparison includes HuggingFace model with full quality metrics")
        else:
            print("\n📊 ONNX vs TensorRT comparison only (use --no-compare to disable HuggingFace)")
        sys.exit(0)
    else:
        print("\n❌ Testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
