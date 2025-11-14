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

    # GPU memory
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
            }

            print("\nGPU Memory:")
            print(f"  Total: {mem_info['gpu']['total_mb']} MB")
            print(f"  Used: {mem_info['gpu']['used_mb']} MB")
            print(f"  Free: {mem_info['gpu']['free_mb']} MB")
    except Exception:
        print("\nGPU Memory: nvidia-smi not available")

    return mem_info


def test_depth_estimation(
    models_dir: str,
    test_image_path: str = None,
    output_path: str = None,
    num_iterations: int = 10,
    use_gpu: bool = True,
    test_tensorrt: bool = True,
):
    """Test depth estimation models"""

    print("🔬 Depth Anything V2 Model Testing")
    print("=" * 60)

    models_dir = Path(models_dir)

    # Default test image
    if test_image_path is None:
        test_image_path = Path(__file__).parent.parent / "docs" / "images" / "bus.jpg"

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

        if not models:
            print("❌ No models could be loaded")
            return False

        print(f"\n📊 Testing {len(models)} model(s): {list(models.keys())}")

        # Warmup
        print("\nWarmup...")
        for model_name, model in models.items():
            print(f"  {model_name}...", end=" ")
            try:
                for _ in range(3):
                    _ = model.infer(image)
                print("✓")
            except Exception as e:
                print(f"❌ {e}")
                models.pop(model_name)

        # Performance testing
        print(f"\nPerformance testing ({num_iterations} iterations)...")

        for model_name, model in models.items():
            print(f"  {model_name}...")
            try:
                for i in range(num_iterations):
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
        metrics = {}
        if "onnx" in depth_maps and "tensorrt" in depth_maps:
            print("\nCalculating metrics...")
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
        if len(perf_stats) > 1 and "onnx" in perf_stats and "tensorrt" in perf_stats:
            if perf_stats["onnx"] and perf_stats["tensorrt"]:
                speedup = (
                    perf_stats["onnx"]["avg_inference_time"]
                    / perf_stats["tensorrt"]["avg_inference_time"]
                )
                print(f"\nTensorRT vs ONNX: {speedup:.2f}x speedup")

        # Quality metrics
        if metrics:
            print("\n=== Quality Metrics ===")
            for comparison, metric_data in metrics.items():
                if "error" in metric_data:
                    print(f"{comparison}: {metric_data['error']}")
                else:
                    print(f"{comparison}:")
                    print(f"  Correlation: {metric_data['correlation']:.6f}")
                    print(f"  MAE: {metric_data['mae']:.6f}")

        # Save visualizations
        if output_path and depth_maps:
            print("\nSaving visualizations...")

            for model_name, depth_map in depth_maps.items():
                model_vis = models[model_name].visualize_depth(depth_map)

                # Resize for side-by-side
                output_height = 400
                image_resized = cv2.resize(
                    image,
                    (
                        int(image.shape[1] * output_height / image.shape[0]),
                        output_height,
                    ),
                )
                model_vis_resized = cv2.resize(
                    model_vis,
                    (
                        int(model_vis.shape[1] * output_height / model_vis.shape[0]),
                        output_height,
                    ),
                )

                side_by_side = np.hstack([image_resized, model_vis_resized])

                model_output = f"{output_path}_{model_name}.jpg"
                cv2.imwrite(model_output, side_by_side)
                print(f"  ✓ {model_output}")

            # Save results JSON
            results = {
                "test_image": str(test_image_path),
                "models_directory": str(models_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "performance": {},
                "quality_metrics": {},
            }

            for model_name, stats in perf_stats.items():
                if stats:
                    results["performance"][model_name] = {
                        "avg_time_ms": float(stats["avg_inference_time"] * 1000),
                        "fps": float(stats["avg_fps"]),
                    }

            for comparison, metric_data in metrics.items():
                if "error" not in metric_data:
                    results["quality_metrics"][comparison] = {
                        "correlation": float(metric_data["correlation"]),
                        "mae": float(metric_data["mae"]),
                    }

            json_output = f"{output_path}_results.json"
            with open(json_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ {json_output}")

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
        description="Test Depth Anything V2 models (ONNX and TensorRT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dir",
        default="models/depth_trt",
        help="Directory containing model files",
    )
    parser.add_argument(
        "--image",
        help="Path to test image (default: docs/images/bus.jpg)",
    )
    parser.add_argument(
        "--output",
        default="depth_test",
        help="Output path prefix",
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
        help="Use CPU for ONNX (default: GPU)",
    )

    args = parser.parse_args()

    success = test_depth_estimation(
        models_dir=args.models_dir,
        test_image_path=args.image,
        output_path=args.output,
        num_iterations=args.iterations,
        use_gpu=not args.cpu,
        test_tensorrt=not args.no_tensorrt,
    )

    if success:
        print("\n🎉 Testing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
