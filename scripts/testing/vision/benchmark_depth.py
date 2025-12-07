#!/usr/bin/env python3
"""
Depth Anything V2 TensorRT Benchmark Script

Simple, focused benchmark for TensorRT depth estimation on Jetson Orin Nano.
Tests inference performance and generates side-by-side visualization.

Usage:
    python scripts/benchmark_depth.py
    python scripts/benchmark_depth.py --image path/to/image.jpg
    python scripts/benchmark_depth.py --iterations 200 --output my_result.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError:
    print("ERROR: TensorRT or PyCUDA not available")
    print("Install with: pip install pycuda")
    sys.exit(1)


class DepthEngine:
    """
    Clean TensorRT inference engine for Depth Anything V2
    Optimized for TensorRT 10.x on Jetson Orin Nano
    """

    def __init__(self, engine_path: str):
        """Initialize TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"Loading TensorRT Engine: {engine_path}")

        # Initialize CUDA
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Discover IO tensors (TensorRT 10.x style)
        self.input_name = None
        self.output_name = None

        num_io_tensors = self.engine.num_io_tensors
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if not self.input_name or not self.output_name:
            raise RuntimeError("Could not find input/output tensors")

        # Get shapes
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        print("✓ Engine loaded successfully")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name} {self.output_shape}")

        # Allocate pinned host memory and device memory
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # Normalization constants (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Performance tracking
        self.inference_times = []

    def infer(self, img_518: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed 518x518 RGB image

        Args:
            img_518: Input image (518, 518, 3) in RGB format, uint8

        Returns:
            Depth map (518, 518) as float32
        """
        # Ensure CUDA context is active
        self.cuda_ctx.push()

        try:
            # 1. Preprocess: Normalize and convert to CHW
            img = img_518.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            # Copy to pinned memory
            np.copyto(self.h_input, img.ravel())

            # 2. Upload to device
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)

            # 3. Execute inference (TensorRT 10.x style)
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))

            start_time = time.time()
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            inference_time = time.time() - start_time

            # Track performance
            self.inference_times.append(inference_time)

            # 4. Download results
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()

            # Reshape to output dimensions
            depth_map = self.h_output.reshape(self.output_shape[1:])

            return depth_map

        finally:
            self.cuda_ctx.pop()

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
        """Cleanup resources"""
        try:
            if hasattr(self, "d_input"):
                self.d_input.free()
            if hasattr(self, "d_output"):
                self.d_output.free()
            if hasattr(self, "context"):
                del self.context
            if hasattr(self, "engine"):
                del self.engine
            if hasattr(self, "cuda_ctx"):
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
        except Exception as e:
            print(f"Warning during cleanup: {e}")


def colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize depth map and apply colormap for visualization

    Args:
        depth_map: Raw depth map (H, W) as float32

    Returns:
        Colored depth visualization (H, W, 3) as uint8
    """
    # Normalize to 0-255
    depth_min = depth_map.min()
    depth_max = depth_map.max()

    if depth_max == depth_min:
        # Handle edge case
        depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # Apply colormap (INFERNO/PLASMA are great for depth)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    return depth_color


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Depth Anything V2 TensorRT inference on Jetson",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        default="assets/images/demo01.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--engine",
        default="models/depth_trt/depth_anything_v2_vits_518.engine",
        help="Path to TensorRT engine",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output",
        default="benchmark_depth_result.jpg",
        help="Output filename for visualization",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    if not Path(args.engine).exists():
        print(f"ERROR: TensorRT engine not found: {args.engine}")
        print("Run model conversion first:")
        print("  python tools/conversion/convert_depth.py")
        return 1

    print("=" * 60)
    print("DEPTH ANYTHING V2 - TENSORRT BENCHMARK")
    print("=" * 60)

    try:
        # Load engine
        depth_net = DepthEngine(args.engine)

        # Load image
        print(f"\nLoading image: {args.image}")
        original_img = cv2.imread(str(args.image))
        if original_img is None:
            print(f"ERROR: Failed to load image: {args.image}")
            return 1

        orig_h, orig_w = original_img.shape[:2]
        print(f"Original resolution: {orig_w}x{orig_h}")
        print("Model resolution: 518x518")

        # Warmup
        print(f"\nWarming up GPU ({args.warmup} iterations)...")
        dummy = cv2.resize(original_img, (518, 518))
        dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)

        for _ in range(args.warmup):
            _ = depth_net.infer(dummy_rgb)

        # Reset timing stats after warmup
        depth_net.inference_times = []

        # Benchmark
        print(f"\nBenchmarking ({args.iterations} iterations)...")
        print("Processing...", end="", flush=True)

        # Pre-resize once for all iterations
        input_resized = cv2.resize(original_img, (518, 518), interpolation=cv2.INTER_LINEAR)
        input_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)

        result_depth = None
        total_time_start = time.time()

        for i in range(args.iterations):
            # Run inference
            depth_raw = depth_net.infer(input_rgb)
            result_depth = depth_raw

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"\rProcessing... {i+1}/{args.iterations}", end="", flush=True)

        total_time = time.time() - total_time_start
        print("\rProcessing... Done!          ")

        # Get statistics
        stats = depth_net.get_performance_stats()

        # Display results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Average inference time: {stats['avg_inference_time']*1000:.2f} ms")
        print(f"Min inference time: {stats['min_inference_time']*1000:.2f} ms")
        print(f"Max inference time: {stats['max_inference_time']*1000:.2f} ms")
        print(f"Std deviation: {stats['std_inference_time']*1000:.2f} ms")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Total iterations: {stats['total_inferences']}")
        print(f"Total benchmark time: {total_time:.2f} s")
        print("=" * 60)

        # Post-process for visualization
        print("\nGenerating visualization...")

        # Resize depth to original resolution
        depth_upscaled = cv2.resize(result_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Colorize
        result_visual = colorize_depth(depth_upscaled)

        # Create side-by-side comparison
        combined = np.hstack((original_img, result_visual))

        # Save result
        cv2.imwrite(args.output, combined)
        print(f"✓ Saved result to: {args.output}")

        # Cleanup
        depth_net.cleanup()

        print("\n✅ Benchmark completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
