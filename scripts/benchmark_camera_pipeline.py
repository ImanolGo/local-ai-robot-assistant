#!/usr/bin/env python3
"""
Benchmark script for Camera Pipeline Performance
Measures processing latency, GPU memory usage, and throughput optimization.

Author: Local AI Robot Team
License: Apache-2.0
"""

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import yaml

# Try to import GPU monitoring
try:
    import pynvml  # noqa F401

    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

# Try to import OpenCV CUDA
try:
    import cv2.cuda as cv2_cuda

    GPU_OPENCV_AVAILABLE = True
except ImportError:
    GPU_OPENCV_AVAILABLE = False


class CameraPipelineBenchmark:
    """
    Comprehensive benchmark suite for camera pipeline performance.

    Features:
    - CPU vs GPU undistortion benchmarks
    - Memory usage monitoring
    - Latency analysis
    - Throughput measurements
    - Frame rate consistency tests
    """

    def __init__(self, config_path: str, calibration_path: str):
        self.config_path = config_path
        self.calibration_path = calibration_path

        # Load configuration and calibration
        self.config = self._load_config()
        self.calibration = self._load_calibration()

        # Benchmark results
        self.results: Dict = {}

        # Initialize GPU monitoring if available
        if GPU_MONITORING_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"Warning: GPU monitoring initialization failed: {e}")
                self.gpu_available = False
        else:
            self.gpu_available = False

        print(f"GPU Monitoring: {'Available' if self.gpu_available else 'Not Available'}")
        print(f"OpenCV CUDA: {'Available' if GPU_OPENCV_AVAILABLE else 'Not Available'}")

    def _load_config(self) -> Dict:
        """Load camera configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()

    def _load_calibration(self) -> Dict:
        """Load camera calibration."""
        try:
            with open(self.calibration_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return self._get_default_calibration()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "camera": {"width": 1640, "height": 1232},
            "undistortion": {
                "interpolation_method": "linear",
                "border_mode": "constant",
                "alpha": 1.0,
            },
        }

    def _get_default_calibration(self) -> Dict:
        """Get default calibration."""
        return {
            "camera_matrix": [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [[-0.3, 0.1, 0.0, 0.0, -0.01]],
            "image_width": 640,
            "image_height": 480,
        }

    def _get_gpu_memory_info(self) -> Optional[Tuple[int, int]]:
        """Get GPU memory usage information."""
        if not self.gpu_available:
            return None

        try:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return memory_info.used, memory_info.total
        except Exception:
            return None

    def _create_test_images(self, count: int = 100) -> List[np.ndarray]:
        """Create test images for benchmarking."""
        width = self.calibration["image_width"]
        height = self.calibration["image_height"]

        images = []
        for i in range(count):
            # Create varied test patterns
            if i % 4 == 0:
                # Checkerboard pattern
                image = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(0, height, 40):
                    for x in range(0, width, 40):
                        if ((y // 40) + (x // 40)) % 2 == 0:
                            image[y : y + 40, x : x + 40] = [255, 255, 255]
            elif i % 4 == 1:
                # Random noise
                image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            elif i % 4 == 2:
                # Gradient pattern
                image = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    intensity = int((y / height) * 255)
                    image[y, :] = [intensity, intensity, intensity]
            else:
                # Circular patterns
                image = np.zeros((height, width, 3), dtype=np.uint8)
                center_x, center_y = width // 2, height // 2
                for y in range(height):
                    for x in range(width):
                        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        intensity = int((np.sin(dist / 20) + 1) * 127)
                        image[y, x] = [intensity, intensity, intensity]

            images.append(image)

        return images

    def benchmark_cpu_undistortion(self, test_images: List[np.ndarray]) -> Dict:
        """Benchmark CPU undistortion performance."""
        print("\n=== CPU Undistortion Benchmark ===")

        # Setup undistortion parameters
        camera_matrix = np.array(self.calibration["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(
            self.calibration["distortion_coefficients"], dtype=np.float32
        ).flatten()

        width = self.calibration["image_width"]
        height = self.calibration["image_height"]

        # Get optimal camera matrix
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (width, height),
            self.config["undistortion"]["alpha"],
            (width, height),
        )

        # Test both cached maps and direct undistortion
        results = {}

        # 1. Test with cached maps
        print("Testing CPU undistortion with cached maps...")
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

        latencies_cached = []
        memory_usage = []

        for image in test_images:
            # Monitor memory
            process = psutil.Process()
            mem_before = process.memory_info().rss

            start_time = time.perf_counter()
            _ = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            end_time = time.perf_counter()

            mem_after = process.memory_info().rss

            latencies_cached.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)

        results["cpu_cached"] = {
            "avg_latency": statistics.mean(latencies_cached),
            "min_latency": min(latencies_cached),
            "max_latency": max(latencies_cached),
            "std_latency": (statistics.stdev(latencies_cached) if len(latencies_cached) > 1 else 0),
            "avg_memory_delta": statistics.mean(memory_usage),
            "fps": 1.0 / statistics.mean(latencies_cached),
        }

        # 2. Test direct undistortion
        print("Testing CPU undistortion without cached maps...")
        latencies_direct = []

        for image in test_images:
            start_time = time.perf_counter()
            _ = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            end_time = time.perf_counter()

            latencies_direct.append(end_time - start_time)

        results["cpu_direct"] = {
            "avg_latency": statistics.mean(latencies_direct),
            "min_latency": min(latencies_direct),
            "max_latency": max(latencies_direct),
            "std_latency": (statistics.stdev(latencies_direct) if len(latencies_direct) > 1 else 0),
            "fps": 1.0 / statistics.mean(latencies_direct),
        }

        # Print results
        for method, metrics in results.items():
            print(f"\n{method.upper()} Results:")
            print(f"  Average latency: {metrics['avg_latency']*1000:.2f}ms")
            print(f"  Min latency: {metrics['min_latency']*1000:.2f}ms")
            print(f"  Max latency: {metrics['max_latency']*1000:.2f}ms")
            print(f"  Std deviation: {metrics['std_latency']*1000:.2f}ms")
            print(f"  Estimated FPS: {metrics['fps']:.1f}")
            if "avg_memory_delta" in metrics:
                print(f"  Avg memory delta: {metrics['avg_memory_delta']/1024:.1f}KB")

        return results

    def benchmark_gpu_undistortion(self, test_images: List[np.ndarray]) -> Optional[Dict]:
        """Benchmark GPU undistortion performance."""
        if not GPU_OPENCV_AVAILABLE:
            print("\n=== GPU Undistortion Benchmark ===")
            print("OpenCV CUDA not available, skipping GPU benchmark")
            return None

        print("\n=== GPU Undistortion Benchmark ===")

        try:
            # Setup undistortion parameters
            camera_matrix = np.array(self.calibration["camera_matrix"], dtype=np.float32)
            dist_coeffs = np.array(
                self.calibration["distortion_coefficients"], dtype=np.float32
            ).flatten()

            width = self.calibration["image_width"]
            height = self.calibration["image_height"]

            # Get optimal camera matrix
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                (width, height),
                self.config["undistortion"]["alpha"],
                (width, height),
            )

            # Create GPU maps
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                None,
                new_camera_matrix,
                (width, height),
                cv2.CV_32FC1,
            )

            gpu_map1 = cv2_cuda.GpuMat()
            gpu_map2 = cv2_cuda.GpuMat()
            gpu_map1.upload(map1)
            gpu_map2.upload(map2)

            # Pre-allocate GPU matrices
            gpu_src = cv2_cuda.GpuMat(height, width, cv2.CV_8UC3)
            gpu_dst = cv2_cuda.GpuMat(height, width, cv2.CV_8UC3)

            print("Testing GPU undistortion...")
            latencies = []
            gpu_memory_usage = []

            for image in test_images:
                # Monitor GPU memory
                gpu_mem_before = self._get_gpu_memory_info()

                start_time = time.perf_counter()

                # Upload to GPU
                gpu_src.upload(image)

                # Apply remap on GPU
                cv2_cuda.remap(gpu_src, gpu_dst, gpu_map1, gpu_map2, cv2.INTER_LINEAR)

                # Download result
                _ = gpu_dst.download()

                end_time = time.perf_counter()

                gpu_mem_after = self._get_gpu_memory_info()

                latencies.append(end_time - start_time)

                if gpu_mem_before and gpu_mem_after:
                    gpu_memory_usage.append(gpu_mem_after[0] - gpu_mem_before[0])

            results = {
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "std_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "fps": 1.0 / statistics.mean(latencies),
            }

            if gpu_memory_usage:
                results["avg_gpu_memory_delta"] = statistics.mean(gpu_memory_usage)

            # Print results
            print("\nGPU Results:")
            print(f"  Average latency: {results['avg_latency']*1000:.2f}ms")
            print(f"  Min latency: {results['min_latency']*1000:.2f}ms")
            print(f"  Max latency: {results['max_latency']*1000:.2f}ms")
            print(f"  Std deviation: {results['std_latency']*1000:.2f}ms")
            print(f"  Estimated FPS: {results['fps']:.1f}")

            if "avg_gpu_memory_delta" in results:
                print(
                    f"  Avg GPU memory delta: {results['avg_gpu_memory_delta']/(1024*1024):.1f}MB"
                )

            return results

        except Exception as e:
            print(f"GPU benchmark failed: {e}")
            return None

    def benchmark_memory_usage(self, test_images: List[np.ndarray]) -> Dict:
        """Benchmark memory usage patterns."""
        print("\n=== Memory Usage Benchmark ===")

        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss
        print(f"Baseline memory usage: {baseline_memory/(1024*1024):.1f}MB")

        # Test map caching impact
        camera_matrix = np.array(self.calibration["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(
            self.calibration["distortion_coefficients"], dtype=np.float32
        ).flatten()
        width = self.calibration["image_width"]
        height = self.calibration["image_height"]

        # Create maps
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

        maps_memory = process.memory_info().rss
        map_memory_usage = maps_memory - baseline_memory

        print(f"Memory after creating maps: {maps_memory/(1024*1024):.1f}MB")
        print(f"Map memory overhead: {map_memory_usage/(1024*1024):.1f}MB")

        # Test processing multiple images
        max_memory = baseline_memory
        memory_samples = []

        for i, image in enumerate(test_images[:50]):  # Test with first 50 images
            # Process image
            _ = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

            current_memory = process.memory_info().rss
            memory_samples.append(current_memory)
            max_memory = max(max_memory, current_memory)

            if i % 10 == 0:
                print(f"  After {i+1} images: {current_memory/(1024*1024):.1f}MB")

        peak_memory_usage = max_memory - baseline_memory

        results = {
            "baseline_memory_mb": baseline_memory / (1024 * 1024),
            "map_memory_overhead_mb": map_memory_usage / (1024 * 1024),
            "peak_memory_usage_mb": peak_memory_usage / (1024 * 1024),
            "avg_processing_memory_mb": statistics.mean(memory_samples) / (1024 * 1024),
        }

        print("\nMemory Summary:")
        print(f"  Peak memory usage: {results['peak_memory_usage_mb']:.1f}MB")
        print(f"  Average processing memory: {results['avg_processing_memory_mb']:.1f}MB")

        return results

    def benchmark_throughput(self, test_images: List[np.ndarray]) -> Dict:
        """Benchmark sustained throughput performance."""
        print("\n=== Throughput Benchmark ===")

        camera_matrix = np.array(self.calibration["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(
            self.calibration["distortion_coefficients"], dtype=np.float32
        ).flatten()
        width = self.calibration["image_width"]
        height = self.calibration["image_height"]

        # Create maps
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            None,
            camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

        # Test sustained processing for 10 seconds
        duration = 10.0  # seconds
        frame_count = 0
        frame_times = []

        print(f"Running sustained processing for {duration} seconds...")

        start_time = time.perf_counter()
        _ = start_time

        while time.perf_counter() - start_time < duration:
            # Process random image from test set
            image_idx = frame_count % len(test_images)
            image = test_images[image_idx]

            frame_start = time.perf_counter()
            _ = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            frame_end = time.perf_counter()

            frame_times.append(frame_end - frame_start)
            frame_count += 1

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                current_time = time.perf_counter()
                elapsed = current_time - start_time
                current_fps = frame_count / elapsed
                print(f"  {frame_count} frames processed, Current FPS: {current_fps:.1f}")

        total_time = time.perf_counter() - start_time
        average_fps = frame_count / total_time

        results = {
            "total_frames": frame_count,
            "total_time": total_time,
            "average_fps": average_fps,
            "avg_frame_time": statistics.mean(frame_times),
            "min_frame_time": min(frame_times),
            "max_frame_time": max(frame_times),
            "frame_time_std": (statistics.stdev(frame_times) if len(frame_times) > 1 else 0),
        }

        print("\nThroughput Results:")
        print(f"  Total frames processed: {results['total_frames']}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Average FPS: {results['average_fps']:.1f}")
        print(f"  Frame time - Avg: {results['avg_frame_time']*1000:.2f}ms")
        print(f"  Frame time - Min: {results['min_frame_time']*1000:.2f}ms")
        print(f"  Frame time - Max: {results['max_frame_time']*1000:.2f}ms")
        print(f"  Frame time - Std: {results['frame_time_std']*1000:.2f}ms")

        return results

    def run_full_benchmark(self) -> Dict:
        """Run the complete benchmark suite."""
        print("=== Camera Pipeline Performance Benchmark ===")
        print(f"Configuration: {self.config_path}")
        print(f"Calibration: {self.calibration_path}")
        print(f"Image size: {self.calibration['image_width']}x{self.calibration['image_height']}")

        # Create test images
        print("\nCreating test images...")
        test_images = self._create_test_images(100)
        print(f"Created {len(test_images)} test images")

        # Run benchmarks
        self.results["cpu_undistortion"] = self.benchmark_cpu_undistortion(test_images)

        gpu_results = self.benchmark_gpu_undistortion(test_images)
        if gpu_results:
            self.results["gpu_undistortion"] = gpu_results

        self.results["memory_usage"] = self.benchmark_memory_usage(test_images)
        self.results["throughput"] = self.benchmark_throughput(test_images)

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """Generate and print benchmark summary."""
        print("\n" + "=" * 60)
        print("                    BENCHMARK SUMMARY")
        print("=" * 60)

        # Performance comparison
        if "cpu_undistortion" in self.results:
            cpu_cached = self.results["cpu_undistortion"].get("cpu_cached", {})
            cpu_direct = self.results["cpu_undistortion"].get("cpu_direct", {})

            print("\nCPU Performance:")
            if cpu_cached:
                print(
                    f"  Cached maps: {cpu_cached['fps']:.1f}"
                    f" FPS ({cpu_cached['avg_latency']*1000:.1f}ms)"
                )
            if cpu_direct:
                print(
                    f"  Direct:      {cpu_direct['fps']:.1f}"
                    f" FPS ({cpu_direct['avg_latency']*1000:.1f}ms)"
                )

        if "gpu_undistortion" in self.results:
            gpu = self.results["gpu_undistortion"]
            print("\nGPU Performance:")
            print(f"  GPU remap:   {gpu['fps']:.1f} FPS ({gpu['avg_latency']*1000:.1f}ms)")

            # GPU vs CPU comparison
            if "cpu_undistortion" in self.results:
                cpu_cached = self.results["cpu_undistortion"].get("cpu_cached", {})
                if cpu_cached:
                    speedup = gpu["fps"] / cpu_cached["fps"]
                    print(f"  Speedup vs CPU cached: {speedup:.1f}x")

        # Memory usage
        if "memory_usage" in self.results:
            mem = self.results["memory_usage"]
            print("\nMemory Usage:")
            print(f"  Baseline: {mem['baseline_memory_mb']:.1f}MB")
            print(f"  Peak: {mem['peak_memory_usage_mb']:.1f}MB")
            print(f"  Map overhead: {mem['map_memory_overhead_mb']:.1f}MB")

        # Throughput
        if "throughput" in self.results:
            throughput = self.results["throughput"]
            print("\nSustained Throughput:")
            print(f"  Average FPS: {throughput['average_fps']:.1f}")
            print(f"  Frame time consistency: ±{throughput['frame_time_std']*1000:.1f}ms")

        # Recommendations
        print("\nRecommendations:")

        if "cpu_undistortion" in self.results:
            cpu_cached = self.results["cpu_undistortion"].get("cpu_cached", {})
            if cpu_cached and cpu_cached["fps"] >= 30:
                print("  ✓ CPU performance sufficient for 30 FPS")
            elif cpu_cached:
                print("  ⚠ CPU performance may struggle with real-time processing")

        if "gpu_undistortion" in self.results:
            gpu = self.results["gpu_undistortion"]
            if gpu["fps"] >= 60:
                print("  ✓ GPU performance excellent for high frame rates")
            elif gpu["fps"] >= 30:
                print("  ✓ GPU performance good for real-time processing")

        if "memory_usage" in self.results:
            mem = self.results["memory_usage"]
            if mem["peak_memory_usage_mb"] < 500:
                print("  ✓ Memory usage within reasonable limits")
            else:
                print("  ⚠ High memory usage - consider optimization")

        print("=" * 60)


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Camera Pipeline Performance Benchmark")
    parser.add_argument(
        "--config",
        default="/home/imanolgo/repos/local-ai-robot-assistant/config/camera_config.yaml",
        help="Path to camera configuration file",
    )
    parser.add_argument(
        "--calibration",
        default="/home/imanolgo/repos/local-ai-robot-assistant/config/camera_calibration.yaml",
        help="Path to camera calibration file",
    )
    parser.add_argument("--output", help="Output file to save benchmark results (JSON format)")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file not found: {args.calibration}")
        sys.exit(1)

    # Run benchmark
    try:
        benchmark = CameraPipelineBenchmark(args.config, args.calibration)
        results = benchmark.run_full_benchmark()

        # Save results if requested
        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
