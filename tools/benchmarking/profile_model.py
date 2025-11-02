#!/usr/bin/env python3
"""
Model Profiling Utility
Comprehensive benchmarking tool for AI models on NVIDIA Jetson Orin Nano

According to architecture.md:
- Profiles inference times for all models (YOLO, FastDepth, Whisper, LLM)
- Measures memory usage, GPU utilization, and thermal performance
- Validates performance targets from architecture requirements
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import nvidia_ml_py3 as nvml
import psutil
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics"""

    model_name: str
    model_type: str
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    std_inference_time_ms: float
    fps: float
    throughput_samples_per_sec: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_used_mb: float
    system_memory_used_mb: float
    gpu_temperature_c: float
    cpu_temperature_c: float
    power_consumption_w: float
    iterations: int
    input_shape: tuple
    output_shape: tuple
    model_size_mb: float
    meets_target: bool
    target_fps: float
    notes: str = ""


class SystemMonitor:
    """System resource monitoring for performance profiling"""

    def __init__(self, poll_interval: float = 0.1):
        """
        Initialize system monitor

        Args:
            poll_interval: Polling interval in seconds
        """
        self.poll_interval = poll_interval
        self.monitoring = False
        self.metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_percent": [],
            "gpu_memory_used": [],
            "gpu_temperature": [],
            "cpu_temperature": [],
            "power_watts": [],
            "timestamps": [],
        }

        # Initialize NVIDIA ML
        try:
            nvml.nvmlInit()
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            logger.info("✓ NVIDIA GPU monitoring initialized")
        except Exception:
            self.gpu_available = False
            logger.warning("⚠ NVIDIA GPU monitoring not available")

    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.debug("System monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1.0)
        logger.debug("System monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            timestamp = time.time()

            # CPU and system memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # CPU temperature (try different paths)
            cpu_temp = self._get_cpu_temperature()

            self.metrics["timestamps"].append(timestamp)
            self.metrics["cpu_percent"].append(cpu_percent)
            self.metrics["memory_percent"].append(memory.percent)
            self.metrics["cpu_temperature"].append(cpu_temp)

            # GPU metrics
            if self.gpu_available:
                try:
                    # GPU utilization
                    gpu_util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.metrics["gpu_percent"].append(gpu_util.gpu)

                    # GPU memory
                    gpu_memory = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.metrics["gpu_memory_used"].append(gpu_memory.used / (1024**2))  # MB

                    # GPU temperature
                    gpu_temp = nvml.nvmlDeviceGetTemperature(
                        self.gpu_handle, nvml.NVML_TEMPERATURE_GPU
                    )
                    self.metrics["gpu_temperature"].append(gpu_temp)

                    # Power consumption
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Watts
                        self.metrics["power_watts"].append(power)
                    except Exception:
                        self.metrics["power_watts"].append(0.0)

                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
                    self.metrics["gpu_percent"].append(0.0)
                    self.metrics["gpu_memory_used"].append(0.0)
                    self.metrics["gpu_temperature"].append(0.0)
                    self.metrics["power_watts"].append(0.0)
            else:
                self.metrics["gpu_percent"].append(0.0)
                self.metrics["gpu_memory_used"].append(0.0)
                self.metrics["gpu_temperature"].append(0.0)
                self.metrics["power_watts"].append(0.0)

            time.sleep(self.poll_interval)

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature from various sources"""
        try:
            # Try thermal zones
            thermal_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp",
                "/sys/class/thermal/thermal_zone2/temp",
            ]

            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        temp = float(f.read().strip()) / 1000.0  # Convert mC to C
                        if 20 < temp < 100:  # Sanity check
                            return temp

            # Try sensors if available
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current

            return 0.0

        except Exception:
            return 0.0

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from monitoring period"""
        if not self.metrics["timestamps"]:
            return {}

        return {
            "avg_cpu_percent": np.mean(self.metrics["cpu_percent"]),
            "avg_memory_percent": np.mean(self.metrics["memory_percent"]),
            "avg_gpu_percent": np.mean(self.metrics["gpu_percent"]),
            "avg_gpu_memory_mb": np.mean(self.metrics["gpu_memory_used"]),
            "avg_gpu_temperature": np.mean(self.metrics["gpu_temperature"]),
            "avg_cpu_temperature": np.mean(self.metrics["cpu_temperature"]),
            "avg_power_watts": np.mean(self.metrics["power_watts"]),
            "max_gpu_temperature": np.max(self.metrics["gpu_temperature"]),
            "max_cpu_temperature": np.max(self.metrics["cpu_temperature"]),
            "max_power_watts": np.max(self.metrics["power_watts"]),
        }

    def reset_metrics(self):
        """Reset collected metrics"""
        for key in self.metrics:
            self.metrics[key].clear()


class ModelProfiler:
    """Profile AI model performance on Jetson Orin Nano"""

    def __init__(self):
        """Initialize model profiler"""
        self.monitor = SystemMonitor()
        self.results = []

        # Performance targets from architecture.md
        self.performance_targets = {
            "yolo": {
                "fps": 20.0,
                "resolution": (640, 480),
                "description": "YOLOv8n object detection",
            },
            "depth": {
                "fps": 15.0,
                "resolution": (320, 240),
                "description": "FastDepth monocular depth estimation",
            },
            "whisper": {
                "rtf": 0.3,  # Real-time factor
                "latency_s": 2.0,
                "description": "Whisper speech recognition",
            },
            "llm": {
                "latency_s": 3.0,
                "memory_gb": 2.5,
                "description": "NanoLLM inference",
            },
        }

    @contextmanager
    def profile_context(self, warmup_iterations: int = 5):
        """Context manager for profiling with system monitoring"""
        # Warmup
        if warmup_iterations > 0:
            logger.info(f"Performing {warmup_iterations} warmup iterations...")
            for _ in range(warmup_iterations):
                yield "warmup"

        # Reset and start monitoring
        self.monitor.reset_metrics()
        self.monitor.start_monitoring()

        try:
            yield "profile"
        finally:
            self.monitor.stop_monitoring()

    def profile_tensorrt_model(
        self,
        engine_path: str,
        input_shape: tuple,
        model_type: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> PerformanceMetrics:
        """
        Profile TensorRT model performance

        Args:
            engine_path: Path to TensorRT engine file
            input_shape: Input tensor shape (B, C, H, W)
            model_type: Type of model (yolo, depth, etc.)
            num_iterations: Number of profiling iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            PerformanceMetrics object with results
        """
        logger.info(f"Profiling TensorRT model: {engine_path}")

        try:
            import pycuda.autoinit  # noqa F401
            import pycuda.driver as cuda  # noqa F401
            import tensorrt as trt  # noqa F401
        except ImportError as e:
            logger.error(f"Required dependencies not available: {e}")
            raise

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_trt_buffers(engine)

        # Create dummy input data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        inputs[0].host = input_data.flatten()

        # Profile with monitoring
        times = []
        output_shapes = []

        with self.profile_context(warmup_iterations) as phase:
            if phase == "warmup":
                self._do_trt_inference(context, bindings, inputs, outputs, stream)
            elif phase == "profile":
                logger.info(f"Running {num_iterations} profiling iterations...")

                for i in range(num_iterations):
                    start_time = time.perf_counter()

                    _ = self._do_trt_inference(context, bindings, inputs, outputs, stream)

                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms

                    if i == 0:  # Get output shape from first iteration
                        for j, out in enumerate(outputs):
                            shape = engine.get_binding_shape(j + len(inputs))
                            output_shapes.append(shape)

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        # Get system metrics
        system_metrics = self.monitor.get_average_metrics()

        # Get model size
        model_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

        # Check if meets target
        target_fps = self.performance_targets.get(model_type, {}).get("fps", 0)
        meets_target = fps >= target_fps if target_fps > 0 else True

        # Create metrics object
        metrics = PerformanceMetrics(
            model_name=os.path.basename(engine_path),
            model_type=model_type,
            avg_inference_time_ms=avg_time,
            min_inference_time_ms=min_time,
            max_inference_time_ms=max_time,
            std_inference_time_ms=std_time,
            fps=fps,
            throughput_samples_per_sec=fps,
            cpu_usage_percent=system_metrics.get("avg_cpu_percent", 0),
            gpu_usage_percent=system_metrics.get("avg_gpu_percent", 0),
            gpu_memory_used_mb=system_metrics.get("avg_gpu_memory_mb", 0),
            system_memory_used_mb=system_metrics.get("avg_memory_percent", 0),
            gpu_temperature_c=system_metrics.get("avg_gpu_temperature", 0),
            cpu_temperature_c=system_metrics.get("avg_cpu_temperature", 0),
            power_consumption_w=system_metrics.get("avg_power_watts", 0),
            iterations=num_iterations,
            input_shape=input_shape,
            output_shape=tuple(output_shapes[0]) if output_shapes else (),
            model_size_mb=model_size_mb,
            meets_target=meets_target,
            target_fps=target_fps,
        )

        self.results.append(metrics)
        return metrics

    def _allocate_trt_buffers(self, engine):
        """Allocate TensorRT buffers"""

        try:
            import pycuda.autoinit  # noqa F401
            import pycuda.driver as cuda  # noqa F401
            import tensorrt as trt  # noqa F401
        except ImportError as e:
            logger.error(f"Required dependencies not available: {e}")
            raise

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def _do_trt_inference(self, context, bindings, inputs, outputs, stream):
        """Run TensorRT inference"""
        import pycuda.driver as cuda

        # Transfer input data
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer outputs
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

        stream.synchronize()
        return [out["host"] for out in outputs]

    def profile_custom_function(
        self,
        func: Callable,
        model_name: str,
        model_type: str,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs,
    ) -> PerformanceMetrics:
        """
        Profile custom function performance

        Args:
            func: Function to profile
            model_name: Name of the model
            model_type: Type of model
            num_iterations: Number of profiling iterations
            warmup_iterations: Number of warmup iterations
            **kwargs: Additional arguments for the function

        Returns:
            PerformanceMetrics object with results
        """
        logger.info(f"Profiling custom function: {model_name}")

        times = []

        with self.profile_context(warmup_iterations) as phase:
            if phase == "warmup":
                func(**kwargs)
            elif phase == "profile":
                logger.info(f"Running {num_iterations} profiling iterations...")

                for i in range(num_iterations):
                    start_time = time.perf_counter()

                    _ = func(**kwargs)

                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        # Get system metrics
        system_metrics = self.monitor.get_average_metrics()

        # Check if meets target
        target_fps = self.performance_targets.get(model_type, {}).get("fps", 0)
        meets_target = fps >= target_fps if target_fps > 0 else True

        # Create metrics object
        metrics = PerformanceMetrics(
            model_name=model_name,
            model_type=model_type,
            avg_inference_time_ms=avg_time,
            min_inference_time_ms=min_time,
            max_inference_time_ms=max_time,
            std_inference_time_ms=std_time,
            fps=fps,
            throughput_samples_per_sec=fps,
            cpu_usage_percent=system_metrics.get("avg_cpu_percent", 0),
            gpu_usage_percent=system_metrics.get("avg_gpu_percent", 0),
            gpu_memory_used_mb=system_metrics.get("avg_gpu_memory_mb", 0),
            system_memory_used_mb=system_metrics.get("avg_memory_percent", 0),
            gpu_temperature_c=system_metrics.get("avg_gpu_temperature", 0),
            cpu_temperature_c=system_metrics.get("avg_cpu_temperature", 0),
            power_consumption_w=system_metrics.get("avg_power_watts", 0),
            iterations=num_iterations,
            input_shape=(),
            output_shape=(),
            model_size_mb=0.0,
            meets_target=meets_target,
            target_fps=target_fps,
        )

        self.results.append(metrics)
        return metrics

    def display_results(self, save_path: Optional[str] = None):
        """Display profiling results in a formatted table"""
        if not self.results:
            logger.info("No profiling results to display")
            return

        # Create table data
        headers = [
            "Model",
            "Type",
            "Avg Time (ms)",
            "FPS",
            "CPU %",
            "GPU %",
            "GPU Mem (MB)",
            "GPU Temp (°C)",
            "Power (W)",
            "Target Met",
        ]

        rows = []
        for metrics in self.results:
            status = "✓" if metrics.meets_target else "✗"
            rows.append(
                [
                    metrics.model_name,
                    metrics.model_type,
                    f"{metrics.avg_inference_time_ms:.2f}",
                    f"{metrics.fps:.2f}",
                    f"{metrics.cpu_usage_percent:.1f}",
                    f"{metrics.gpu_usage_percent:.1f}",
                    f"{metrics.gpu_memory_used_mb:.1f}",
                    f"{metrics.gpu_temperature_c:.1f}",
                    f"{metrics.power_consumption_w:.1f}",
                    status,
                ]
            )

        print("\n" + "=" * 100)
        print("MODEL PROFILING RESULTS")
        print("=" * 100)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Performance summary
        print("\nPERFORMance SUMMARY:")
        total_models = len(self.results)
        passed_models = sum(1 for r in self.results if r.meets_target)
        print(f"Models tested: {total_models}")
        print(f"Performance targets met: {passed_models}/{total_models}")

        # Save detailed results
        if save_path:
            self.save_results(save_path)
            print(f"Detailed results saved to: {save_path}")

    def save_results(self, output_path: str):
        """Save detailed results to JSON file"""
        results_data = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "performance_targets": self.performance_targets,
            "results": [asdict(metrics) for metrics in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": sys.platform,
        }

        if self.monitor.gpu_available:
            try:
                gpu_name = nvml.nvmlDeviceGetName(self.monitor.gpu_handle).decode()
                gpu_memory = nvml.nvmlDeviceGetMemoryInfo(self.monitor.gpu_handle)
                info.update(
                    {
                        "gpu_name": gpu_name,
                        "gpu_memory_total_gb": gpu_memory.total / (1024**3),
                    }
                )
            except Exception:
                pass

        return info

    def create_performance_plots(self, output_dir: str):
        """Create performance visualization plots"""
        if not self.results:
            logger.info("No results to plot")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")

        # 1. Inference time comparison
        plt.figure(figsize=(12, 6))
        models = [r.model_name for r in self.results]
        times = [r.avg_inference_time_ms for r in self.results]
        colors = ["green" if r.meets_target else "red" for r in self.results]

        plt.bar(models, times, color=colors, alpha=0.7)
        plt.ylabel("Average Inference Time (ms)")
        plt.title("Model Inference Time Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "inference_times.png", dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Resource utilization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # CPU usage
        axes[0, 0].bar(models, [r.cpu_usage_percent for r in self.results])
        axes[0, 0].set_ylabel("CPU Usage (%)")
        axes[0, 0].set_title("CPU Utilization")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # GPU usage
        axes[0, 1].bar(models, [r.gpu_usage_percent for r in self.results])
        axes[0, 1].set_ylabel("GPU Usage (%)")
        axes[0, 1].set_title("GPU Utilization")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # GPU memory
        axes[1, 0].bar(models, [r.gpu_memory_used_mb for r in self.results])
        axes[1, 0].set_ylabel("GPU Memory (MB)")
        axes[1, 0].set_title("GPU Memory Usage")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Temperature
        axes[1, 1].bar(models, [r.gpu_temperature_c for r in self.results])
        axes[1, 1].set_ylabel("Temperature (°C)")
        axes[1, 1].set_title("GPU Temperature")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "resource_utilization.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance plots saved to: {output_dir}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Profile AI model performance on Jetson Orin Nano")
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directory containing TensorRT models to profile",
    )
    parser.add_argument(
        "--output-dir",
        default="./profiling_results",
        help="Directory to save profiling results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of profiling iterations per model",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument(
        "--create-plots",
        action="store_true",
        help="Create performance visualization plots",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize profiler
    profiler = ModelProfiler()

    try:
        # Find TensorRT models
        models_dir = Path(args.models_dir)
        trt_files = list(models_dir.rglob("*.trt")) + list(models_dir.rglob("*.engine"))

        if not trt_files:
            logger.warning(f"No TensorRT models found in {models_dir}")
            logger.info("Example models to profile:")
            logger.info("  - YOLO: python tools/conversion/convert_yolo.py")
            logger.info("  - FastDepth: python tools/conversion/convert_depth.py")
            return

        logger.info(f"Found {len(trt_files)} TensorRT models to profile")

        # Profile each model
        for model_path in trt_files:
            model_name = model_path.stem

            # Determine model type from name/path
            if "yolo" in model_name.lower():
                model_type = "yolo"
                input_shape = (1, 3, 640, 480)
            elif "depth" in model_name.lower() or "fastdepth" in model_name.lower():
                model_type = "depth"
                input_shape = (1, 3, 240, 320)
            else:
                model_type = "unknown"
                input_shape = (1, 3, 224, 224)  # Default

            logger.info(f"Profiling {model_name} ({model_type})...")

            try:
                metrics = profiler.profile_tensorrt_model(
                    engine_path=str(model_path),
                    input_shape=input_shape,
                    model_type=model_type,
                    num_iterations=args.iterations,
                    warmup_iterations=args.warmup,
                )
                logger.info(f"✓ {model_name}: {metrics.fps:.2f} FPS")

            except Exception as e:
                logger.error(f"Failed to profile {model_name}: {e}")

        # Display and save results
        profiler.display_results()

        results_file = output_dir / "profiling_results.json"
        profiler.save_results(str(results_file))

        # Create plots if requested
        if args.create_plots:
            profiler.create_performance_plots(str(output_dir))

        print("\n✓ Profiling completed successfully!")
        print(f"Results saved to: {output_dir}")

    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
