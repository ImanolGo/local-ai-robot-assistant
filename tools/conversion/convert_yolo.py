#!/usr/bin/env python3
"""
YOLO Model Conversion Script
Converts YOLOv11 models from PyTorch → ONNX → TensorRT FP16

According to architecture.md:
- Target: YOLOv11n optimized with TensorRT FP16
- Pipeline: DeepStream-based for hardware acceleration
- Performance Target: 20+ FPS at 640x480 resolution
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import nvidia_ml_py3 as nvml
import onnx
import psutil
import tensorrt as trt
from tabulate import tabulate
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class YOLOConverter:
    """
    Converts YOLO models to TensorRT optimized engines

    Supports the conversion pipeline: PyTorch → ONNX → TensorRT FP16
    Optimized for NVIDIA Jetson Orin Nano deployment
    """

    def __init__(
        self,
        model_name: str = "YOLOv11n",
        input_shape: Tuple[int, int, int, int] = (1, 3, 640, 480),
        workspace_size: int = 1 << 28,
    ):  # 256MB
        """
        Initialize YOLO converter

        Args:
            model_name: YOLO model variant (YOLOv11n, YOLOv11s, etc.)
            input_shape: Input tensor shape (batch, channels, height, width)
            workspace_size: TensorRT workspace size in bytes
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.workspace_size = workspace_size

        # Initialize NVIDIA ML for GPU monitoring
        try:
            nvml.nvmlInit()
            self.gpu_available = True
        except Exception:
            self.gpu_available = False
            logger.warning("NVIDIA ML not available - GPU monitoring disabled")

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

    def download_model(self, output_dir: str) -> str:
        """
        Download and load YOLO model

        Args:
            output_dir: Directory to save downloaded model

        Returns:
            Path to the downloaded model
        """
        logger.info(f"Loading YOLO model: {self.model_name}")

        # Load model using ultralytics
        model = YOLO(f"{self.model_name}.pt")

        # Save to output directory
        model_path = os.path.join(output_dir, f"{self.model_name}.pt")
        model.save(model_path)

        logger.info(f"Model saved to: {model_path}")
        return model_path

    def convert_to_onnx(
        self, pytorch_model_path: str, onnx_output_path: str, dynamic_batch: bool = True
    ) -> str:
        """
        Convert PyTorch YOLO model to ONNX format

        Args:
            pytorch_model_path: Path to PyTorch model
            onnx_output_path: Output path for ONNX model
            dynamic_batch: Whether to use dynamic batch size

        Returns:
            Path to ONNX model
        """
        logger.info("Converting PyTorch model to ONNX...")

        # Load YOLO model
        model = YOLO(pytorch_model_path)

        # Export to ONNX with optimization
        success = model.export(
            format="onnx",
            imgsz=(self.input_shape[2], self.input_shape[3]),  # Height, Width
            dynamic=dynamic_batch,
            simplify=True,
            opset=11,  # Compatible with TensorRT
            verbose=False,
        )

        if not success:
            raise RuntimeError("Failed to export YOLO model to ONNX")

        # Move the exported file to desired location
        exported_path = pytorch_model_path.replace(".pt", ".onnx")
        if exported_path != onnx_output_path:
            os.rename(exported_path, onnx_output_path)

        # Verify ONNX model
        self._verify_onnx_model(onnx_output_path)

        logger.info(f"ONNX model saved to: {onnx_output_path}")
        return onnx_output_path

    def _verify_onnx_model(self, onnx_path: str) -> None:
        """Verify ONNX model integrity"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise

    def convert_to_tensorrt(
        self,
        onnx_path: str,
        trt_output_path: str,
        precision: str = "fp16",
        max_batch_size: int = 1,
    ) -> str:
        """
        Convert ONNX model to TensorRT engine

        Args:
            onnx_path: Path to ONNX model
            trt_output_path: Output path for TensorRT engine
            precision: Precision mode (fp32, fp16, int8)
            max_batch_size: Maximum batch size

        Returns:
            Path to TensorRT engine
        """
        logger.info(f"Converting ONNX to TensorRT {precision.upper()}...")

        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                error_msg = "Failed to parse ONNX model. Errors:\n"
                for i in range(parser.num_errors):
                    error_msg += f"  {parser.get_error(i)}\n"
                raise RuntimeError(error_msg)

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size

        # Set precision
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✓ FP16 precision enabled")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("✓ INT8 precision enabled")
            # Note: INT8 calibration would need to be implemented
        else:
            logger.info("Using FP32 precision")

        # Set optimization level
        config.builder_optimization_level = 5  # Maximum optimization

        # Configure input shape
        input_tensor = network.get_input(0)
        if input_tensor.shape[0] == -1:  # Dynamic batch
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input_tensor.name,
                (1, *self.input_shape[1:]),  # min
                (max_batch_size, *self.input_shape[1:]),  # opt
                (max_batch_size, *self.input_shape[1:]),  # max
            )
            config.add_optimization_profile(profile)

        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        build_time = time.time() - start_time
        logger.info(f"✓ Engine built in {build_time:.2f} seconds")

        # Serialize and save engine
        with open(trt_output_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine saved to: {trt_output_path}")

        # Display engine info
        self._display_engine_info(engine)

        return trt_output_path

    def _display_engine_info(self, engine: trt.ICudaEngine) -> None:
        """Display TensorRT engine information"""
        info = []
        info.append(["Property", "Value"])
        info.append(["Max Batch Size", engine.max_batch_size])
        info.append(["Num Bindings", engine.num_bindings])
        info.append(["Has Implicit Batch", engine.has_implicit_batch_dimension])

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            binding_type = "Input" if is_input else "Output"
            info.append([f"Binding {i} ({binding_type})", f"{name}: {shape} {dtype}"])

        logger.info("TensorRT Engine Information:")
        print(tabulate(info, headers="firstrow", tablefmt="grid"))

    def benchmark_model(
        self, engine_path: str, num_iterations: int = 100, warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark TensorRT engine performance

        Args:
            engine_path: Path to TensorRT engine
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking engine: {engine_path}")

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_buffers(engine)

        # Create dummy input data
        input_data = np.random.randn(*self.input_shape).astype(np.float32)
        inputs[0].host = input_data.flatten()

        # Warmup
        logger.info(f"Warming up for {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            self._do_inference(context, bindings, inputs, outputs, stream)

        # Benchmark
        logger.info(f"Running benchmark for {num_iterations} iterations...")

        start_time = time.time()
        for _ in range(num_iterations):
            self._do_inference(context, bindings, inputs, outputs, stream)
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time

        # Get memory usage
        memory_info = self._get_memory_info()

        results = {
            "avg_inference_time_ms": avg_time * 1000,
            "fps": fps,
            "total_time_s": total_time,
            "iterations": num_iterations,
            "memory_usage_mb": memory_info,
        }

        # Display results
        self._display_benchmark_results(results)

        return results

    def _allocate_buffers(self, engine: trt.ICudaEngine):
        """Allocate buffers for TensorRT inference"""
        import pycuda.autoinit  # noqa F401
        import pycuda.driver as cuda  # noqa F401

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def _do_inference(self, context, bindings, inputs, outputs, stream):
        """Run TensorRT inference"""
        import pycuda.driver as cuda

        # Transfer input data to device
        for inp in inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Transfer predictions back
        for out in outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

        # Synchronize stream
        stream.synchronize()

        return [out["host"] for out in outputs]

    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        info = {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "percent": memory.percent,
        }

        if self.gpu_available:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                info.update(
                    {
                        "gpu_total_mb": gpu_memory.total / (1024 * 1024),
                        "gpu_used_mb": gpu_memory.used / (1024 * 1024),
                        "gpu_free_mb": gpu_memory.free / (1024 * 1024),
                    }
                )
            except Exception:
                pass

        return info

    def _display_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Display benchmark results in a formatted table"""
        data = [
            ["Metric", "Value"],
            ["Average Inference Time", f"{results['avg_inference_time_ms']:.2f} ms"],
            ["Frames Per Second", f"{results['fps']:.2f} FPS"],
            ["Total Benchmark Time", f"{results['total_time_s']:.2f} s"],
            ["Number of Iterations", results["iterations"]],
        ]

        memory = results["memory_usage_mb"]
        data.append(
            [
                "System Memory Used",
                f"{memory['used_mb']:.1f} MB ({memory['percent']:.1f}%)",
            ]
        )

        if "gpu_used_mb" in memory:
            data.append(["GPU Memory Used", f"{memory['gpu_used_mb']:.1f} MB"])

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(tabulate(data, headers="firstrow", tablefmt="grid"))

        # Performance assessment
        target_fps = 20.0  # From architecture requirements
        if results["fps"] >= target_fps:
            print(f"\n✓ Performance target met: {results['fps']:.1f} FPS >= {target_fps} FPS")
        else:
            print(f"\n⚠ Performance below target: {results['fps']:.1f} FPS < {target_fps} FPS")

    def convert_full_pipeline(self, output_dir: str, skip_existing: bool = True) -> Dict[str, str]:
        """
        Run complete conversion pipeline: PyTorch → ONNX → TensorRT

        Args:
            output_dir: Directory to save all converted models
            skip_existing: Skip conversion if files already exist

        Returns:
            Dictionary with paths to converted models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        paths = {
            "pytorch": output_dir / f"{self.model_name}.pt",
            "onnx": output_dir / f"{self.model_name}.onnx",
            "tensorrt": output_dir / f"{self.model_name}_fp16.trt",
        }

        logger.info(f"Starting full conversion pipeline for {self.model_name}")
        logger.info(f"Output directory: {output_dir}")

        # Step 1: Download PyTorch model
        if not paths["pytorch"].exists() or not skip_existing:
            self.download_model(str(output_dir))
        else:
            logger.info(f"Skipping download - PyTorch model exists: {paths['pytorch']}")

        # Step 2: Convert to ONNX
        if not paths["onnx"].exists() or not skip_existing:
            self.convert_to_onnx(str(paths["pytorch"]), str(paths["onnx"]))
        else:
            logger.info(f"Skipping ONNX conversion - file exists: {paths['onnx']}")

        # Step 3: Convert to TensorRT
        if not paths["tensorrt"].exists() or not skip_existing:
            self.convert_to_tensorrt(str(paths["onnx"]), str(paths["tensorrt"]))
        else:
            logger.info(f"Skipping TensorRT conversion - file exists: {paths['tensorrt']}")

        # Step 4: Benchmark
        benchmark_results = self.benchmark_model(str(paths["tensorrt"]))

        # Save benchmark results
        import json

        with open(output_dir / f"{self.model_name}_benchmark.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)

        logger.info("✓ Full conversion pipeline completed successfully!")

        return {k: str(v) for k, v in paths.items()}


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT optimized engines"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="YOLOv11n",
        choices=["YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"],
        help="YOLO model variant to convert",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./models/yolo_trt",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Input image size (width height)",
    )
    parser.add_argument(
        "--workspace-size", type=int, default=256, help="TensorRT workspace size in MB"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if output files already exist",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create converter
    converter = YOLOConverter(
        model_name=args.model,
        input_shape=(1, 3, args.input_size[1], args.input_size[0]),  # B,C,H,W
        workspace_size=args.workspace_size * 1024 * 1024,  # Convert MB to bytes
    )

    try:
        # Run conversion pipeline
        paths = converter.convert_full_pipeline(
            output_dir=args.output_dir, skip_existing=args.skip_existing
        )

        print("\n" + "=" * 60)
        print("CONVERSION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"PyTorch model: {paths['pytorch']}")
        print(f"ONNX model:    {paths['onnx']}")
        print(f"TensorRT engine: {paths['tensorrt']}")
        print("\nFiles are ready for deployment in ROS2 perception nodes.")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
