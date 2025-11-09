#!/usr/bin/env python3
"""
YOLO Model Conversion Script
Converts YOLOv11 models to TensorRT using Ultralytics native export

According to architecture.md:
- Target: YOLOv11n optimized with TensorRT FP16
- Pipeline: Direct PyTorch → TensorRT using Ultralytics export
- Performance Target: 20+ FPS at 640x480 resolution

This script replaces the previous complex PyTorch → ONNX → TensorRT pipeline
with Ultralytics' native TensorRT export functionality as documented at:
https://docs.ultralytics.com/modes/export/

Key improvements over manual conversion:
1. Uses Ultralytics' optimized TensorRT export (format="engine")
2. Supports all modern export arguments (workspace, int8, dynamic, etc.)
3. Eliminates manual ONNX intermediate step
4. More reliable and maintainable
5. Follows official Ultralytics documentation

Usage Examples:
    # Basic FP16 conversion
    python convert_yolo.py --model yolo11n --precision fp16

    # INT8 with calibration
    python convert_yolo.py --model yolo11n --precision int8 --calibration-data coco8.yaml

    # Compare precisions
    python convert_yolo.py --model yolo11n --compare-precisions

    # Demo the basic Ultralytics approach
    python convert_yolo.py --demo

Export Arguments (based on Ultralytics documentation):
- format: 'engine' for TensorRT
- imgsz: Image size (int or tuple)
- half: FP16 quantization (bool)
- int8: INT8 quantization (bool)
- dynamic: Dynamic input sizes (bool)
- workspace: TensorRT workspace in GB (float)
- batch: Export batch size (int)
- device: GPU device ('0', 'cpu', etc.)
- data: Calibration dataset for INT8 (str)
- nms: Include NMS in model (bool)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Optional imports with fallbacks
try:
    import nvidia_ml_py3 as nvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    nvml = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

    def tabulate(data, headers=None, tablefmt="grid"):
        """Fallback tabulate implementation"""
        return "\n".join([str(row) for row in data])


try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class YOLOConverter:
    """
    Converts YOLO models to TensorRT using Ultralytics native export

    Uses the simplified Ultralytics export API for direct PyTorch → TensorRT conversion
    Optimized for NVIDIA Jetson Orin Nano deployment
    """

    def __init__(
        self,
        model_name: str = "yolo11n",
        input_size: tuple = (640, 480),
        workspace_gb: float = 1.0,
    ):
        """
        Initialize YOLO converter

        Args:
            model_name: YOLO model variant (yolo11n, yolo11s, etc.)
            input_size: Input image size (width, height)
            workspace_gb: TensorRT workspace size in GB
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

        self.model_name = model_name
        self.input_size = input_size  # (width, height)
        self.workspace_gb = workspace_gb

        # Initialize NVIDIA ML for GPU monitoring
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_available = True
            except Exception:
                self.nvml_available = False
                logger.warning("NVIDIA ML initialization failed - GPU monitoring disabled")
        else:
            self.nvml_available = False
            logger.warning("NVIDIA ML not available - GPU monitoring disabled")

        # Check GPU availability using PyTorch (more reliable than NVML)
        try:
            import torch

            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("No CUDA GPU detected")
        except ImportError:
            self.gpu_available = False
            logger.warning("PyTorch not available - falling back to CPU")

    def convert_to_tensorrt(
        self,
        output_dir: str,
        precision: str = "fp16",
        dynamic: bool = False,
        int8_calibration_data: Optional[str] = None,
        batch_size: int = 1,
        optimize_for_jetson: bool = True,
    ) -> str:
        """
        Convert YOLO model directly to TensorRT engine using Ultralytics export

        Args:
            output_dir: Directory to save the TensorRT engine
            precision: Precision mode ("fp16", "int8", "fp32")
            dynamic: Enable dynamic input sizes
            int8_calibration_data: Path to calibration dataset for INT8 quantization
            batch_size: Export batch size
            optimize_for_jetson: Apply Jetson-specific optimizations

        Returns:
            Path to the TensorRT engine file
        """
        logger.info(f"Converting {self.model_name} to TensorRT {precision.upper()}...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load YOLO model
        model = YOLO(f"{self.model_name}.pt")

        # Prepare export arguments based on documentation
        export_args = {
            "format": "engine",  # TensorRT format
            "imgsz": self.input_size,  # Image size (width, height) or int for square
            "half": precision == "fp16",  # FP16 quantization
            "int8": precision == "int8",  # INT8 quantization
            "dynamic": dynamic,  # Dynamic input sizes
            "workspace": self.workspace_gb,  # TensorRT workspace size in GB
            "batch": batch_size,  # Export batch size
            "device": (
                "0" if self.gpu_available else "cpu"
            ),  # GPU device (automatically uses GPU for TensorRT)
            "verbose": True,  # Verbose output
        }

        # Add INT8 calibration dataset if provided
        if precision == "int8" and int8_calibration_data:
            export_args["data"] = int8_calibration_data

        # For Jetson optimization, ensure NMS is included for end-to-end inference
        if optimize_for_jetson:
            export_args["nms"] = True  # Include NMS in model for faster inference

        logger.info(f"Export arguments: {export_args}")

        # Export the model
        start_time = time.time()
        try:
            export_result = model.export(**export_args)
            export_time = time.time() - start_time

            # The exported engine file path
            if isinstance(export_result, str):
                engine_path = export_result
            else:
                # For newer versions, export might return Path object or model object
                engine_path = str(export_result)

            # Move to desired output directory if needed
            expected_path = output_dir / f"{self.model_name}_{precision}.engine"
            if engine_path != str(expected_path):
                import shutil

                shutil.move(engine_path, expected_path)
                engine_path = str(expected_path)

            logger.info(f"✓ TensorRT export completed in {export_time:.2f} seconds")
            logger.info(f"Engine saved to: {engine_path}")

            return engine_path

        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            logger.error("Make sure you have:")
            logger.error("1. NVIDIA GPU with TensorRT installed")
            logger.error("2. Proper CUDA environment")
            logger.error("3. Ultralytics package with TensorRT support")
            raise

    def validate_accuracy(
        self,
        original_model_path: str,
        engine_path: str,
        validation_data: Optional[str] = None,
        imgsz: int = 640,
    ) -> Dict[str, Any]:
        """
        Validate converted model accuracy against original PyTorch model

        Args:
            original_model_path: Path to original PyTorch model
            engine_path: Path to TensorRT engine
            validation_data: Path to validation dataset (uses COCO val if None)
            imgsz: Image size for validation

        Returns:
            Dictionary with accuracy comparison results
        """
        logger.info("Validating converted model accuracy...")

        try:
            # Load original PyTorch model
            original_model = YOLO(original_model_path)
            logger.info(f"Loaded original model: {original_model_path}")

            # Load TensorRT model
            trt_model = YOLO(engine_path)
            logger.info(f"Loaded TensorRT model: {engine_path}")

            # Use COCO val dataset if no validation data provided
            if validation_data is None:
                # Download COCO val sample for validation (small subset)
                validation_data = "coco8.yaml"  # Small COCO subset for quick validation
                logger.info("Using COCO8 dataset for quick validation")

            # Validate original model
            logger.info("Validating original PyTorch model...")
            original_results = original_model.val(
                data=validation_data,
                imgsz=imgsz,
                verbose=False,
                save=False,
                plots=False,
            )

            # Validate TensorRT model
            logger.info("Validating TensorRT model...")
            trt_results = trt_model.val(
                data=validation_data,
                imgsz=imgsz,
                verbose=False,
                save=False,
                plots=False,
            )

            # Extract mAP metrics
            original_map50 = original_results.box.map50
            original_map50_95 = original_results.box.map

            trt_map50 = trt_results.box.map50
            trt_map50_95 = trt_results.box.map

            # Calculate accuracy drop
            map50_drop = abs(original_map50 - trt_map50)
            map50_95_drop = abs(original_map50_95 - trt_map50_95)

            # Check if accuracy drop is within acceptable range (2%)
            accuracy_acceptable = map50_95_drop <= 0.02

            validation_results = {
                "original_map50": float(original_map50),
                "original_map50_95": float(original_map50_95),
                "trt_map50": float(trt_map50),
                "trt_map50_95": float(trt_map50_95),
                "map50_drop": float(map50_drop),
                "map50_95_drop": float(map50_95_drop),
                "accuracy_acceptable": bool(accuracy_acceptable),
                "validation_dataset": validation_data,
                "image_size": int(imgsz),
            }

            # Display results
            self._display_accuracy_results(validation_results)

            return validation_results

        except Exception as e:
            logger.error(f"Accuracy validation failed: {e}")
            logger.warning("Skipping accuracy validation - continuing with conversion")
            return {
                "error": str(e),
                "accuracy_acceptable": None,
                "validation_skipped": True,
            }

    def _display_accuracy_results(self, results: Dict[str, Any]) -> None:
        """Display accuracy validation results"""
        if results.get("validation_skipped"):
            print("\n⚠️ Accuracy validation was skipped due to error")
            return

        data = [
            ["Metric", "Original PyTorch", "TensorRT", "Drop"],
            [
                "mAP@0.5",
                f"{results['original_map50']:.4f}",
                f"{results['trt_map50']:.4f}",
                f"{results['map50_drop']:.4f}",
            ],
            [
                "mAP@0.5:0.95",
                f"{results['original_map50_95']:.4f}",
                f"{results['trt_map50_95']:.4f}",
                f"{results['map50_95_drop']:.4f}",
            ],
        ]

        print("\n" + "=" * 60)
        print("ACCURACY VALIDATION RESULTS")
        print("=" * 60)
        print(tabulate(data, headers="firstrow", tablefmt="grid"))

        if results["accuracy_acceptable"]:
            print(
                f"\n✅ Accuracy validation passed: mAP drop {results['map50_95_drop']:.4f} <= 0.02"
            )
        else:
            print(
                f"\n⚠️ Accuracy validation warning: mAP drop {results['map50_95_drop']:.4f} > 0.02"
            )
            print("Consider using higher precision or adjusting conversion parameters")

    def benchmark_model(
        self,
        engine_path: str,
        test_image: Optional[str] = None,
        num_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Benchmark TensorRT engine using YOLO inference

        Args:
            engine_path: Path to TensorRT engine
            test_image: Path to test image (uses built-in test image if None)
            num_iterations: Number of inference iterations for benchmarking

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking TensorRT engine: {engine_path}")

        # Load the TensorRT model
        trt_model = YOLO(engine_path)

        # Use test image or default
        if test_image is None:
            test_image = "https://ultralytics.com/images/bus.jpg"

        # Warmup
        logger.info("Warming up model...")
        for _ in range(10):
            _ = trt_model(test_image, verbose=False)

        # Benchmark inference
        logger.info(f"Running {num_iterations} inference iterations...")
        start_time = time.time()

        for _ in range(num_iterations):
            results = trt_model(test_image, verbose=False)

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time

        # Get memory usage
        memory_info = self._get_memory_info()

        # Get detection info from last result
        detection_count = 0
        if results and len(results) > 0:
            detection_count = len(results[0].boxes) if results[0].boxes is not None else 0

        benchmark_results = {
            "avg_inference_time_ms": avg_time * 1000,
            "fps": fps,
            "total_time_s": total_time,
            "iterations": num_iterations,
            "detections_last_run": detection_count,
            "memory_usage": memory_info,
            "engine_path": engine_path,
            "input_size": self.input_size,
            "model_name": self.model_name,
        }

        # Display results
        self._display_benchmark_results(benchmark_results)

        return benchmark_results

    def _get_memory_info(self) -> Dict[str, float]:
        """Get system and GPU memory information"""
        info = {}

        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            info.update(
                {
                    "system_total_mb": memory.total / (1024 * 1024),
                    "system_available_mb": memory.available / (1024 * 1024),
                    "system_used_mb": memory.used / (1024 * 1024),
                    "system_percent": memory.percent,
                }
            )
        else:
            info.update(
                {
                    "system_total_mb": 0,
                    "system_available_mb": 0,
                    "system_used_mb": 0,
                    "system_percent": 0,
                }
            )

        if self.gpu_available and self.nvml_available:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                info.update(
                    {
                        "gpu_total_mb": gpu_memory.total / (1024 * 1024),
                        "gpu_used_mb": gpu_memory.used / (1024 * 1024),
                        "gpu_free_mb": gpu_memory.free / (1024 * 1024),
                        "gpu_percent": (gpu_memory.used / gpu_memory.total) * 100,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")

        return info

    def _display_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Display benchmark results in a formatted table"""
        data = [
            ["Metric", "Value"],
            ["Model", results["model_name"]],
            ["Input Size", f"{results['input_size'][0]}x{results['input_size'][1]}"],
            ["Average Inference Time", f"{results['avg_inference_time_ms']:.2f} ms"],
            ["Frames Per Second", f"{results['fps']:.2f} FPS"],
            ["Total Benchmark Time", f"{results['total_time_s']:.2f} s"],
            ["Iterations", results["iterations"]],
            ["Detections (Last Run)", results["detections_last_run"]],
        ]

        memory = results["memory_usage"]
        data.append(
            [
                "System Memory Used",
                f"{memory['system_used_mb']:.1f} MB ({memory['system_percent']:.1f}%)",
            ]
        )

        if "gpu_used_mb" in memory:
            data.append(
                [
                    "GPU Memory Used",
                    f"{memory['gpu_used_mb']:.1f} MB ({memory['gpu_percent']:.1f}%)",
                ]
            )

        print("\n" + "=" * 50)
        print("TENSORRT BENCHMARK RESULTS")
        print("=" * 50)
        print(tabulate(data, headers="firstrow", tablefmt="grid"))

        # Performance assessment against architecture requirements
        target_fps = 20.0  # From architecture.md requirements
        target_inference_ms = 50.0  # Target: <50ms inference time on Jetson

        if results["fps"] >= target_fps:
            print(f"\n✅ Performance target met: {results['fps']:.1f} FPS >= {target_fps} FPS")
        else:
            print(f"\n⚠️ Performance below target: {results['fps']:.1f} FPS < {target_fps} FPS")
            print("Consider:")
            print("- Reducing input resolution")
            print("- Using INT8 quantization")
            print("- Optimizing TensorRT workspace size")

        if results["avg_inference_time_ms"] <= target_inference_ms:
            print(
                f"✅ Inference time target met: {results['avg_inference_time_ms']:.1f}ms <=\
                      {target_inference_ms}ms"
            )
        else:
            print(
                f"⚠️ Inference time above target: {results['avg_inference_time_ms']:.1f}ms >\
                      {target_inference_ms}ms"
            )

    def convert_with_multiple_precisions(
        self,
        output_dir: str,
        precisions: list = None,
        int8_calibration_data: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Convert model to multiple TensorRT precisions for comparison

        Args:
            output_dir: Output directory for engines
            precisions: List of precisions to test (default: ["fp16", "int8"])
            int8_calibration_data: Calibration data for INT8

        Returns:
            Dictionary mapping precision to engine path
        """
        if precisions is None:
            precisions = ["fp16", "int8"]

        results = {}
        benchmark_results = {}

        for precision in precisions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Converting to {precision.upper()}")
            logger.info(f"{'='*60}")

            try:
                engine_path = self.convert_to_tensorrt(
                    output_dir=output_dir,
                    precision=precision,
                    int8_calibration_data=(int8_calibration_data if precision == "int8" else None),
                )
                results[precision] = engine_path

                # Benchmark each engine
                benchmark_results[precision] = self.benchmark_model(engine_path)

            except Exception as e:
                logger.error(f"Failed to convert {precision}: {e}")
                continue

        # Save comparison results
        output_path = Path(output_dir)
        with open(output_path / f"{self.model_name}_precision_comparison.json", "w") as f:
            json.dump(benchmark_results, f, indent=2, cls=NumpyEncoder)

        # Display comparison table
        self._display_precision_comparison(benchmark_results)

        return results

    def _display_precision_comparison(self, benchmark_results: Dict[str, Dict[str, Any]]) -> None:
        """Display comparison of different precision modes"""
        if not benchmark_results:
            return

        print("\n" + "=" * 80)
        print("PRECISION COMPARISON")
        print("=" * 80)

        data = [["Precision", "FPS", "Inference Time (ms)", "GPU Memory (MB)"]]

        for precision, results in benchmark_results.items():
            fps = results.get("fps", 0)
            inf_time = results.get("avg_inference_time_ms", 0)
            gpu_mem = results.get("memory_usage", {}).get("gpu_used_mb", "N/A")

            data.append(
                [
                    precision.upper(),
                    f"{fps:.2f}",
                    f"{inf_time:.2f}",
                    (f"{gpu_mem:.1f}" if isinstance(gpu_mem, (int, float)) else str(gpu_mem)),
                ]
            )

        print(tabulate(data, headers="firstrow", tablefmt="grid"))

        # Find best precision for Jetson deployment
        best_precision = max(
            benchmark_results.keys(), key=lambda p: benchmark_results[p].get("fps", 0)
        )
        best_fps = benchmark_results[best_precision]["fps"]

        print(f"\n🎯 Recommended precision for Jetson deployment: {best_precision.upper()}")
        print(f"   Performance: {best_fps:.2f} FPS")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Convert YOLO models to TensorRT using Ultralytics native export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with FP16 precision
  python convert_yolo.py --model yolo11n --precision fp16

  # Convert with INT8 quantization and calibration data
  python convert_yolo.py --model yolo11n --precision int8 --calibration-data coco8.yaml

  # Compare multiple precisions
  python convert_yolo.py --model yolo11n --compare-precisions

  # Optimize for Jetson Orin Nano
  python convert_yolo.py --model yolo11n --precision fp16 --jetson-optimize
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        default="yolo11n",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="YOLO model variant to convert (default: yolo11n)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./models/yolo_trt",
        help="Output directory for converted models (default: ./models/yolo_trt)",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=[640, 480],
        metavar=("WIDTH", "HEIGHT"),
        help="Input image size in pixels (default: 640 480)",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=1.0,
        help="TensorRT workspace size in GB (default: 1.0)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "int8", "fp32"],
        default="fp16",
        help="TensorRT precision mode (default: fp16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Export batch size (default: 1)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input sizes",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration dataset YAML for INT8 quantization (e.g., coco8.yaml)",
    )
    parser.add_argument(
        "--test-image",
        type=str,
        help="Path to test image for benchmarking (uses default if not provided)",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--jetson-optimize",
        action="store_true",
        help="Apply Jetson-specific optimizations (includes NMS in model)",
    )
    parser.add_argument(
        "--compare-precisions",
        action="store_true",
        help="Convert and compare FP16 and INT8 precisions",
    )
    parser.add_argument(
        "--validate-accuracy",
        action="store_true",
        help="Validate accuracy against original model (requires validation dataset)",
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        help="Path to validation dataset YAML (default: coco8.yaml for quick validation)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.precision == "int8" and not args.calibration_data and not args.compare_precisions:
        logger.warning("INT8 quantization without calibration data may result in poor accuracy")
        logger.warning("Consider providing --calibration-data for better results")

    # Create converter
    converter = YOLOConverter(
        model_name=args.model,
        input_size=tuple(args.input_size),
        workspace_gb=args.workspace_gb,
    )

    try:
        if args.compare_precisions:
            # Compare multiple precisions
            logger.info("🔄 Comparing multiple precisions...")

            precisions = ["fp16"]
            if args.calibration_data:
                precisions.append("int8")
            else:
                logger.info("Skipping INT8 comparison - no calibration data provided")

            engine_paths = converter.convert_with_multiple_precisions(
                output_dir=args.output_dir,
                precisions=precisions,
                int8_calibration_data=args.calibration_data,
            )

            print("\n" + "=" * 60)
            print("PRECISION COMPARISON COMPLETED")
            print("=" * 60)
            for precision, path in engine_paths.items():
                print(f"{precision.upper()} engine: {path}")

        else:
            # Single precision conversion
            logger.info(f"🔄 Converting {args.model} to TensorRT {args.precision.upper()}...")

            engine_path = converter.convert_to_tensorrt(
                output_dir=args.output_dir,
                precision=args.precision,
                dynamic=args.dynamic,
                int8_calibration_data=args.calibration_data,
                batch_size=args.batch_size,
                optimize_for_jetson=args.jetson_optimize,
            )

            # Benchmark the model
            logger.info("🔄 Benchmarking converted model...")
            benchmark_results = converter.benchmark_model(
                engine_path=engine_path,
                test_image=args.test_image,
                num_iterations=args.benchmark_iterations,
            )

            # Validate accuracy if requested
            validation_results = {}
            if args.validate_accuracy:
                logger.info("🔄 Validating model accuracy...")
                original_model_path = f"{args.model}.pt"
                validation_results = converter.validate_accuracy(
                    original_model_path=original_model_path,
                    engine_path=engine_path,
                    validation_data=args.validation_data,
                )

                # Combine results
                benchmark_results["accuracy_validation"] = validation_results

            # Save benchmark results
            output_dir = Path(args.output_dir)
            benchmark_file = output_dir / f"{args.model}_{args.precision}_benchmark.json"
            with open(benchmark_file, "w") as f:
                json.dump(benchmark_results, f, indent=2, cls=NumpyEncoder)

            print("\n" + "=" * 60)
            print("CONVERSION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"TensorRT Engine: {engine_path}")
            print(f"Precision: {args.precision.upper()}")
            print(f"Input Size: {args.input_size[0]}x{args.input_size[1]}")
            print(f"Performance: {benchmark_results['fps']:.2f} FPS")
            print(f"Benchmark Results: {benchmark_file}")
            print("\n✅ Ready for deployment in ROS2 perception nodes!")

            # Jetson deployment tips
            if args.jetson_optimize:
                print("\n🎯 Jetson Optimization Tips:")
                print("- Model includes NMS for faster end-to-end inference")
                print("- Consider setting GPU to max performance mode:")
                print("  sudo jetson_clocks")
                print("- Monitor thermal throttling during inference")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Ensure NVIDIA GPU is available and TensorRT is installed")
        logger.error("2. Check CUDA installation: nvidia-smi")
        logger.error("3. Verify Ultralytics installation: pip show ultralytics")
        logger.error("4. For Jetson: ensure JetPack SDK is properly installed")
        sys.exit(1)


def demo_ultralytics_tensorrt_export():
    """
    Demo function showing the correct Ultralytics TensorRT export according to documentation
    """
    print("=" * 60)
    print("ULTRALYTICS TENSORRT EXPORT DEMO")
    print("=" * 60)

    # Check if Ultralytics is available
    if not ULTRALYTICS_AVAILABLE:
        print("❌ Ultralytics not installed. Install with: pip install ultralytics")
        return

    # Example from Ultralytics documentation
    try:
        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        print("✅ Model loaded successfully")

        # Export the model to TensorRT (basic example)
        print("🔄 Exporting to TensorRT engine format...")
        model.export(format="engine")  # creates 'yolo11n.engine'

        print("✅ Basic export completed: yolo11n.engine")

        # Load the exported TensorRT model
        trt_model = YOLO("yolo11n.engine")

        print("✅ TensorRT model loaded successfully")

        # Run inference (example from docs)
        print("🔄 Running inference on test image...")
        results = trt_model("https://ultralytics.com/images/bus.jpg")

        print(f"✅ Inference completed! Detected {len(results[0].boxes)} objects")

        # Advanced export with all options
        print("\n🔄 Advanced export with optimization options...")
        advanced_result = model.export(
            format="engine",  # TensorRT format
            imgsz=640,  # Image size
            half=True,  # FP16 quantization
            dynamic=False,  # Static input sizes for better optimization
            workspace=1.0,  # 1GB workspace
            int8=False,  # Skip INT8 for this demo
            batch=1,  # Batch size 1
            device=0,  # GPU device 0
            verbose=True,  # Verbose output
        )

        print(f"✅ Advanced export completed: {advanced_result}")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("Make sure you have:")
        print("- NVIDIA GPU with CUDA")
        print("- TensorRT installed")
        print("- Ultralytics with TensorRT support")


if __name__ == "__main__":
    # Check if running demo
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_ultralytics_tensorrt_export()
    else:
        main()
