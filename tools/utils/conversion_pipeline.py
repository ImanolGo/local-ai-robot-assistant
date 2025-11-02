#!/usr/bin/env python3
"""
Standardized Model Conversion Pipeline Template
Universal framework for converting AI models following the architecture.md specification

Pipeline: PyTorch/HuggingFace → ONNX → TensorRT
Optimized for NVIDIA Jetson Orin Nano deployment
"""

import os
import sys
import argparse
import logging
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import tempfile

import torch
import tensorrt as trt
import numpy as np
import onnx
import onnxruntime as ort
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for model conversion pipeline"""

    model_name: str
    model_type: str
    input_shape: tuple
    output_dir: str
    precision: str = "fp16"
    workspace_size_mb: int = 256
    max_batch_size: int = 1
    optimization_level: int = 5
    dynamic_batch: bool = False
    calibration_data: Optional[str] = None
    custom_params: Dict[str, Any] = None


@dataclass
class ConversionResult:
    """Results from model conversion pipeline"""

    model_name: str
    model_type: str
    success: bool
    pytorch_path: Optional[str] = None
    onnx_path: Optional[str] = None
    tensorrt_path: Optional[str] = None
    benchmark_results: Optional[Dict[str, Any]] = None
    conversion_time_s: float = 0.0
    model_size_mb: float = 0.0
    error_message: str = ""


class BaseModelConverter(ABC):
    """Abstract base class for model converters"""

    def __init__(self, config: ConversionConfig):
        """
        Initialize base converter

        Args:
            config: Conversion configuration
        """
        self.config = config
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_or_create_pytorch_model(self) -> str:
        """
        Load or create PyTorch model

        Returns:
            Path to PyTorch model file
        """
        pass

    @abstractmethod
    def convert_pytorch_to_onnx(self, pytorch_path: str, onnx_path: str) -> str:
        """
        Convert PyTorch model to ONNX

        Args:
            pytorch_path: Path to PyTorch model
            onnx_path: Output path for ONNX model

        Returns:
            Path to ONNX model
        """
        pass

    def convert_onnx_to_tensorrt(self, onnx_path: str, trt_path: str) -> str:
        """
        Convert ONNX model to TensorRT engine (standardized implementation)

        Args:
            onnx_path: Path to ONNX model
            trt_path: Output path for TensorRT engine

        Returns:
            Path to TensorRT engine
        """
        logger.info(f"Converting ONNX to TensorRT {self.config.precision.upper()}...")

        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
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
        config.max_workspace_size = self.config.workspace_size_mb * 1024 * 1024

        # Set precision
        if self.config.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✓ FP16 precision enabled")
        elif self.config.precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("✓ INT8 precision enabled")
            if self.config.calibration_data:
                # Note: INT8 calibration would be implemented here
                logger.info("INT8 calibration data provided")
        else:
            logger.info("Using FP32 precision")

        # Set optimization level
        config.builder_optimization_level = self.config.optimization_level

        # Handle dynamic batch size
        if self.config.dynamic_batch:
            input_tensor = network.get_input(0)
            if input_tensor.shape[0] == -1:
                profile = builder.create_optimization_profile()
                profile.set_shape(
                    input_tensor.name,
                    (1, *self.config.input_shape[1:]),  # min
                    (self.config.max_batch_size, *self.config.input_shape[1:]),  # opt
                    (self.config.max_batch_size, *self.config.input_shape[1:]),  # max
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
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine saved to: {trt_path}")
        self._display_engine_info(engine)

        return trt_path

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

    def verify_onnx_model(self, onnx_path: str) -> None:
        """Verify ONNX model integrity"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model verification passed")

            # Test inference
            session = ort.InferenceSession(onnx_path)
            dummy_input = np.random.randn(*self.config.input_shape).astype(np.float32)

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: dummy_input})

            logger.info(
                f"✓ ONNX inference test passed - output shape: {output[0].shape}"
            )

        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise

    def benchmark_tensorrt_model(
        self, trt_path: str, num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark TensorRT model performance

        Args:
            trt_path: Path to TensorRT engine
            num_iterations: Number of benchmark iterations

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Benchmarking TensorRT model: {trt_path}")

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            logger.warning("PyCUDA not available - skipping benchmark")
            return {}

        # Load engine
        with open(trt_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_buffers(engine)

        # Create dummy input
        input_data = np.random.randn(*self.config.input_shape).astype(np.float32)
        inputs[0].host = input_data.flatten()

        # Warmup
        for _ in range(10):
            self._do_inference(context, bindings, inputs, outputs, stream)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            self._do_inference(context, bindings, inputs, outputs, stream)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0

        results = {
            "avg_inference_time_ms": avg_time,
            "std_inference_time_ms": std_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "fps": fps,
            "iterations": num_iterations,
            "input_shape": self.config.input_shape,
            "precision": self.config.precision,
        }

        logger.info(f"✓ Benchmark completed: {fps:.2f} FPS")
        return results

    def _allocate_buffers(self, engine):
        """Allocate TensorRT buffers"""
        import pycuda.driver as cuda

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

    def _do_inference(self, context, bindings, inputs, outputs, stream):
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

    def run_conversion_pipeline(self, skip_existing: bool = True) -> ConversionResult:
        """
        Run complete conversion pipeline

        Args:
            skip_existing: Skip conversion if files already exist

        Returns:
            ConversionResult with paths and metrics
        """
        start_time = time.time()

        try:
            logger.info(f"Starting conversion pipeline for {self.config.model_name}")
            logger.info(f"Model type: {self.config.model_type}")
            logger.info(f"Target input shape: {self.config.input_shape}")
            logger.info(f"Output directory: {self.config.output_dir}")

            # Define file paths
            pytorch_path = os.path.join(
                self.config.output_dir, f"{self.config.model_name}.pt"
            )
            onnx_path = os.path.join(
                self.config.output_dir, f"{self.config.model_name}.onnx"
            )
            trt_path = os.path.join(
                self.config.output_dir,
                f"{self.config.model_name}_{self.config.precision}.trt",
            )

            # Step 1: Load/create PyTorch model
            if not os.path.exists(pytorch_path) or not skip_existing:
                pytorch_path = self.load_or_create_pytorch_model()
            else:
                logger.info(
                    f"Skipping PyTorch model creation - file exists: {pytorch_path}"
                )

            # Step 2: Convert to ONNX
            if not os.path.exists(onnx_path) or not skip_existing:
                onnx_path = self.convert_pytorch_to_onnx(pytorch_path, onnx_path)
                self.verify_onnx_model(onnx_path)
            else:
                logger.info(f"Skipping ONNX conversion - file exists: {onnx_path}")

            # Step 3: Convert to TensorRT
            if not os.path.exists(trt_path) or not skip_existing:
                trt_path = self.convert_onnx_to_tensorrt(onnx_path, trt_path)
            else:
                logger.info(f"Skipping TensorRT conversion - file exists: {trt_path}")

            # Step 4: Benchmark
            benchmark_results = self.benchmark_tensorrt_model(trt_path)

            # Calculate metrics
            conversion_time = time.time() - start_time
            model_size_mb = (
                os.path.getsize(trt_path) / (1024 * 1024)
                if os.path.exists(trt_path)
                else 0
            )

            # Create result
            result = ConversionResult(
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                success=True,
                pytorch_path=pytorch_path,
                onnx_path=onnx_path,
                tensorrt_path=trt_path,
                benchmark_results=benchmark_results,
                conversion_time_s=conversion_time,
                model_size_mb=model_size_mb,
            )

            # Save conversion metadata
            self._save_conversion_metadata(result)

            logger.info("✓ Full conversion pipeline completed successfully!")
            return result

        except Exception as e:
            conversion_time = time.time() - start_time
            logger.error(f"Conversion pipeline failed: {e}")

            return ConversionResult(
                model_name=self.config.model_name,
                model_type=self.config.model_type,
                success=False,
                conversion_time_s=conversion_time,
                error_message=str(e),
            )

    def _save_conversion_metadata(self, result: ConversionResult):
        """Save conversion metadata to JSON file"""
        metadata = {
            "conversion_config": asdict(self.config),
            "conversion_result": asdict(result),
            "timestamp": time.time(),
        }

        metadata_path = os.path.join(
            self.config.output_dir, f"{self.config.model_name}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Conversion metadata saved to: {metadata_path}")


class ConversionPipeline:
    """Main pipeline orchestrator for model conversions"""

    def __init__(self):
        """Initialize conversion pipeline"""
        self.converters = {}
        self.results = []

    def register_converter(self, model_type: str, converter_class: type):
        """
        Register a model converter for a specific type

        Args:
            model_type: Type of model (e.g., 'yolo', 'depth', 'whisper')
            converter_class: Converter class that inherits from BaseModelConverter
        """
        self.converters[model_type] = converter_class
        logger.info(f"Registered converter for model type: {model_type}")

    def convert_model(
        self, config: ConversionConfig, skip_existing: bool = True
    ) -> ConversionResult:
        """
        Convert a model using the appropriate converter

        Args:
            config: Conversion configuration
            skip_existing: Skip conversion if files already exist

        Returns:
            ConversionResult with paths and metrics
        """
        if config.model_type not in self.converters:
            raise ValueError(
                f"No converter registered for model type: {config.model_type}"
            )

        converter_class = self.converters[config.model_type]
        converter = converter_class(config)

        result = converter.run_conversion_pipeline(skip_existing)
        self.results.append(result)

        return result

    def convert_multiple_models(
        self, configs: List[ConversionConfig], skip_existing: bool = True
    ) -> List[ConversionResult]:
        """
        Convert multiple models

        Args:
            configs: List of conversion configurations
            skip_existing: Skip conversion if files already exist

        Returns:
            List of ConversionResults
        """
        results = []

        for config in configs:
            logger.info(
                f"Converting model {len(results) + 1}/{len(configs)}: {config.model_name}"
            )

            try:
                result = self.convert_model(config, skip_existing)
                results.append(result)

                if result.success:
                    logger.info(f"✓ {config.model_name} converted successfully")
                else:
                    logger.error(
                        f"✗ {config.model_name} conversion failed: {result.error_message}"
                    )

            except Exception as e:
                logger.error(f"✗ {config.model_name} conversion failed: {e}")
                results.append(
                    ConversionResult(
                        model_name=config.model_name,
                        model_type=config.model_type,
                        success=False,
                        error_message=str(e),
                    )
                )

        return results

    def display_results_summary(self):
        """Display summary of all conversion results"""
        if not self.results:
            logger.info("No conversion results to display")
            return

        # Create summary table
        headers = ["Model", "Type", "Status", "FPS", "Size (MB)", "Time (s)"]
        rows = []

        for result in self.results:
            status = "✓" if result.success else "✗"
            fps = (
                result.benchmark_results.get("fps", 0)
                if result.benchmark_results
                else 0
            )

            rows.append(
                [
                    result.model_name,
                    result.model_type,
                    status,
                    f"{fps:.2f}" if fps > 0 else "N/A",
                    (
                        f"{result.model_size_mb:.1f}"
                        if result.model_size_mb > 0
                        else "N/A"
                    ),
                    f"{result.conversion_time_s:.1f}",
                ]
            )

        print("\n" + "=" * 80)
        print("MODEL CONVERSION RESULTS SUMMARY")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Statistics
        total_models = len(self.results)
        successful_conversions = sum(1 for r in self.results if r.success)

        print(f"\nConversion Statistics:")
        print(f"Total models: {total_models}")
        print(f"Successful conversions: {successful_conversions}/{total_models}")
        print(f"Success rate: {successful_conversions/total_models*100:.1f}%")

    def save_batch_results(self, output_path: str):
        """Save batch conversion results to JSON file"""
        batch_results = {
            "timestamp": time.time(),
            "total_models": len(self.results),
            "successful_conversions": sum(1 for r in self.results if r.success),
            "results": [asdict(result) for result in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(batch_results, f, indent=2)

        logger.info(f"Batch results saved to: {output_path}")


def create_sample_config(
    model_type: str, output_dir: str = "./models"
) -> ConversionConfig:
    """
    Create sample configuration for different model types

    Args:
        model_type: Type of model ('yolo', 'depth', 'whisper')
        output_dir: Output directory for converted models

    Returns:
        ConversionConfig with appropriate settings
    """
    configs = {
        "yolo": ConversionConfig(
            model_name="yolov8n",
            model_type="yolo",
            input_shape=(1, 3, 640, 480),
            output_dir=os.path.join(output_dir, "yolo_trt"),
            precision="fp16",
            workspace_size_mb=256,
            max_batch_size=1,
        ),
        "depth": ConversionConfig(
            model_name="fastdepth",
            model_type="depth",
            input_shape=(1, 3, 240, 320),
            output_dir=os.path.join(output_dir, "depth_trt"),
            precision="fp16",
            workspace_size_mb=256,
            max_batch_size=1,
        ),
        "whisper": ConversionConfig(
            model_name="whisper_tiny",
            model_type="whisper",
            input_shape=(1, 80, 3000),  # Mel spectrogram
            output_dir=os.path.join(output_dir, "whisper_trt"),
            precision="fp16",
            workspace_size_mb=256,
            max_batch_size=1,
        ),
    }

    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")

    return configs[model_type]


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Standardized Model Conversion Pipeline for Jetson Orin Nano"
    )
    parser.add_argument(
        "--model-type",
        choices=["yolo", "depth", "whisper"],
        help="Type of model to convert",
    )
    parser.add_argument(
        "--output-dir", default="./models", help="Output directory for converted models"
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode",
    )
    parser.add_argument(
        "--workspace-size", type=int, default=256, help="TensorRT workspace size in MB"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if output files already exist",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize pipeline
    pipeline = ConversionPipeline()

    # Note: In actual implementation, converters would be registered here
    # pipeline.register_converter('yolo', YOLOConverter)
    # pipeline.register_converter('depth', FastDepthConverter)
    # pipeline.register_converter('whisper', WhisperConverter)

    try:
        if args.model_type:
            # Convert single model
            config = create_sample_config(args.model_type, args.output_dir)
            config.precision = args.precision
            config.workspace_size_mb = args.workspace_size

            logger.info(f"Converting {args.model_type} model...")
            logger.info(
                f"Note: This is a template - actual converter implementation needed"
            )

            # In actual implementation:
            # result = pipeline.convert_model(config, args.skip_existing)

        else:
            logger.info("Model Conversion Pipeline Template")
            logger.info("=====================================")
            logger.info("")
            logger.info(
                "This template provides a standardized framework for model conversion."
            )
            logger.info("To use it, implement specific converters for each model type:")
            logger.info("")
            logger.info("1. Inherit from BaseModelConverter")
            logger.info("2. Implement load_or_create_pytorch_model()")
            logger.info("3. Implement convert_pytorch_to_onnx()")
            logger.info("4. Register converter with pipeline")
            logger.info("")
            logger.info("Example usage:")
            logger.info("  python tools/utils/conversion_pipeline.py --model-type yolo")
            logger.info(
                "  python tools/utils/conversion_pipeline.py --model-type depth"
            )
            logger.info(
                "  python tools/utils/conversion_pipeline.py --model-type whisper"
            )
            logger.info("")
            logger.info("For actual model conversion, use specific scripts:")
            logger.info("  python tools/conversion/convert_yolo.py")
            logger.info("  python tools/conversion/convert_depth.py")
            logger.info("  python tools/conversion/convert_whisper.py")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
