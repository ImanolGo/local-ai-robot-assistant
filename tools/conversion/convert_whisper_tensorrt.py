#!/usr/bin/env python3
"""
Convert Whisper to TensorRT for Jetson Optimization

This script converts the Whisper Tiny model to TensorRT format for optimal performance
on NVIDIA Jetson Orin Nano. This is an alternative to faster-whisper for achieving
the <0.3x real-time factor and <300MB memory targets.

Usage:
    python tools/conversion/convert_whisper_tensorrt.py [--validate] [--benchmark]

Requirements:
    - TensorRT 8.4+
    - ONNX model of Whisper Tiny
    - NVIDIA Jetson with CUDA
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import whisper

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "whisper_tiny_trt"

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class WhisperTensorRTConverter:
    """Convert Whisper model to TensorRT format for Jetson optimization."""

    def __init__(self):
        """Initialize the converter."""
        self.model_dir = MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def export_whisper_to_onnx(self) -> bool:
        """Export Whisper Tiny to ONNX format.

        Returns:
            bool: True if export successful
        """
        try:
            print("📥 Loading Whisper Tiny model...")

            # Load the whisper model
            model = whisper.load_model("tiny")
            model.eval()

            # Create dummy input for encoder
            mel_shape = (1, 80, 3000)  # Batch, mel bins, time steps
            dummy_mel = torch.randn(mel_shape)

            # Export encoder to ONNX
            encoder_path = self.model_dir / "whisper_tiny_encoder.onnx"
            print(f"🔄 Exporting encoder to {encoder_path}...")

            torch.onnx.export(
                model.encoder,
                dummy_mel,
                encoder_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["mel"],
                output_names=["encoder_output"],
                dynamic_axes={
                    "mel": {2: "mel_length"},
                    "encoder_output": {1: "encoder_length"},
                },
            )

            # For decoder, we need a more complex setup
            # Create dummy inputs for decoder
            tokens = torch.tensor([[50258, 50259, 50359, 50363]])  # Start tokens
            audio_features = torch.randn(1, 1500, 384)  # Encoder output shape

            # Export decoder to ONNX
            decoder_path = self.model_dir / "whisper_tiny_decoder.onnx"
            print(f"🔄 Exporting decoder to {decoder_path}...")

            # Create a simplified decoder for export
            class WhisperDecoderWrapper(torch.nn.Module):
                def __init__(self, decoder):
                    super().__init__()
                    self.decoder = decoder

                def forward(self, tokens, audio_features):
                    return self.decoder(tokens, audio_features)

            decoder_wrapper = WhisperDecoderWrapper(model.decoder)

            torch.onnx.export(
                decoder_wrapper,
                (tokens, audio_features),
                decoder_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["tokens", "audio_features"],
                output_names=["logits"],
                dynamic_axes={
                    "tokens": {1: "token_length"},
                    "audio_features": {1: "audio_length"},
                },
            )

            print("✅ ONNX export completed")
            return True

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return False

    def convert_onnx_to_tensorrt(
        self, onnx_path: Path, engine_path: Path, precision: str = "fp16"
    ) -> bool:
        """Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            precision: Precision mode (fp32, fp16, int8)

        Returns:
            bool: True if conversion successful
        """
        try:
            print(f"🔄 Converting {onnx_path} to TensorRT...")

            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX model
            with open(onnx_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    print("❌ Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        print(f"Error {error}: {parser.get_error(error)}")
                    return False

            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 28  # 256MB

            # Set precision
            if precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("✅ FP16 optimization enabled")
                else:
                    print("⚠️ FP16 not supported, falling back to FP32")
            elif precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    print("✅ INT8 optimization enabled")
                else:
                    print("⚠️ INT8 not supported, falling back to FP32")

            # Build engine
            print("🔨 Building TensorRT engine... (this may take a while)")
            serialized_engine = builder.build_serialized_network(network, config)

            if serialized_engine is None:
                print("❌ Failed to build TensorRT engine")
                return False

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            print(f"✅ TensorRT engine saved: {engine_path}")
            return True

        except Exception as e:
            print(f"❌ TensorRT conversion failed: {e}")
            return False

    def validate_tensorrt_model(self, engine_path: Path) -> bool:
        """Validate the TensorRT model by running inference.

        Args:
            engine_path: Path to TensorRT engine

        Returns:
            bool: True if validation successful
        """
        try:
            import pycuda.driver as cuda

            print(f"🧪 Validating TensorRT model: {engine_path}")

            # Load engine
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                print("❌ Failed to load engine")
                return False

            context = engine.create_execution_context()

            # Get input/output specs
            input_shape = engine.get_binding_shape(0)
            output_shape = engine.get_binding_shape(1)

            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")

            # Create dummy input
            input_size = trt.volume(input_shape)
            output_size = trt.volume(output_shape)

            # Allocate memory
            h_input = np.random.randn(input_size).astype(np.float32)
            h_output = np.empty(output_size, dtype=np.float32)

            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)

            # Copy input to device
            cuda.memcpy_htod(d_input, h_input)

            # Run inference
            start_time = time.perf_counter()
            context.execute_v2([int(d_input), int(d_output)])
            cuda.Context.synchronize()
            inference_time = time.perf_counter() - start_time

            # Copy output back
            cuda.memcpy_dtoh(h_output, d_output)

            print(f"✅ Validation successful! Inference time: {inference_time*1000:.1f}ms")
            return True

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    def benchmark_tensorrt_model(self, engine_path: Path, num_runs: int = 100) -> dict:
        """Benchmark the TensorRT model performance.

        Args:
            engine_path: Path to TensorRT engine
            num_runs: Number of benchmark runs

        Returns:
            dict: Benchmark results
        """
        try:
            import psutil
            import pycuda.driver as cuda

            print(f"🏃 Benchmarking TensorRT model: {engine_path}")

            # Load engine
            runtime = trt.Runtime(TRT_LOGGER)
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()

            # Get input/output specs
            input_shape = engine.get_binding_shape(0)
            output_shape = engine.get_binding_shape(1)

            input_size = trt.volume(input_shape)
            output_size = trt.volume(output_shape)

            # Allocate memory
            h_input = np.random.randn(input_size).astype(np.float32)
            h_output = np.empty(output_size, dtype=np.float32)

            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)

            # Copy input to device
            cuda.memcpy_htod(d_input, h_input)

            # Warm up
            for _ in range(10):
                context.execute_v2([int(d_input), int(d_output)])
            cuda.Context.synchronize()

            # Benchmark
            times = []
            _ = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            for _ in range(num_runs):
                start_time = time.perf_counter()
                context.execute_v2([int(d_input), int(d_output)])
                cuda.Context.synchronize()
                end_time = time.perf_counter()

                times.append(end_time - start_time)

            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Calculate statistics
            avg_time = np.mean(times) * 1000  # Convert to ms
            min_time = np.min(times) * 1000
            max_time = np.max(times) * 1000
            std_time = np.std(times) * 1000

            results = {
                "avg_inference_time_ms": avg_time,
                "min_inference_time_ms": min_time,
                "max_inference_time_ms": max_time,
                "std_inference_time_ms": std_time,
                "memory_usage_mb": memory_after,
                "num_runs": num_runs,
                "input_shape": input_shape,
                "output_shape": output_shape,
            }

            print("📊 Benchmark Results:")
            print(f"   Average: {avg_time:.2f}ms")
            print(f"   Min: {min_time:.2f}ms")
            print(f"   Max: {max_time:.2f}ms")
            print(f"   Std: {std_time:.2f}ms")
            print(f"   Memory: {memory_after:.1f}MB")

            return results

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert Whisper to TensorRT")
    parser.add_argument("--validate", action="store_true", help="Validate converted models")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark converted models")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="TensorRT precision mode",
    )

    args = parser.parse_args()

    converter = WhisperTensorRTConverter()

    print("🤖 Whisper TensorRT Conversion for Jetson")
    print("=" * 50)

    # Step 1: Export to ONNX
    if not converter.export_whisper_to_onnx():
        print("❌ ONNX export failed")
        return 1

    # Step 2: Convert to TensorRT
    encoder_onnx = MODEL_DIR / "whisper_tiny_encoder.onnx"
    decoder_onnx = MODEL_DIR / "whisper_tiny_decoder.onnx"

    encoder_engine = MODEL_DIR / f"whisper_tiny_encoder_{args.precision}.engine"
    decoder_engine = MODEL_DIR / f"whisper_tiny_decoder_{args.precision}.engine"

    if encoder_onnx.exists():
        if not converter.convert_onnx_to_tensorrt(encoder_onnx, encoder_engine, args.precision):
            print("❌ Encoder TensorRT conversion failed")
            return 1

    if decoder_onnx.exists():
        if not converter.convert_onnx_to_tensorrt(decoder_onnx, decoder_engine, args.precision):
            print("❌ Decoder TensorRT conversion failed")
            return 1

    # Step 3: Validation
    if args.validate:
        print("\n🧪 Validation Phase")
        if encoder_engine.exists():
            converter.validate_tensorrt_model(encoder_engine)
        if decoder_engine.exists():
            converter.validate_tensorrt_model(decoder_engine)

    # Step 4: Benchmarking
    if args.benchmark:
        print("\n🏃 Benchmark Phase")
        results = {}

        if encoder_engine.exists():
            print("Benchmarking encoder...")
            results["encoder"] = converter.benchmark_tensorrt_model(encoder_engine)

        if decoder_engine.exists():
            print("Benchmarking decoder...")
            results["decoder"] = converter.benchmark_tensorrt_model(decoder_engine)

        # Save benchmark results
        results_path = MODEL_DIR / f"tensorrt_benchmark_{args.precision}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"📋 Benchmark results saved: {results_path}")

    print("\n✅ Whisper TensorRT conversion completed!")
    print(f"📁 Models saved in: {MODEL_DIR}")

    return 0


if __name__ == "__main__":
    exit(main())
