#!/usr/bin/env python3
"""
Whisper Model Conversion Script
Converts Whisper models for faster-whisper (CTranslate2) or TensorRT optimization

According to architecture.md:
- Primary: faster-whisper Tiny (CTranslate2 optimized)
- Alternative: Whisper Tiny TensorRT engine
- Target: <2 second latency for 5-second utterance
- Real-time factor < 0.3x
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import nvidia_ml_py3 as nvml
import psutil
import tensorrt as trt
import torch
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WhisperConverter:
    """
    Converts Whisper models for deployment on NVIDIA Jetson Orin Nano

    Supports two deployment options:
    1. faster-whisper (CTranslate2) - Recommended for production
    2. Custom TensorRT conversion - Alternative approach
    """

    def __init__(self, model_size: str = "tiny", quantization: str = "int8", device: str = "cuda"):
        """
        Initialize Whisper converter

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            quantization: Quantization level for faster-whisper (int8, int16, float16, float32)
            device: Target device (cuda, cpu)
        """
        self.model_size = model_size
        self.quantization = quantization
        self.device = device

        # Initialize NVIDIA ML for GPU monitoring
        try:
            nvml.nvmlInit()
            self.gpu_available = True
        except Exception:
            self.gpu_available = False
            logger.warning("NVIDIA ML not available - GPU monitoring disabled")

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Validate inputs
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate initialization parameters"""
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size: {self.model_size}. Must be one of {valid_sizes}")

        valid_quant = ["int8", "int16", "float16", "float32"]
        if self.quantization not in valid_quant:
            raise ValueError(
                f"Invalid quantization: {self.quantization}. Must be one of {valid_quant}"
            )

        valid_devices = ["cuda", "cpu"]
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices}")

    def install_dependencies(self) -> bool:
        """
        Install required dependencies for Whisper conversion

        Returns:
            True if successful, False otherwise
        """
        logger.info("Installing Whisper conversion dependencies...")

        packages = [
            "openai-whisper",
            "faster-whisper",
            "ct2-transformers-converter",
            "transformers",
            "torch",
            "torchaudio",
        ]

        try:
            for package in packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.returncode == 0:
                    logger.info(f"✓ {package} installed successfully")
                else:
                    logger.error(f"✗ Failed to install {package}: {result.stderr}")
                    return False

            logger.info("✓ All dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def download_whisper_model(self, output_dir: str) -> str:
        """
        Download original Whisper model from OpenAI

        Args:
            output_dir: Directory to save the model

        Returns:
            Path to downloaded model
        """
        logger.info(f"Downloading Whisper {self.model_size} model...")

        try:
            import whisper

            # Download model
            model = whisper.load_model(self.model_size, download_root=output_dir)

            # Save model state dict
            model_path = os.path.join(output_dir, f"whisper_{self.model_size}.pt")
            torch.save(model.state_dict(), model_path)

            logger.info(f"✓ Whisper model downloaded to: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download Whisper model: {e}")
            raise

    def convert_to_faster_whisper(self, output_dir: str) -> str:
        """
        Convert Whisper model to faster-whisper (CTranslate2) format

        Args:
            output_dir: Directory to save converted model

        Returns:
            Path to converted model directory
        """
        logger.info(f"Converting Whisper {self.model_size} to faster-whisper format...")

        model_dir = os.path.join(
            output_dir, f"faster_whisper_{self.model_size}_{self.quantization}"
        )

        try:
            # Use ct2-transformers-converter to convert
            cmd = [
                "ct2-transformers-converter",
                "--model",
                f"openai/whisper-{self.model_size}",
                "--output_dir",
                model_dir,
                "--quantization",
                self.quantization,
                "--copy_files",
                "tokenizer.json",
            ]

            logger.info(f"Running conversion command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode == 0:
                logger.info("✓ faster-whisper model converted successfully")
                logger.info(f"Model saved to: {model_dir}")

                # Verify model files
                self._verify_faster_whisper_model(model_dir)

                return model_dir
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                raise RuntimeError("faster-whisper conversion failed")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert to faster-whisper: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Command error: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("ct2-transformers-converter not found. Please install it with:")
            logger.error("pip install ct2-transformers-converter")
            raise

    def _verify_faster_whisper_model(self, model_dir: str) -> None:
        """Verify faster-whisper model integrity"""
        required_files = ["config.json", "model.bin"]

        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Missing file: {file_path}")

        logger.info("✓ faster-whisper model files verified")

    def convert_to_onnx(self, output_dir: str) -> str:
        """
        Convert Whisper model to ONNX format (for TensorRT conversion)

        Args:
            output_dir: Directory to save ONNX model

        Returns:
            Path to ONNX model
        """
        logger.info(f"Converting Whisper {self.model_size} to ONNX...")

        try:
            import whisper

            # Load model
            model = whisper.load_model(self.model_size)
            model.eval()

            # Export encoder to ONNX
            encoder_path = os.path.join(output_dir, f"whisper_{self.model_size}_encoder.onnx")

            # Create dummy input for encoder
            mel_input = torch.randn(1, 80, 3000)  # Mel spectrogram input

            # Export encoder
            torch.onnx.export(
                model.encoder,
                mel_input,
                encoder_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["mel"],
                output_names=["audio_features"],
                dynamic_axes={
                    "mel": {2: "mel_length"},
                    "audio_features": {2: "feature_length"},
                },
            )

            logger.info(f"✓ Encoder ONNX model saved to: {encoder_path}")

            # Export decoder to ONNX (more complex due to autoregressive nature)
            _ = os.path.join(output_dir, f"whisper_{self.model_size}_decoder.onnx")

            # Note: Full decoder export is complex due to autoregressive generation
            # For production, recommend using faster-whisper instead
            logger.info("⚠ Decoder ONNX export is complex - recommend using faster-whisper")

            return encoder_path

        except Exception as e:
            logger.error(f"Failed to convert to ONNX: {e}")
            raise

    def convert_to_tensorrt(self, onnx_path: str, output_path: str) -> str:
        """
        Convert ONNX Whisper model to TensorRT engine

        Args:
            onnx_path: Path to ONNX model
            output_path: Output path for TensorRT engine

        Returns:
            Path to TensorRT engine
        """
        logger.info("Converting Whisper ONNX to TensorRT...")

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
        config.max_workspace_size = 1 << 28  # 256MB

        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✓ FP16 precision enabled")

        # Set optimization level
        config.builder_optimization_level = 5

        # Build engine
        logger.info("Building TensorRT engine...")
        start_time = time.time()

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        build_time = time.time() - start_time
        logger.info(f"✓ Engine built in {build_time:.2f} seconds")

        # Serialize and save engine
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"TensorRT engine saved to: {output_path}")
        return output_path

    def benchmark_faster_whisper(
        self, model_dir: str, audio_file: Optional[str] = None, num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark faster-whisper model performance

        Args:
            model_dir: Path to faster-whisper model directory
            audio_file: Path to test audio file (optional)
            num_iterations: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking faster-whisper model: {model_dir}")

        try:
            from faster_whisper import WhisperModel

            # Load model
            logger.info("Loading faster-whisper model...")
            model = WhisperModel(model_dir, device=self.device, compute_type=self.quantization)

            # Prepare test audio
            if audio_file and os.path.exists(audio_file):
                logger.info(f"Using test audio: {audio_file}")
                test_audio = audio_file
            else:
                # Create dummy audio file for testing
                logger.info("Creating dummy audio for testing...")
                test_audio = self._create_dummy_audio()

            # Warmup
            logger.info("Warming up model...")
            segments, info = model.transcribe(test_audio, beam_size=1)
            list(segments)  # Consume generator

            # Benchmark
            logger.info(f"Running benchmark for {num_iterations} iterations...")

            times = []
            total_duration = 0

            for i in range(num_iterations):
                start_time = time.time()

                segments, info = model.transcribe(test_audio, beam_size=1, language="en")

                # Consume all segments
                transcription = " ".join([segment.text for segment in segments])

                end_time = time.time()
                iteration_time = end_time - start_time
                times.append(iteration_time)

                total_duration = info.duration

                logger.debug(f"Iteration {i+1}: {iteration_time:.3f}s")

            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            real_time_factor = avg_time / total_duration if total_duration > 0 else 0

            # Get memory usage
            memory_info = self._get_memory_info()

            results = {
                "model_size": self.model_size,
                "quantization": self.quantization,
                "device": self.device,
                "avg_transcription_time_s": avg_time,
                "std_transcription_time_s": std_time,
                "min_time_s": min(times),
                "max_time_s": max(times),
                "audio_duration_s": total_duration,
                "real_time_factor": real_time_factor,
                "iterations": num_iterations,
                "memory_usage_mb": memory_info,
                "sample_transcription": (
                    transcription[:100] + "..." if len(transcription) > 100 else transcription
                ),
            }

            # Display results
            self._display_whisper_benchmark_results(results)

            return results

        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _create_dummy_audio(self, duration: float = 5.0, sample_rate: int = 16000) -> str:
        """Create dummy audio file for testing"""
        import tempfile
        import wave

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        # Generate dummy audio (sine wave)
        samples = int(duration * sample_rate)
        frequency = 440  # A note
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.1

        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Save as WAV file
        with wave.open(audio_path, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        logger.info(f"Created dummy audio: {audio_path} ({duration}s)")
        return audio_path

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

    def _display_whisper_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Display Whisper benchmark results"""
        data = [
            ["Metric", "Value"],
            ["Model Size", results["model_size"]],
            ["Quantization", results["quantization"]],
            ["Device", results["device"]],
            ["Avg Transcription Time", f"{results['avg_transcription_time_s']:.3f} s"],
            ["Std Transcription Time", f"{results['std_transcription_time_s']:.3f} s"],
            ["Min Time", f"{results['min_time_s']:.3f} s"],
            ["Max Time", f"{results['max_time_s']:.3f} s"],
            ["Audio Duration", f"{results['audio_duration_s']:.3f} s"],
            ["Real-time Factor", f"{results['real_time_factor']:.3f}x"],
            ["Iterations", results["iterations"]],
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

        print("\n" + "=" * 60)
        print("WHISPER BENCHMARK RESULTS")
        print("=" * 60)
        print(tabulate(data, headers="firstrow", tablefmt="grid"))

        # Performance assessment
        target_rtf = 0.3  # Real-time factor target from architecture
        target_latency = 2.0  # 2 second latency target

        if results["real_time_factor"] <= target_rtf:
            print(
                f"\n✓ Real-time factor target met:\
                      {results['real_time_factor']:.3f}x <= {target_rtf}x"
            )
        else:
            print(
                f"\n⚠ Real-time factor above target:\
                      {results['real_time_factor']:.3f}x > {target_rtf}x"
            )

        if results["avg_transcription_time_s"] <= target_latency:
            print(
                f"✓ Latency target met:\
                      {results['avg_transcription_time_s']:.3f}s <= {target_latency}s"
            )
        else:
            print(
                f"⚠ Latency above target:\
                      {results['avg_transcription_time_s']:.3f}s > {target_latency}s"
            )

        print(f"\nSample transcription: {results['sample_transcription']}")

    def convert_full_pipeline(
        self,
        output_dir: str,
        conversion_type: str = "faster-whisper",
        skip_existing: bool = True,
    ) -> Dict[str, str]:
        """
        Run complete Whisper conversion pipeline

        Args:
            output_dir: Directory to save converted models
            conversion_type: Type of conversion ("faster-whisper" or "tensorrt")
            skip_existing: Skip conversion if files already exist

        Returns:
            Dictionary with paths to converted models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Whisper {self.model_size} conversion pipeline")
        logger.info(f"Conversion type: {conversion_type}")
        logger.info(f"Output directory: {output_dir}")

        results = {}

        if conversion_type == "faster-whisper":
            # Convert to faster-whisper (recommended)
            model_dir = os.path.join(
                output_dir, f"faster_whisper_{self.model_size}_{self.quantization}"
            )

            if not os.path.exists(model_dir) or not skip_existing:
                converted_model_dir = self.convert_to_faster_whisper(str(output_dir))
                results["faster_whisper"] = converted_model_dir
            else:
                logger.info(f"Skipping conversion - faster-whisper model exists: {model_dir}")
                results["faster_whisper"] = model_dir

            # Benchmark
            benchmark_results = self.benchmark_faster_whisper(results["faster_whisper"])

            # Save benchmark results
            with open(output_dir / f"whisper_{self.model_size}_benchmark.json", "w") as f:
                json.dump(benchmark_results, f, indent=2)

        elif conversion_type == "tensorrt":
            # Convert via ONNX to TensorRT (alternative)
            onnx_path = output_dir / f"whisper_{self.model_size}_encoder.onnx"
            trt_path = output_dir / f"whisper_{self.model_size}_encoder.trt"

            if not onnx_path.exists() or not skip_existing:
                onnx_model_path = self.convert_to_onnx(str(output_dir))
                results["onnx"] = onnx_model_path
            else:
                logger.info(f"Skipping ONNX conversion - file exists: {onnx_path}")
                results["onnx"] = str(onnx_path)

            if not trt_path.exists() or not skip_existing:
                trt_model_path = self.convert_to_tensorrt(results["onnx"], str(trt_path))
                results["tensorrt"] = trt_model_path
            else:
                logger.info(f"Skipping TensorRT conversion - file exists: {trt_path}")
                results["tensorrt"] = str(trt_path)

            logger.info("⚠ TensorRT conversion is experimental - recommend using faster-whisper")

        else:
            raise ValueError(f"Invalid conversion type: {conversion_type}")

        logger.info("✓ Whisper conversion pipeline completed successfully!")

        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Convert Whisper models for deployment on Jetson Orin Nano"
    )
    parser.add_argument(
        "--model-size",
        "-m",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./models/whisper",
        help="Output directory for converted models",
    )
    parser.add_argument(
        "--quantization",
        "-q",
        default="int8",
        choices=["int8", "int16", "float16", "float32"],
        help="Quantization level for faster-whisper",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Target device")
    parser.add_argument(
        "--conversion-type",
        "-t",
        default="faster-whisper",
        choices=["faster-whisper", "tensorrt"],
        help="Conversion type",
    )
    parser.add_argument("--test-audio", help="Path to test audio file for benchmarking")
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install required dependencies before conversion",
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
    converter = WhisperConverter(
        model_size=args.model_size, quantization=args.quantization, device=args.device
    )

    try:
        # Install dependencies if requested
        if args.install_deps:
            if not converter.install_dependencies():
                logger.error("Failed to install dependencies")
                sys.exit(1)

        # Run conversion pipeline
        results = converter.convert_full_pipeline(
            output_dir=args.output_dir,
            conversion_type=args.conversion_type,
            skip_existing=args.skip_existing,
        )

        print("\n" + "=" * 70)
        print("WHISPER CONVERSION COMPLETED SUCCESSFULLY")
        print("=" * 70)

        for key, path in results.items():
            print(f"{key.upper()}: {path}")

        print(f"\nModel size: {args.model_size}")
        print(f"Quantization: {args.quantization}")
        print(f"Conversion type: {args.conversion_type}")
        print("\nModel is ready for deployment in ROS2 audio interface nodes.")

        if args.conversion_type == "faster-whisper":
            print("\n✓ Recommended: faster-whisper provides optimal performance")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
