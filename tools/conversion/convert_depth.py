#!/usr/bin/env python3
"""
Depth Anything V2 Model Conversion Script

Converts Depth Anything V2 Small monocular depth estimation model to TensorRT FP16
for deployment on NVIDIA Jetson Orin Nano Super.

Features:
- Downloads model from HuggingFace Hub
- Exports to ONNX format with verification
- Converts to TensorRT FP16 engine
- Creates configuration files for deployment
- Performance: 20+ FPS at 518x518 resolution target
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DepthAnythingV2Converter:
    """
    Depth Anything V2 model converter for Jetson Orin Nano

    Handles download, ONNX export, and TensorRT conversion of Depth Anything V2 Small
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        models_base_dir: Optional[Path] = None,
    ):
        """
        Initialize converter

        Args:
            model_name: HuggingFace model name/path
            models_base_dir: Base directory for models (default: <repo_root>/models)
        """
        self.model_name = model_name

        # Use CPU for export to avoid memory issues during conversion
        self.device = "cpu"

        # Setup directory structure
        if models_base_dir is None:
            models_base_dir = Path(__file__).parent.parent.parent / "models"

        self.models_dir = Path(models_base_dir)
        # Use depth_trt as the single directory for all model files
        self.local_model_path = self.models_dir / "depth_trt"
        self.depth_trt_path = self.models_dir / "depth_trt"
        self.onnx_path = self.depth_trt_path / "depth_anything_v2_small.onnx"
        self.engine_path = self.depth_trt_path / "depth_anything_v2_small.trt"

        # Model configuration
        self.input_size = (518, 518)  # Standard size for Depth Anything V2
        self.batch_size = 1

        # Create directories
        self.depth_trt_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized converter for {model_name}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Device: {self.device}")

    def download_model(self) -> None:
        """Download Depth Anything V2 Small model from HuggingFace"""
        logger.info(f"Downloading model: {self.model_name}")

        try:
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.local_model_path,
                local_dir_use_symlinks=False,
                ignore_patterns=["*.git*", "*.md", "*.txt"],
            )
            logger.info(f"✓ Model downloaded to: {self.local_model_path}")

        except Exception as e:
            logger.error(f"✗ Failed to download model: {e}")
            raise

    def export_to_onnx(self) -> None:
        """Export PyTorch model to ONNX format"""
        logger.info("Exporting model to ONNX format")

        try:
            # Load model and processor
            logger.info("Loading PyTorch model...")
            model = AutoModelForDepthEstimation.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float32,  # Use float32 for ONNX export
            )
            _ = AutoImageProcessor.from_pretrained(self.local_model_path)

            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            logger.info("✓ Model loaded successfully")

            # Create dummy input matching expected dimensions
            dummy_input = torch.randn(
                self.batch_size,
                3,
                self.input_size[0],
                self.input_size[1],
                dtype=torch.float32,
                device=self.device,
            )

            logger.info(f"Dummy input shape: {dummy_input.shape}")

            # Test forward pass to verify model works
            with torch.no_grad():
                test_output = model(pixel_values=dummy_input)
                logger.info(f"Model output shape: {test_output.predicted_depth.shape}")

            # Export to ONNX with dynamic shapes
            logger.info("Exporting to ONNX...")
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=17,  # Use latest stable opset
                do_constant_folding=True,
                input_names=["pixel_values"],
                output_names=["predicted_depth"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
                    "predicted_depth": {0: "batch_size", 1: "height", 2: "width"},
                },
                verbose=False,
            )

            logger.info(f"✓ ONNX model saved to: {self.onnx_path}")
            logger.info(f"  File size: {self.onnx_path.stat().st_size / (1024*1024):.1f} MB")

            # Verify ONNX model
            self._verify_onnx_model()

        except Exception as e:
            logger.error(f"✗ ONNX export failed: {e}")
            raise

    def _verify_onnx_model(self) -> None:
        """Verify ONNX model can be loaded and run"""
        try:
            import onnx
            import onnxruntime as ort

            logger.info("Verifying ONNX model...")

            # Load and check ONNX model
            onnx_model = onnx.load(str(self.onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model structure valid")

            # Test inference with ONNX Runtime
            ort_session = ort.InferenceSession(
                str(self.onnx_path),
                providers=["CPUExecutionProvider"],
            )

            # Create test input
            test_input = np.random.randn(
                self.batch_size, 3, self.input_size[0], self.input_size[1]
            ).astype(np.float32)

            # Run inference
            outputs = ort_session.run(None, {"pixel_values": test_input})

            logger.info("✓ ONNX inference successful")
            logger.info(f"  Output shape: {outputs[0].shape}")
            logger.info(f"  Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")

        except ImportError as e:
            logger.warning(f"⚠ Cannot verify ONNX model: {e}")
            logger.warning("  Install onnx and onnxruntime for verification")
        except Exception as e:
            logger.error(f"✗ ONNX verification failed: {e}")
            raise

    def convert_to_tensorrt(self, workspace_size_mb: int = 256) -> None:
        """
        Convert ONNX model to TensorRT engine on Jetson

        Args:
            workspace_size_mb: TensorRT workspace memory pool size in MB \
                (uses --memPoolSize=workspace:N)
        """
        logger.info("Converting ONNX to TensorRT engine")

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        # Check if TensorRT is available
        trtexec_path = Path("/usr/src/tensorrt/bin/trtexec")
        if not trtexec_path.exists():
            raise RuntimeError(
                "trtexec not found. Ensure TensorRT is installed on Jetson.\n"
                "This step must be run on the Jetson device."
            )

        # TensorRT conversion command optimized for Jetson Orin Nano
        # Use fixed shapes to reduce memory consumption during optimization
        cmd = [
            str(trtexec_path),
            f"--onnx={self.onnx_path}",
            f"--saveEngine={self.engine_path}",
            "--fp16",  # Use FP16 for memory efficiency and speed
            f"--memPoolSize=workspace:{workspace_size_mb}",  # Reduced workspace memory
            # Memory optimization flags for Jetson
            "--builderOptimizationLevel=3",  # Use aggressive optimization
            "--avgTiming=1",  # Reduce timing iterations to save memory
            # Use fixed shape instead of dynamic to reduce memory usage
            f"--shapes=pixel_values:{self.batch_size}x3x{self.input_size[0]}x{self.input_size[1]}",
            "--verbose",
            "--useSpinWait",
            "--noDataTransfers",
            "--skipInference",  # Skip inference to save time and memory
        ]

        try:
            logger.info("Running TensorRT conversion (this may take 5-15 minutes)...")
            logger.info(f"Command: {' '.join(cmd)}")

            # Run conversion
            result = subprocess.run(
                cmd,
                cwd=self.engine_path.parent,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.returncode != 0:
                logger.error("✗ TensorRT conversion failed:")
                logger.error(f"STDOUT:\n{result.stdout}")
                logger.error(f"STDERR:\n{result.stderr}")
                raise RuntimeError("TensorRT conversion failed")

            logger.info(f"✓ TensorRT engine created: {self.engine_path}")
            logger.info(f"  Engine size: {self.engine_path.stat().st_size / (1024*1024):.1f} MB")

            # Parse performance metrics from output
            self._parse_trtexec_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("✗ TensorRT conversion timed out after 30 minutes")
            raise
        except Exception as e:
            logger.error(f"✗ TensorRT conversion error: {e}")
            raise

    def _parse_trtexec_output(self, output: str) -> None:
        """Parse and log performance metrics from trtexec output"""
        lines = output.split("\n")
        for line in lines:
            if "mean" in line.lower() and "ms" in line.lower():
                logger.info(f"  Performance: {line.strip()}")
            elif "throughput" in line.lower():
                logger.info(f"  {line.strip()}")

    def create_config_files(self) -> None:
        """Create configuration files for the depth model"""
        config_dir = self.depth_trt_path

        # Model configuration
        model_config = {
            "model_name": "depth_anything_v2_small",
            "model_type": "depth_estimation",
            "framework": "tensorrt",
            "version": "v2",
            "input_shape": [self.batch_size, 3, self.input_size[0], self.input_size[1]],
            "output_shape": [self.batch_size, self.input_size[0], self.input_size[1]],
            "preprocessing": {
                "resize": self.input_size,
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "color_format": "RGB",
            },
            "performance": {
                "target_fps": 20,
                "max_latency_ms": 50,
                "memory_limit_mb": 1000,
            },
            "files": {
                "onnx": str(self.onnx_path.name),
                "tensorrt": str(self.engine_path.name),
            },
        }

        config_path = config_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"✓ Model configuration saved to: {config_path}")

        # Preprocessor configuration (compatible with transformers)
        preprocessor_config = {
            "do_normalize": True,
            "do_resize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "keep_aspect_ratio": False,
            "resample": 3,  # PIL.Image.BICUBIC
            "size": {"height": self.input_size[0], "width": self.input_size[1]},
            "size_divisor": 14,  # Patch size for vision transformer
        }

        preprocessor_path = config_dir / "preprocessor_config.json"
        with open(preprocessor_path, "w") as f:
            json.dump(preprocessor_config, f, indent=2)

        logger.info(f"✓ Preprocessor configuration saved to: {preprocessor_path}")

    def full_conversion_pipeline(
        self, skip_tensorrt: bool = False, workspace_size_mb: int = 256
    ) -> None:
        """
        Run the complete model conversion pipeline

        Args:
            skip_tensorrt: If True, only export to ONNX
            workspace_size_mb: TensorRT workspace memory pool size in MB
        """
        logger.info("=" * 70)
        logger.info("Starting Depth Anything V2 Conversion Pipeline")
        logger.info("=" * 70)

        start_time = time.time()

        try:
            # Step 1: Download model
            if not (self.local_model_path / "config.json").exists():
                logger.info("\n[1/4] Downloading model from HuggingFace...")
                self.download_model()
            else:
                logger.info(f"\n[1/4] Model already exists at: {self.local_model_path}")

            # Step 2: Export to ONNX
            if not self.onnx_path.exists():
                logger.info("\n[2/4] Exporting to ONNX...")
                self.export_to_onnx()
            else:
                logger.info(f"\n[2/4] ONNX model already exists at: {self.onnx_path}")

            # Step 3: Convert to TensorRT (if on Jetson and requested)
            if skip_tensorrt:
                logger.info("\n[3/4] Skipping TensorRT conversion (--onnx-only flag)")
            elif self.engine_path.exists():
                logger.info(f"\n[3/4] TensorRT engine already exists at: {self.engine_path}")
            else:
                logger.info("\n[3/4] Converting to TensorRT...")
                if not torch.cuda.is_available():
                    logger.warning("⚠ CUDA not available - TensorRT conversion skipped")
                    logger.warning("  Run this script on the Jetson device for TensorRT conversion")
                else:
                    self.convert_to_tensorrt(workspace_size_mb)

            # Step 4: Create configuration files
            logger.info("\n[4/4] Creating configuration files...")
            self.create_config_files()

            elapsed_time = time.time() - start_time

            logger.info("\n" + "=" * 70)
            logger.info(f"✓ Conversion pipeline completed in {elapsed_time:.2f} seconds")
            logger.info("=" * 70)
            logger.info("\nGenerated files:")
            logger.info(f"  - Model weights: {self.local_model_path}")
            logger.info(f"  - ONNX model: {self.onnx_path}")
            if self.engine_path.exists():
                logger.info(f"  - TensorRT engine: {self.engine_path}")
            logger.info(f"  - Config files: {self.depth_trt_path}")

        except Exception as e:
            logger.error(f"\n✗ Conversion pipeline failed: {e}")
            raise


def main():
    """Main conversion script entry point"""
    parser = argparse.ArgumentParser(
        description="Convert Depth Anything V2 Small model for Jetson deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Base directory for models (default: <repo_root>/models)",
    )
    parser.add_argument(
        "--workspace-size",
        type=int,
        default=256,
        help="TensorRT workspace memory pool size in MB (for --memPoolSize=workspace:N, \
            default optimized for Jetson Orin Nano)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download if already exists",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export to ONNX, skip TensorRT conversion",
    )

    args = parser.parse_args()

    try:
        # Initialize converter
        converter = DepthAnythingV2Converter(args.model_name, models_base_dir=args.models_dir)

        if args.onnx_only:
            # Only run ONNX export
            if not args.skip_download:
                converter.download_model()
            converter.export_to_onnx()
            converter.create_config_files()
            logger.info("✓ ONNX export completed successfully!")
        else:
            # Run full pipeline with workspace size argument
            converter.full_conversion_pipeline(
                skip_tensorrt=False, workspace_size_mb=args.workspace_size
            )
            logger.info("✓ Model conversion completed successfully!")

        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠ Conversion interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n✗ Conversion failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
