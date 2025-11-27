#!/usr/bin/env python3
"""
Native Depth Anything V2 Model Conversion Script

Converts Depth Anything V2 Small monocular depth estimation model to TensorRT FP16
for deployment on NVIDIA Jetson Orin Nano Super.

This uses the native model definition from the official repository for optimized
ONNX graph generation and superior TensorRT performance.

Features:
- Uses official Depth Anything V2 repository model definition
- Exports clean ONNX format without transformers overhead
- Converts to TensorRT FP16 engine with optimal settings
- Creates configuration files for deployment
- Performance: 25+ FPS at 518x518 resolution target
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DepthAnythingV2Converter:
    """
    Native Depth Anything V2 model converter for Jetson Orin Nano

    Uses official repository model definition for optimized ONNX/TensorRT conversion
    """

    def __init__(
        self,
        models_base_dir: Optional[Path] = None,
        input_size: int = 518,
    ):
        """
        Initialize converter

        Args:
            models_base_dir: Base directory for models (default: <repo_root>/models)
            input_size: Model input resolution (default: 518)
        """
        # Setup directory structure
        if models_base_dir is None:
            models_base_dir = Path(__file__).parent.parent.parent / "models"

        self.models_dir = Path(models_base_dir)
        self.depth_trt_path = self.models_dir / "depth_trt"
        self.repo_dir = self.depth_trt_path / "Depth-Anything-V2"
        self.weights_path = self.depth_trt_path / "depth_anything_v2_vits.pth"

        # Output paths
        self.onnx_path = self.depth_trt_path / f"depth_anything_v2_vits_{input_size}.onnx"
        self.engine_path = self.depth_trt_path / f"depth_anything_v2_vits_{input_size}.engine"

        # Model configuration
        self.input_size = input_size
        self.batch_size = 1

        # Create directories
        self.depth_trt_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized native converter")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Input size: {self.input_size}x{self.input_size}")

    def verify_prerequisites(self) -> bool:
        """Verify that repository and weights are available"""
        if not self.repo_dir.exists():
            logger.error(f"✗ Repository not found at: {self.repo_dir}")
            logger.error("  Run download_models.py first to clone the repository")
            return False

        if not self.weights_path.exists():
            logger.error(f"✗ Weights not found at: {self.weights_path}")
            logger.error("  Run download_models.py first to download weights")
            return False

        logger.info("✓ Prerequisites verified")
        return True

    def export_to_onnx(self) -> None:
        """Export native PyTorch model to ONNX format"""
        logger.info("Exporting native model to ONNX format")

        try:
            # Add repository to Python path
            sys.path.insert(0, str(self.repo_dir))

            # Import native model definition
            from depth_anything_v2.dpt import DepthAnythingV2

            logger.info("Loading native model...")

            # Native Model Configuration (Small/ViT-S)
            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
            }

            # Initialize model
            model = DepthAnythingV2(**model_configs["vits"])

            # Load weights
            state_dict = torch.load(self.weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            logger.info("✓ Native model loaded successfully")

            # Create dummy input
            dummy_input = torch.ones((self.batch_size, 3, self.input_size, self.input_size))

            # Test forward pass
            with torch.no_grad():
                test_output = model(dummy_input)
                logger.info(f"Model output shape: {test_output.shape}")

            # Export to ONNX
            logger.info(f"Exporting to ONNX (Input: {self.input_size}x{self.input_size})...")
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_path,
                opset_version=14,  # Opset 14 is stable for TensorRT 8.6/10.x
                export_params=True,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
                verbose=False,
            )

            logger.info(f"✓ ONNX model saved to: {self.onnx_path}")
            logger.info(f"  File size: {self.onnx_path.stat().st_size / (1024*1024):.1f} MB")

            # Verify ONNX model
            self._verify_onnx_model()

        except ImportError as e:
            logger.error(f"✗ Failed to import native model: {e}")
            logger.error("  Ensure the Depth-Anything-V2 repository is properly cloned")
            raise
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
                self.batch_size, 3, self.input_size, self.input_size
            ).astype(np.float32)

            # Run inference
            outputs = ort_session.run(None, {"input": test_input})

            logger.info("✓ ONNX inference successful")
            logger.info(f"  Output shape: {outputs[0].shape}")
            logger.info(f"  Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")

        except ImportError as e:
            logger.warning(f"⚠ Cannot verify ONNX model: {e}")
            logger.warning("  Install onnx and onnxruntime for verification")
        except Exception as e:
            logger.error(f"✗ ONNX verification failed: {e}")
            raise

    def convert_to_tensorrt(self, workspace_size_mb: int = 512) -> None:
        """
        Convert ONNX model to TensorRT engine on Jetson

        Args:
            workspace_size_mb: TensorRT workspace memory pool size in MB
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

        # TensorRT conversion command optimized for Jetson Orin Nano with native model
        cmd = [
            str(trtexec_path),
            f"--onnx={self.onnx_path}",
            f"--saveEngine={self.engine_path}",
            "--fp16",  # Crucial for Orin performance
            "--best",  # Try all tactical optimizations
            "--avgTiming=10",
            "--useCudaGraph",  # Optimizes kernel launching
            "--infStreams=1",  # Minimize overhead
            "--verbose",
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
            "model_name": "depth_anything_v2_vits",
            "model_type": "depth_estimation",
            "framework": "tensorrt",
            "version": "v2_native",
            "model_variant": "small_vits",
            "input_shape": [self.batch_size, 3, self.input_size, self.input_size],
            "output_shape": [self.batch_size, self.input_size, self.input_size],
            "preprocessing": {
                "resize": [self.input_size, self.input_size],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "color_format": "RGB",
            },
            "performance": {
                "target_fps": 25,
                "max_latency_ms": 40,
                "memory_limit_mb": 800,
            },
            "files": {
                "weights": "depth_anything_v2_vits.pth",
                "onnx": str(self.onnx_path.name),
                "tensorrt": str(self.engine_path.name),
                "repository": "Depth-Anything-V2",
            },
            "notes": "Native model definition for optimized ONNX graph",
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
            "size": {"height": self.input_size, "width": self.input_size},
            "size_divisor": 14,  # Patch size for vision transformer
        }

        preprocessor_path = config_dir / "preprocessor_config.json"
        with open(preprocessor_path, "w") as f:
            json.dump(preprocessor_config, f, indent=2)

        logger.info(f"✓ Preprocessor configuration saved to: {preprocessor_path}")

    def full_conversion_pipeline(
        self, skip_tensorrt: bool = False, workspace_size_mb: int = 512
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
            # Step 1: Verify prerequisites
            logger.info("\n[1/4] Verifying prerequisites...")
            if not self.verify_prerequisites():
                logger.error("\n✗ Prerequisites check failed")
                logger.error(
                    "  Run: python scripts/setup/download_models.py --models depth_anything_v2"
                )
                raise RuntimeError("Missing prerequisites")

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
            logger.info(f"  - Repository: {self.repo_dir}")
            logger.info(f"  - Model weights: {self.weights_path}")
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
        description="Convert native Depth Anything V2 Small model for Jetson deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Model input resolution (default: 518)",
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
        help="TensorRT workspace memory pool size in MB",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export to ONNX, skip TensorRT conversion",
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 70)
        logger.info("Native Depth Anything V2 Model Converter")
        logger.info("=" * 70)
        logger.info("Using native model definition for optimal ONNX graph")
        logger.info(f"Input size: {args.input_size}x{args.input_size}")

        # Initialize converter
        converter = DepthAnythingV2Converter(
            models_base_dir=args.models_dir, input_size=args.input_size
        )

        if args.onnx_only:
            # Only run ONNX export
            if converter.verify_prerequisites():
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
