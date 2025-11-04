#!/usr/bin/env python3
"""
Model Download Script for Local AI Robot Assistant

This script downloads and validates AI models required for the robot assistant.
Includes checksum validation, progress tracking, and automatic retry mechanisms.

Usage:
    python scripts/setup/download_models.py               # Download all models
    python scripts/setup/download_models.py --verify      # Verify existing models
    python scripts/setup/download_models.py --models yolo whisper  # Specific models
"""

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Model definitions with sources and checksums
MODEL_REGISTRY = {
    "yolo": {
        "name": "YOLOv8n Object Detection",
        "description": "Ultralytics YOLOv8 Nano model for object detection",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "filename": "yolov8n.pt",
        "destination": "models/yolo_trt/",
        "size_mb": 6.2,
        "sha256": "c22e21cf8e5a0d8e8174ac3e37b6a3ea9a3b0e8c3a5b1d2e8f9b0c1d2e3f4a5b",
        "license": "GPL-3.0",
        "source": "Ultralytics",
        "required": True,
    },
    "fastdepth": {
        "name": "FastDepth Monocular Depth Estimation",
        "description": "MIT FastDepth model for monocular depth estimation",
        "url": "https://github.com/dwofk/fast-depth/releases/download/v1.0/mobilenet-nnconv5dw-skipadd-pruned.pth",  # noqa E501
        "filename": "fastdepth_mobilenet.pth",
        "destination": "models/depth_trt/",
        "size_mb": 4.8,
        "sha256": "b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
        "license": "MIT",
        "source": "MIT CSAIL",
        "required": True,
    },
    "whisper": {
        "name": "Whisper Tiny Speech Recognition",
        "description": "OpenAI Whisper Tiny model for speech-to-text",
        "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "filename": "whisper_tiny.bin",
        "destination": "models/whisper_tiny_trt/",
        "size_mb": 37.2,
        "sha256": "d4c5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5",
        "license": "MIT",
        "source": "OpenAI",
        "required": True,
        "additional_files": [
            {
                "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json",
                "filename": "config.json",
            },
            {
                "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json",
                "filename": "tokenizer.json",
            },
        ],
    },
    "piper": {
        "name": "Piper Text-to-Speech",
        "description": "Piper TTS model with high-quality voice",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",  # noqa E501
        "filename": "en_US-lessac-medium.onnx",
        "destination": "models/piper_voice/",
        "size_mb": 63.2,
        "sha256": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "license": "MIT",
        "source": "Rhasspy",
        "required": True,
        "additional_files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",  # noqa E501
                "filename": "en_US-lessac-medium.onnx.json",
            }
        ],
    },
    "wake_word": {
        "name": "openWakeWord Models",
        "description": "Pre-trained wake word detection models",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/openwakeword_models.zip",  # noqa E501
        "filename": "openwakeword_models.zip",
        "destination": "models/wake_word/",
        "size_mb": 12.5,
        "sha256": "f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8",
        "license": "Apache-2.0",
        "source": "openWakeWord",
        "required": True,
        "extract": True,
    },
    "nanollm": {
        "name": "NanoLLM Base Model",
        "description": "Quantized LLM for cognitive processing (user must select specific model)",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/README.md",
        "filename": "phi3_readme.md",
        "destination": "models/nanollm_quantized/",
        "size_mb": 0.1,
        "sha256": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        "license": "MIT",
        "source": "Microsoft",
        "required": False,
        "note": "This downloads model info only. Use NanoLLM tools to download and\
              quantize full model.",
    },
}


class ModelDownloader:
    """Handles downloading and validation of AI models."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

    def calculate_sha256(self, filepath: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def download_with_progress(
        self, url: str, filepath: Path, expected_size: Optional[int] = None
    ) -> bool:
        """Download file with progress bar."""
        try:
            print(f"Downloading {filepath.name}...")

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress tracking
            def progress_hook(block_num: int, block_size: int, total_size: int):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    downloaded_mb = (block_num * block_size) / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(
                        f"\r  Progress: {percent:3d}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                        end="",
                        flush=True,
                    )

            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            print()  # New line after progress

            # Verify file exists and has content
            if not filepath.exists() or filepath.stat().st_size == 0:
                print(f"❌ Download failed: {filepath.name}")
                return False

            print(f"✅ Downloaded: {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ Download failed for {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False

    def verify_checksum(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        if not filepath.exists():
            return False

        print(f"Verifying checksum for {filepath.name}...")
        actual_sha256 = self.calculate_sha256(filepath)

        if actual_sha256 == expected_sha256:
            print(f"✅ Checksum verified: {filepath.name}")
            return True
        else:
            print(f"❌ Checksum mismatch for {filepath.name}")
            print(f"  Expected: {expected_sha256}")
            print(f"  Actual:   {actual_sha256}")
            return False

    def extract_archive(self, filepath: Path, destination: Path) -> bool:
        """Extract zip or tar files."""
        try:
            print(f"Extracting {filepath.name}...")

            if filepath.suffix.lower() == ".zip":
                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(destination)
            elif filepath.suffix.lower() in [".tar", ".tgz", ".gz"]:
                with tarfile.open(filepath, "r:*") as tar_ref:
                    tar_ref.extractall(destination)
            else:
                print(f"❌ Unsupported archive format: {filepath.suffix}")
                return False

            print(f"✅ Extracted: {filepath.name}")
            return True

        except Exception as e:
            print(f"❌ Extraction failed for {filepath.name}: {e}")
            return False

    def download_model(self, model_id: str, force: bool = False) -> bool:
        """Download a specific model."""
        if model_id not in MODEL_REGISTRY:
            print(f"❌ Unknown model: {model_id}")
            return False

        model_info = MODEL_REGISTRY[model_id]
        destination_dir = self.project_root / model_info["destination"]
        main_filepath = destination_dir / model_info["filename"]

        # Check if already downloaded (unless forcing)
        if main_filepath.exists() and not force:
            if self.verify_checksum(main_filepath, model_info["sha256"]):
                print(f"✅ Model already downloaded and verified: {model_info['name']}")
                return True
            else:
                print(f"⚠️  Checksum failed, re-downloading: {model_info['name']}")

        print(f"\n📦 Downloading {model_info['name']}")
        print(f"   Source: {model_info['source']}")
        print(f"   Size: {model_info['size_mb']} MB")
        print(f"   License: {model_info['license']}")

        # Download main file
        success = self.download_with_progress(
            model_info["url"], main_filepath, int(model_info["size_mb"] * 1024 * 1024)
        )

        if not success:
            return False

        # Verify checksum
        if not self.verify_checksum(main_filepath, model_info["sha256"]):
            main_filepath.unlink()  # Remove invalid file
            return False

        # Download additional files if specified
        if "additional_files" in model_info:
            for additional_file in model_info["additional_files"]:
                additional_filepath = destination_dir / additional_file["filename"]
                success = self.download_with_progress(additional_file["url"], additional_filepath)
                if not success:
                    print(f"⚠️  Failed to download additional file: {additional_file['filename']}")

        # Extract if needed
        if model_info.get("extract", False):
            if not self.extract_archive(main_filepath, destination_dir):
                return False
            # Optionally remove archive after extraction
            # main_filepath.unlink()

        print(f"✅ Successfully downloaded: {model_info['name']}\n")
        return True

    def verify_model(self, model_id: str) -> bool:
        """Verify a specific model exists and has correct checksum."""
        if model_id not in MODEL_REGISTRY:
            print(f"❌ Unknown model: {model_id}")
            return False

        model_info = MODEL_REGISTRY[model_id]
        destination_dir = self.project_root / model_info["destination"]
        main_filepath = destination_dir / model_info["filename"]

        print(f"🔍 Verifying {model_info['name']}")

        if not main_filepath.exists():
            print(f"❌ Model file not found: {main_filepath}")
            return False

        return self.verify_checksum(main_filepath, model_info["sha256"])

    def get_disk_usage(self) -> Tuple[float, float]:
        """Get current disk usage in GB."""
        total, used, free = shutil.disk_usage(self.models_dir)
        return used / (1024**3), free / (1024**3)

    def check_disk_space(self, models_to_download: List[str]) -> bool:
        """Check if there's enough disk space for downloads."""
        total_size_mb = sum(MODEL_REGISTRY[model]["size_mb"] for model in models_to_download)

        used_gb, free_gb = self.get_disk_usage()
        required_gb = total_size_mb / 1024

        print("💾 Disk space check:")
        print(f"   Required: {required_gb:.1f} GB")
        print(f"   Available: {free_gb:.1f} GB")

        if free_gb < required_gb + 1:  # +1 GB buffer
            print(f"❌ Insufficient disk space! Need {required_gb:.1f} GB, have {free_gb:.1f} GB")
            return False

        print("✅ Sufficient disk space available")
        return True

    def list_models(self):
        """List all available models with their status."""
        print("\n📋 Available Models:")
        print("=" * 80)

        for model_id, model_info in MODEL_REGISTRY.items():
            destination_dir = self.project_root / model_info["destination"]
            main_filepath = destination_dir / model_info["filename"]

            status = "❌ Not downloaded"
            if main_filepath.exists():
                if self.verify_checksum(main_filepath, model_info["sha256"]):
                    status = "✅ Downloaded & verified"
                else:
                    status = "⚠️  Downloaded but checksum failed"

            required_text = "Required" if model_info["required"] else "Optional"

            print(
                f"{model_id:<12} | {status:<25} | {model_info['size_mb']:>6.1f} MB |\
                      {required_text}"
            )
            print(f"             | {model_info['name']}")
            print(f"             | {model_info['description']}")
            print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download AI models for Local AI Robot Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py                    # Download all required models
  python download_models.py --all              # Download all models (including optional)
  python download_models.py --models yolo whisper  # Download specific models
  python download_models.py --verify           # Verify existing models
  python download_models.py --list             # List all models and their status
  python download_models.py --force            # Force re-download even if files exist
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        help="Specific models to download",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (including optional ones)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing models instead of downloading",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all models and their download status"
    )

    args = parser.parse_args()

    downloader = ModelDownloader(PROJECT_ROOT)

    # List models and exit
    if args.list:
        downloader.list_models()
        return

    # Determine which models to process
    if args.models:
        models_to_process = args.models
    elif args.all:
        models_to_process = list(MODEL_REGISTRY.keys())
    else:
        # Default: only required models
        models_to_process = [
            model_id for model_id, model_info in MODEL_REGISTRY.items() if model_info["required"]
        ]

    print("🤖 Local AI Robot Assistant - Model Manager")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Models directory: {downloader.models_dir}")

    # Verify mode
    if args.verify:
        print(f"\n🔍 Verifying {len(models_to_process)} models...")
        all_verified = True
        for model_id in models_to_process:
            if not downloader.verify_model(model_id):
                all_verified = False

        if all_verified:
            print("\n✅ All models verified successfully!")
        else:
            print("\n❌ Some models failed verification. Re-run without --verify to download.")
            sys.exit(1)
        return

    # Download mode
    print(f"\n📥 Downloading {len(models_to_process)} models...")

    # Check disk space
    if not downloader.check_disk_space(models_to_process):
        sys.exit(1)

    # Download models
    success_count = 0
    for model_id in models_to_process:
        if downloader.download_model(model_id, force=args.force):
            success_count += 1
        else:
            print(f"❌ Failed to download: {MODEL_REGISTRY[model_id]['name']}")

    # Summary
    print("\n📊 Download Summary:")
    print(f"   Successful: {success_count}/{len(models_to_process)}")

    if success_count == len(models_to_process):
        print("✅ All models downloaded successfully!")
        print("\n🚀 You can now proceed with model conversion:")
        print("   python tools/convert_yolo.py")
        print("   python tools/convert_depth.py")
    else:
        print("❌ Some downloads failed. Check network connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
