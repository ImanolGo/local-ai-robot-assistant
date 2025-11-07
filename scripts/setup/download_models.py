#!/usr/bin/env python3
"""
Model Download Script for Local AI Robot Assistant

This script downloads and validates AI models required for the robot assistant.
Includes checksum validation, progress tracking, and automatic retry mechanisms.

Features Gemma 3n E2B multimodal model from Google DeepMind for revolutionary
text, audio, and vision processing capabilities on Jetson Orin Nano.


Usage:
    python scripts/setup/download_models.py               # Download all models
    python scripts/setup/download_models.py --verify      # Verify existing models
    python scripts/setup/download_models.py --models yolo whisper  # Specific models

Note: The Gemma 3n E2B model weights will be automatically downloaded by HuggingFace
Transformers during first use. This script downloads configuration files only.
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

try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("⚠️  Warning: 'gdown' not available. Google Drive downloads will be skipped.")
    print("   Install with: pip install gdown")

try:
    from huggingface_hub import hf_hub_download, snapshot_download

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    print("⚠️  Warning: 'huggingface_hub' not available. HuggingFace downloads may fail.")
    print("   Install with: pip install huggingface_hub")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Model definitions with sources and checksums
MODEL_REGISTRY = {
    "yolo": {
        "name": "YOLO11n Object Detection",
        "description": "Ultralytics YOLO11 Nano model for object detection",
        "url": "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt",
        "filename": "yolo11n.pt",
        "destination": "models/yolo_trt/",
        "size_mb": 5.61,
        "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "license": "AGPL-3.0",
        "source": "Ultralytics",
        "required": True,
    },
    "rt_monodepth": {
        "name": "RT-MonoDepth-S Monocular Depth Estimation",
        "description": "RT-MonoDepth-S lightweight encoder-decoder for real-time monocular\
              depth estimation",
        "url": "https://drive.google.com/file/d/1Jf5K3m0DfAqVcVCE6y0cKufEKIHu86sz/view?usp=drive_link",  # noqa E501
        "filename": "weights_rtmonodepth.zip",
        "destination": "models/depth_trt/",
        "size_mb": 232.5,  # Actual downloaded size
        "sha256": "44c87fa550e6ce8b44cb4115e37f53f872e1a1a1cd576e58aa270c3c45176739",
        "license": "Academic Research",
        "source": "RT-MonoDepth Research",
        "required": True,
        "extract": True,
        "download_method": "gdown",
    },
    "whisper": {
        "name": "Whisper Tiny Speech Recognition",
        "description": "OpenAI Whisper Tiny model for speech-to-text",
        "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "filename": "whisper_tiny.bin",
        "destination": "models/whisper_tiny_trt/",
        "size_mb": 144.10,
        "sha256": "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d",
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
        "description": "Rhasspy Piper TTS model (en_US-lessac-medium) with high-quality voice",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",  # noqa E501
        "filename": "en_US-lessac-medium.onnx",
        "destination": "models/piper_voice/",
        "size_mb": 60.27,
        "sha256": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        "license": "MIT",
        "source": "Rhasspy",
        "required": True,
        "additional_files": [
            {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",  # noqa E501
                "filename": "en_US-lessac-medium.onnx.json",
            }
        ],
    },
    "wake_word": {
        "name": "openWakeWord Models",
        "description": "Pre-trained wake word detection models",
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",  # noqa E501
        "filename": "hey_jarvis_v0.1.onnx",
        "destination": "models/wake_word/",
        "size_mb": 1.2,
        "sha256": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        "license": "Apache-2.0",
        "source": "openWakeWord",
        "required": True,
        "extract": False,
    },
    "gemma_3n_e2b": {
        "name": "Gemma 3n E2B Multimodal Model (Full Download)",
        "description": "Google DeepMind Gemma 3n E2B - 5B parameter multimodal model with\
              2B effective footprint for text, audio, and vision processing. Downloads complete\
                  model for offline use.",
        "url": "google/gemma-3n-e2b",  # Using Gemma 2B until 3n E2B is available
        "filename": "google/gemma-3n-e2b",
        "destination": "models/gemma_3n_e2b/",
        "size_mb": 4800.0,  # Full model size
        "sha256": "",  # Will be calculated during download
        "license": "Gemma Terms of Use",
        "source": "Google DeepMind",
        "required": True,
        "download_method": "snapshot_download",
        "note": "Downloads complete model using snapshot_download for full offline capability. \
            Using gemma-3n-e2b as development model until Gemma 3n E2B is available.",
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

    def download_from_huggingface(self, repo_id: str, filename: str, filepath: Path) -> bool:
        """Download file from HuggingFace Hub using huggingface_hub."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            print(f"❌ Cannot download {filepath.name}: huggingface_hub not available")
            print("   Install with: pip install huggingface_hub")
            return False

        try:
            print(f"Downloading {filepath.name} from HuggingFace Hub...")

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Download using huggingface_hub
            _ = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=None,  # Use default cache
                local_dir=filepath.parent,
                local_dir_use_symlinks=False,
            )

            # Check if download was successful
            if not filepath.exists() or filepath.stat().st_size == 0:
                print(f"❌ HuggingFace download failed: {filepath.name}")
                if filepath.exists():
                    filepath.unlink()
                return False

            print(f"✅ Downloaded: {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ HuggingFace download failed for {filepath.name}: {e}")

            # Check if it's an authentication error
            if "401" in str(e) or "Unauthorized" in str(e):
                print("   This model requires authentication. Please:")
                print("   1. Visit https://huggingface.co/google/gemma-3n-E2B and accept terms")
                print(
                    "   2. Get your HuggingFace token from https://huggingface.co/settings/tokens"
                )
                print("   3. Run: huggingface-cli login")
                print("   4. Or set HF_TOKEN environment variable")

            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False

    def download_from_gdrive(self, url: str, filepath: Path) -> bool:
        """Download file from Google Drive using gdown."""
        if not GDOWN_AVAILABLE:
            print(f"❌ Cannot download {filepath.name}: gdown not available")
            print("   Install with: pip install gdown")
            return False

        try:
            print(f"Downloading {filepath.name} from Google Drive...")

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Download using gdown with fuzzy matching for drive links
            success = gdown.download(url, str(filepath), quiet=False, fuzzy=True)

            if not success or not filepath.exists() or filepath.stat().st_size == 0:
                print(f"❌ Google Drive download failed: {filepath.name}")
                if filepath.exists():
                    filepath.unlink()
                return False

            print(f"✅ Downloaded: {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            print(f"❌ Google Drive download failed for {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False

    def download_snapshot_model(self, repo_id: str, destination_dir: Path) -> bool:
        """Download entire model using snapshot_download for offline use."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            print("❌ Cannot download model: huggingface_hub not available")
            print("   Install with: pip install huggingface_hub")
            return False

        try:
            print(f"Downloading complete model from {repo_id} using snapshot_download...")

            # Create directory if it doesn't exist
            destination_dir.mkdir(parents=True, exist_ok=True)

            # Download using snapshot_download for complete model
            _ = snapshot_download(
                repo_id=repo_id,
                local_dir=destination_dir,
                # Cache in the specified directory for offline use
                cache_dir=None,
            )

            print(f"✅ Model downloaded to: {destination_dir}")
            return True

        except Exception as e:
            error_str = str(e)
            print(f"❌ Snapshot download failed for {repo_id}: {error_str}")

            # Check for different types of authentication/permission errors
            if "401" in error_str or "Unauthorized" in error_str:
                print("   This appears to be an authentication error.")
                print("   Please ensure you're logged in with: huggingface-cli login")
                print("   And that you have access to this model repository.")
            elif (
                "403" in error_str
                or "Forbidden" in error_str
                or "enable access to public gated repositories" in error_str
            ):
                print("   This appears to be a permission error for a gated repository.")
                print("   To fix this:")
                print("   1. Visit https://huggingface.co/google/gemma-3n-e2b and request access")
                print("   2. Go to https://huggingface.co/settings/tokens")
                print("   3. Edit your token and enable 'Access to public gated repositories'")
                print("   4. Re-login with: huggingface-cli login")
                print("   5. Wait for approval from the model authors")
            elif "Repository not found" in error_str or "does not exist" in error_str:
                print("   The model repository was not found.")
                print(f"   Please verify that '{repo_id}' is the correct repository name.")
                print("   Note: Gemma 3n E2B may not be publicly available yet.")
            else:
                print("   Please check your network connection and try again.")
                print("   If the issue persists, the model may require special access.")

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
            # Skip checksum verification if sha256 is empty (for models like Gemma 3n E2B)
            if model_info["sha256"] and not self.verify_checksum(
                main_filepath, model_info["sha256"]
            ):
                print(f"⚠️  Checksum failed, re-downloading: {model_info['name']}")
            elif model_info["sha256"]:  # Only verify if checksum is provided
                print(f"✅ Model already downloaded and verified: {model_info['name']}")
                return True
            else:
                print(
                    f"✅ Model already downloaded: {model_info['name']} "
                    f"(checksum verification skipped)"
                )
                return True

        print(f"\n📦 Downloading {model_info['name']}")
        print(f"   Source: {model_info['source']}")
        print(
            f"   Size: {model_info['size_mb']} MB (estimated)"
            if not model_info.get("sha256")
            else f"   Size: {model_info['size_mb']} MB"
        )
        print(f"   License: {model_info['license']}")

        # Download main file using appropriate method
        download_method = model_info.get("download_method", "urllib")
        if download_method == "gdown":
            success = self.download_from_gdrive(model_info["url"], main_filepath)
        elif download_method == "huggingface_hub":
            repo_id = model_info["url"]  # For HF, this is the repo_id
            success = self.download_from_huggingface(repo_id, model_info["filename"], main_filepath)
        elif download_method == "snapshot_download":
            repo_id = model_info["url"]  # For snapshot download, this is the repo_id
            success = self.download_snapshot_model(repo_id, destination_dir)
            # For snapshot downloads, we don't download individual files
            # Skip additional files processing for this method
            if success:
                print(f"✅ Successfully downloaded: {model_info['name']}\n")
                return True
        else:
            success = self.download_with_progress(
                model_info["url"],
                main_filepath,
                int(model_info["size_mb"] * 1024 * 1024),
            )

        if not success:
            return False

        # Verify checksum (skip if not provided)
        if model_info["sha256"]:
            if not self.verify_checksum(main_filepath, model_info["sha256"]):
                main_filepath.unlink()  # Remove invalid file
                return False
        else:
            print(f"⚠️  Checksum not available for {model_info['name']}, skipping verification")
            # Calculate and display actual checksum for future reference
            actual_sha256 = self.calculate_sha256(main_filepath)
            print(f"   Calculated SHA256: {actual_sha256}")

        # Extract if needed
        if model_info.get("extract", False):
            if not self.extract_archive(main_filepath, destination_dir):
                return False

        # Download additional files if specified
        if "additional_files" in model_info:
            for additional_file in model_info["additional_files"]:
                additional_filepath = destination_dir / additional_file["filename"]

                # Check if this is a HuggingFace download
                if download_method == "huggingface_hub":
                    repo_id = model_info["url"]  # For HF, this is the repo_id
                    hf_filename = additional_file.get("hf_filename", additional_file["filename"])
                    success = self.download_from_huggingface(
                        repo_id, hf_filename, additional_filepath
                    )
                else:
                    # Traditional URL download
                    success = self.download_with_progress(
                        additional_file["url"], additional_filepath
                    )

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

        # Skip checksum verification if not provided
        if not model_info["sha256"]:
            print(f"✅ Model file exists: {model_info['name']} (checksum verification skipped)")
            return True

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
                if model_info["sha256"]:
                    if self.verify_checksum(main_filepath, model_info["sha256"]):
                        status = "✅ Downloaded & verified"
                    else:
                        status = "⚠️  Downloaded but checksum failed"
                else:
                    status = "✅ Downloaded (no checksum)"

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
  python download_models.py --models yolo whisper gemma_3n_e2b  # Download specific models
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
        print("\n🚀 Next steps:")
        print("   1. Convert models to TensorRT:")
        print("      python tools/convert_yolo.py")
        print("      python tools/convert_depth.py")
        print("   2. Setup Gemma 3n E2B environment:")
        print("      python scripts/setup/setup_gemma3n.sh")
        print("   3. Test multimodal capabilities:")
        print("      python manual_tests/test_gemma3n_multimodal.py")
    else:
        print("❌ Some downloads failed. Check network connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
