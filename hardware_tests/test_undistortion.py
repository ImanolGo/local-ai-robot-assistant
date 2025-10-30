#!/usr/bin/env python3
"""DeepStream-based camera undistortion testing for NVIDIA Jetson IMX219.

Tests camera undistortion using calibration data with hardware-accelerated
capture via NVIDIA DeepStream. Creates before/after comparisons and validates
calibration quality with USB audio feedback for headless operation.

Usage:
  # Test with live camera capture
  python hardware_tests/test_undistortion.py

  # Test on existing calibration images
  python hardware_tests/test_undistortion.py --mode existing

  # Test both live and existing images
  python hardware_tests/test_undistortion.py --mode both

  # Custom calibration file
  python hardware_tests/test_undistortion.py --calibration config/custom_calibration.yaml

This script follows the project's guidelines: validate hardware connections,
handle errors gracefully, and provide detailed performance metrics with
USB audio feedback for headless operation.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import gi
import numpy as np
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger("undistortion_test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)


class USBAudioInterface:
    """USB audio interface for headless feedback using attached USB speakers."""

    def __init__(self):
        """Initialize USB audio interface and detect devices."""
        self.usb_speaker_device = None
        self.audio_available = False
        self._detect_usb_audio_devices()

    def _detect_usb_audio_devices(self) -> None:
        """Detect connected USB audio devices for output."""
        try:
            # List audio devices using aplay
            result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                # Parse output to find USB audio devices
                lines = result.stdout.split("\n")
                for line in lines:
                    if "USB" in line.upper() and "card" in line.lower():
                        # Extract card number
                        try:
                            card_num = line.split("card ")[1].split(":")[0]
                            self.usb_speaker_device = f"hw:{card_num},0"
                            self.audio_available = True
                            logger.info(f"Found USB audio device: {self.usb_speaker_device}")
                            break
                        except (IndexError, ValueError):
                            continue

            if not self.audio_available:
                logger.warning("No USB audio devices found - audio feedback disabled")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not detect audio devices: {e}")
            self.audio_available = False

    def speak(self, text: str) -> None:
        """Text-to-speech output using USB speakers.

        Args:
            text: Text to speak
        """
        if not self.audio_available:
            return

        try:
            # Use espeak with ALSA device specification
            subprocess.run(
                ["espeak", "-s", "150", "-a", "80", text],  # Speed  # Amplitude
                env={**os.environ, "ALSA_PCM_DEVICE": self.usb_speaker_device},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"TTS failed: {e}")

    def beep(self, frequency: int = 1000, duration: float = 0.1) -> None:
        """Generate beep sound using USB speakers.

        Args:
            frequency: Beep frequency in Hz
            duration: Beep duration in seconds
        """
        if not self.audio_available:
            # Fallback to terminal bell
            print("\a", end="", flush=True)
            return

        try:
            # Generate beep using speaker-test
            subprocess.run(
                [
                    "speaker-test",
                    "-D",
                    self.usb_speaker_device,
                    "-t",
                    "sine",
                    "-f",
                    str(frequency),
                    "-l",
                    "1",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to terminal bell
            print("\a", end="", flush=True)


class DeepStreamUndistortionTester:
    """Undistortion testing using NVIDIA DeepStream hardware acceleration.

    Tests camera undistortion using calibration data with hardware-accelerated
    image capture and processing. Creates visual comparisons and validates
    calibration quality for IMX219 fisheye camera.
    """

    def __init__(
        self,
        calibration_file: str = "config/camera_calibration.yaml",
        device_id: int = 0,
    ):
        """Initialize undistortion tester with DeepStream pipeline.

        Args:
            calibration_file: Path to camera calibration YAML file
            device_id: Camera device ID

        Raises:
            FileNotFoundError: If calibration file not found
            ValueError: If calibration data invalid
        """
        self.calibration_file = calibration_file
        self.device_id = device_id

        # Initialize GStreamer
        Gst.init(None)

        # Initialize audio interface
        self.audio = USBAudioInterface()

        # Camera calibration parameters
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.img_size: Optional[Tuple[int, int]] = None
        self.calibration_data: Optional[Dict] = None

        # Load calibration data
        self.load_calibration()

        logger.info("Undistortion tester initialized")
        logger.info(f"Calibration file: {calibration_file}")
        logger.info(f"Image size: {self.img_size[0]}x{self.img_size[1]}")

    def load_calibration(self) -> None:
        """Load calibration parameters from YAML file.

        Raises:
            FileNotFoundError: If calibration file not found
            ValueError: If calibration data invalid
        """
        if not os.path.exists(self.calibration_file):
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_file}")

        with open(self.calibration_file, "r") as f:
            self.calibration_data = yaml.safe_load(f)

        # Extract calibration parameters
        try:
            self.camera_matrix = np.array(self.calibration_data["camera_matrix"])
            self.dist_coeffs = np.array(self.calibration_data["distortion_coefficients"])
            self.img_size = (
                self.calibration_data["image_width"],
                self.calibration_data["image_height"],
            )
        except KeyError as e:
            raise ValueError(f"Invalid calibration data - missing key: {e}")

        logger.info("✓ Calibration loaded successfully")
        logger.info(
            f"  Calibration date: {self.calibration_data.get('calibration_date', 'Unknown')}"
        )
        logger.info(
            f"  Number of calibration images: {self.calibration_data.get('num_images', 'Unknown')}"
        )
        logger.info(
            f"  Checkerboard size: {self.calibration_data.get('checkerboard_size', 'Unknown')}"
        )

    def build_camera_pipeline(
        self, width: int = 1640, height: int = 1232, fps: int = 30
    ) -> Gst.Pipeline:
        """Build DeepStream pipeline for IMX219 camera capture.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frame rate

        Returns:
            Configured GStreamer pipeline

        Raises:
            Exception: If pipeline creation fails
        """
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={self.device_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1,format=NV12 ! "
            "nvvideoconvert ! "
            "video/x-raw,format=RGBA ! "
            "appsink name=sink emit-signals=True max-buffers=1 drop=True"
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
            logger.debug(f"Created DeepStream pipeline: {pipeline_str}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise

    def capture_frame(
        self, pipeline: Gst.Pipeline, width: int, height: int
    ) -> Optional[np.ndarray]:
        """Capture single frame from DeepStream pipeline.

        Args:
            pipeline: GStreamer pipeline
            width: Frame width
            height: Frame height

        Returns:
            Captured frame as numpy array or None if failed
        """
        sink = pipeline.get_by_name("sink")

        # Pull sample with timeout
        sample = sink.emit("try-pull-sample", Gst.SECOND * 2)
        if not sample:
            logger.warning("Timeout capturing frame")
            return None

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            logger.warning("Failed to map buffer")
            return None

        try:
            # Convert RGBA to BGR for OpenCV
            arr = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 4))
            frame_bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
            return frame_bgr
        finally:
            buffer.unmap(map_info)

    def undistort_image(
        self, img: np.ndarray, alpha: float = 0.5
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Apply undistortion to image using calibration parameters.

        Args:
            img: Input distorted image
            alpha: Free scaling parameter (0=maximize crop, 1=minimize crop)

        Returns:
            Tuple of (undistorted_image, roi) where roi is (x, y, w, h)
        """
        h, w = img.shape[:2]

        # Get optimal new camera matrix with specified alpha
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha, (w, h)
        )

        # Create undistortion maps for better quality
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            newcameramtx,
            (w, h),
            cv2.CV_32FC1,
        )

        # Apply undistortion using remap for better interpolation
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Only crop if alpha is low (aggressive cropping)
        if alpha < 0.8:
            x, y, w_crop, h_crop = roi
            if w_crop > 0 and h_crop > 0:
                dst = dst[y : y + h_crop, x : x + w_crop]

        return dst, roi

    def create_comparison_image(self, original: np.ndarray, undistorted: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image with labels.

        Args:
            original: Original distorted image
            undistorted: Undistorted image

        Returns:
            Side-by-side comparison image
        """
        # Resize undistorted to match original if needed
        if original.shape != undistorted.shape:
            undistorted = cv2.resize(undistorted, (original.shape[1], original.shape[0]))

        # Create labeled copies
        original_labeled = original.copy()
        undistorted_labeled = undistorted.copy()

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Red text for original (distorted)
        cv2.putText(
            original_labeled,
            "Original (Distorted)",
            (10, 40),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Green text for corrected (undistorted)
        cv2.putText(
            undistorted_labeled,
            "Corrected (Undistorted)",
            (10, 40),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

        # Create horizontal side-by-side comparison
        comparison = np.hstack([original_labeled, undistorted_labeled])

        return comparison

    def add_evaluation_grid(self, img: np.ndarray) -> np.ndarray:
        """Add evaluation grid lines to help assess distortion correction.

        Args:
            img: Input image

        Returns:
            Image with grid overlay
        """
        img_with_grid = img.copy()
        h, w = img.shape[:2]

        # Grid parameters
        grid_color = (0, 255, 255)  # Yellow
        thickness = 1

        # Vertical lines
        for x in range(w // 6, w, w // 6):
            cv2.line(img_with_grid, (x, 0), (x, h), grid_color, thickness)

        # Horizontal lines
        for y in range(h // 6, h, h // 6):
            cv2.line(img_with_grid, (0, y), (w, y), grid_color, thickness)

        return img_with_grid

    def capture_and_test_live(self, output_dir: str = "hardware_tests/undistortion_tests") -> int:
        """Capture live test images and create undistortion comparisons.

        Args:
            output_dir: Directory to save test results

        Returns:
            Number of test images processed
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("LIVE UNDISTORTION TESTING")
        logger.info("=" * 60)
        logger.info("\nInstructions:")
        logger.info("- Point camera at scenes with straight lines")
        logger.info("- Good test subjects: door frames, windows, grid patterns, buildings")
        logger.info("- Press ENTER to capture and test")
        logger.info("- Press 'q' to quit")
        logger.info("=" * 60)

        # Create DeepStream pipeline
        try:
            pipeline = self.build_camera_pipeline()
            pipeline.set_state(Gst.State.PLAYING)
            time.sleep(2)  # Warm-up period
            logger.info("Camera pipeline started successfully")
        except Exception as e:
            logger.error(f"Failed to start camera pipeline: {e}")
            return 0

        self.audio.speak("Ready to test undistortion")

        test_count = 0
        width, height = 1640, 1232  # Test resolution

        try:
            while True:
                # Capture frame
                frame = self.capture_frame(pipeline, width, height)
                if frame is None:
                    continue

                # Save preview
                preview_path = os.path.join(output_dir, "preview.jpg")
                cv2.imwrite(preview_path, frame)

                print(f"\rPreview saved to: {preview_path}", end="", flush=True)

                # User interaction
                response = (
                    input("\n\nPress ENTER to capture test image, 'q' to quit: ").strip().lower()
                )

                if response == "q":
                    break
                elif response == "":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"live_test_{test_count:02d}_{timestamp}"

                    # Save original
                    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
                    cv2.imwrite(original_path, frame)
                    logger.info(f"✓ Original saved: {original_path}")

                    # Apply undistortion with balanced alpha (less cropping)
                    logger.info("  Processing undistortion...")
                    undistorted, roi = self.undistort_image(frame, alpha=0.5)

                    # Save undistorted
                    undistorted_path = os.path.join(output_dir, f"{base_name}_undistorted.jpg")
                    cv2.imwrite(undistorted_path, undistorted)
                    logger.info(f"✓ Undistorted saved: {undistorted_path}")

                    # Create and save comparison
                    comparison = self.create_comparison_image(frame, undistorted)
                    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
                    cv2.imwrite(comparison_path, comparison)
                    logger.info(f"✓ Comparison saved: {comparison_path}")

                    # Create grid overlay comparison for evaluation
                    frame_grid = self.add_evaluation_grid(frame)
                    undistorted_grid = self.add_evaluation_grid(undistorted)
                    grid_comparison = self.create_comparison_image(frame_grid, undistorted_grid)
                    grid_path = os.path.join(output_dir, f"{base_name}_grid_comparison.jpg")
                    cv2.imwrite(grid_path, grid_comparison)
                    logger.info(f"✓ Grid comparison saved: {grid_path}")

                    # Test different alpha values for comparison
                    for alpha_val in [0.0, 1.0]:
                        alpha_name = f"alpha{alpha_val:.1f}".replace(".", "_")
                        undist_alpha, _ = self.undistort_image(frame, alpha=alpha_val)
                        alpha_path = os.path.join(output_dir, f"{base_name}_{alpha_name}.jpg")
                        cv2.imwrite(alpha_path, undist_alpha)
                        logger.info(f"✓ Alpha {alpha_val} version saved: {alpha_path}")

                    logger.info(
                        f"ROI (Region of Interest): x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}"
                    )
                    logger.info("Alpha parameter used: 0.5 (balanced cropping)")

                    self.audio.speak(f"Test image {test_count + 1} complete")
                    test_count += 1
                    logger.info("-" * 60)

        except KeyboardInterrupt:
            logger.info("\nTesting interrupted by user")
        finally:
            pipeline.set_state(Gst.State.NULL)

        logger.info("=" * 60)
        logger.info(f"LIVE TESTING COMPLETE! {test_count} test images processed")
        logger.info(f"Results saved in: {output_dir}/")
        logger.info("=" * 60)

        return test_count

    def test_existing_images(
        self,
        input_dir: str = "hardware_tests/calibration_images",
        output_dir: str = "hardware_tests/undistortion_tests",
    ) -> int:
        """Test undistortion on existing calibration images.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save test results

        Returns:
            Number of images processed
        """
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("TESTING UNDISTORTION ON EXISTING IMAGES")
        logger.info("=" * 60)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Get all image files
        if not os.path.exists(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return 0

        image_files = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return 0

        logger.info(f"Found {len(image_files)} images. Processing...")
        self.audio.speak(f"Processing {len(image_files)} images")

        processed_count = 0

        for i, img_file in enumerate(sorted(image_files)):
            img_path = os.path.join(input_dir, img_file)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not load: {img_file}")
                continue

            logger.info(f"[{i+1}/{len(image_files)}] Processing: {img_file}")

            # Apply undistortion with balanced alpha
            undistorted, roi = self.undistort_image(img, alpha=0.5)

            # Generate output filenames
            base_name = os.path.splitext(img_file)[0]

            # Save undistorted image
            undistorted_path = os.path.join(output_dir, f"{base_name}_undistorted.jpg")
            cv2.imwrite(undistorted_path, undistorted)

            # Create and save comparison
            comparison = self.create_comparison_image(img, undistorted)
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
            cv2.imwrite(comparison_path, comparison)

            # Create grid overlay comparison for evaluation
            img_grid = self.add_evaluation_grid(img)
            undistorted_grid = self.add_evaluation_grid(undistorted)
            grid_comparison = self.create_comparison_image(img_grid, undistorted_grid)
            grid_path = os.path.join(output_dir, f"{base_name}_grid_comparison.jpg")
            cv2.imwrite(grid_path, grid_comparison)

            # Test different alpha values for comparison
            for alpha_val in [0.0, 1.0]:
                alpha_name = f"alpha{alpha_val:.1f}".replace(".", "_")
                undist_alpha, _ = self.undistort_image(img, alpha=alpha_val)
                alpha_path = os.path.join(output_dir, f"{base_name}_{alpha_name}.jpg")
                cv2.imwrite(alpha_path, undist_alpha)

            logger.info(f"  ✓ Undistorted: {undistorted_path}")
            logger.info(f"  ✓ Comparison: {comparison_path}")
            logger.info(f"  ✓ Grid comparison: {grid_path}")
            logger.info("  ✓ Alpha variants: alpha_0_0.jpg, alpha_1_0.jpg")

            processed_count += 1

        logger.info("=" * 60)
        logger.info(f"PROCESSING COMPLETE! {processed_count} images processed")
        logger.info(f"Results saved in: {output_dir}/")
        logger.info("=" * 60)

        return processed_count

    def analyze_undistortion_quality(
        self, output_dir: str = "hardware_tests/undistortion_tests"
    ) -> None:
        """Analyze and report on undistortion quality.

        Args:
            output_dir: Directory containing test results
        """
        logger.info("=" * 60)
        logger.info("UNDISTORTION QUALITY ANALYSIS")
        logger.info("=" * 60)

        # Count processed images
        if os.path.exists(output_dir):
            comparison_files = [f for f in os.listdir(output_dir) if f.endswith("_comparison.jpg")]
            grid_files = [f for f in os.listdir(output_dir) if f.endswith("_grid_comparison.jpg")]

            logger.info(f"Generated {len(comparison_files)} comparison images")
            logger.info(f"Generated {len(grid_files)} grid evaluation images")

        logger.info("\nCalibration Parameters:")
        if self.calibration_data:
            logger.info(
                f"  Checkerboard size: "
                f"{self.calibration_data.get('checkerboard_size', 'Unknown')}"
            )
            logger.info(
                f"  Number of calibration images: "
                f"{self.calibration_data.get('num_images', 'Unknown')}"
            )
            logger.info(
                f"  Square size: {self.calibration_data.get('square_size_mm', 'Unknown')}mm"
            )

        logger.info("\nEvaluation Guidelines:")
        logger.info("  ✓ Straight lines should appear straighter in undistorted images")
        logger.info("  ✓ Grid lines should be more parallel and perpendicular")
        logger.info("  ✓ Reduced barrel/pincushion distortion at image edges")
        logger.info("  ✓ Better geometric accuracy overall")
        logger.info("  ✓ Objects should appear less 'fisheye' distorted")

        logger.info("\nRecommendations:")
        logger.info("  - Review grid comparison images for line straightness")
        logger.info("  - Check edge regions for distortion correction")
        logger.info("  - Validate with known geometric patterns")
        logger.info("  - If quality is poor, consider recalibration with more images")

        logger.info("=" * 60)


def run_undistortion_tests(
    mode: str = "live",
    calibration_file: str = "config/camera_calibration.yaml",
    device_id: int = 0,
    input_dir: str = "hardware_tests/calibration_images",
    output_dir: str = "hardware_tests/undistortion_tests",
    alpha: float = 0.5,
) -> bool:
    """Run undistortion testing workflow.

    Args:
        mode: Testing mode ('live', 'existing', or 'both')
        calibration_file: Path to calibration file
        device_id: Camera device ID
        input_dir: Directory with existing images (for 'existing' mode)
        output_dir: Directory to save test results
        alpha: Alpha parameter for undistortion (0.0=max crop, 1.0=min crop)

    Returns:
        True if testing successful, False otherwise
    """
    try:
        # Check if calibration file exists
        if not os.path.exists(calibration_file):
            logger.error(f"Calibration file not found: {calibration_file}")
            logger.error("Please run calibrate_camera.py first to generate calibration data")
            return False

        # Create tester
        tester = DeepStreamUndistortionTester(calibration_file, device_id)

        total_processed = 0

        # Run tests based on mode
        if mode in ["existing", "both"]:
            logger.info("Testing on existing calibration images...")
            count = tester.test_existing_images(input_dir, output_dir)
            total_processed += count

            if mode == "both":
                input("\nPress ENTER to continue with live camera testing...")

        if mode in ["live", "both"]:
            logger.info("Starting live camera testing...")
            count = tester.capture_and_test_live(output_dir)
            total_processed += count

        # Analyze results
        tester.analyze_undistortion_quality(output_dir)

        logger.info("=" * 60)
        logger.info("UNDISTORTION TESTING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {total_processed}")
        logger.info(f"Results saved in: {output_dir}/")
        logger.info("=" * 60)

        return total_processed > 0

    except Exception as e:
        logger.error(f"Undistortion testing failed: {e}")
        return False


def main() -> None:
    """Main entry point with CLI argument parsing."""
    examples = (
        "Examples:\n"
        "  # Test with live camera capture (balanced cropping)\n"
        "  python hardware_tests/test_undistortion.py\n\n"
        "  # Test with minimal cropping (keep more image area)\n"
        "  python hardware_tests/test_undistortion.py --alpha 1.0\n\n"
        "  # Test with maximum cropping (best quality, less area)\n"
        "  python hardware_tests/test_undistortion.py --alpha 0.0\n\n"
        "  # Test on existing calibration images\n"
        "  python hardware_tests/test_undistortion.py --mode existing\n\n"
        "  # Test both existing and live images\n"
        "  python hardware_tests/test_undistortion.py --mode both\n\n"
        "  # Use custom calibration file\n"
        "  python hardware_tests/test_undistortion.py --calibration config/custom_cal.yaml\n"
    )

    parser = argparse.ArgumentParser(
        description="DeepStream camera undistortion testing for NVIDIA Jetson IMX219",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "existing", "both"],
        default="live",
        help="Testing mode: live camera, existing images, or both (default: live)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="config/camera_calibration.yaml",
        help="Path to camera calibration file (default: config/camera_calibration.yaml)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="hardware_tests/calibration_images",
        help="Input directory for existing images (default: hardware_tests/calibration_images)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for undistortion (0.0=max crop, 1.0=min crop, default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hardware_tests/undistortion_tests",
        help="Output directory for test results (default: hardware_tests/undistortion_tests)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("JETSON IMX219 UNDISTORTION TESTING (DeepStream)")
    logger.info("=" * 60)
    logger.info(f"Testing mode: {args.mode}")
    logger.info(f"Calibration file: {args.calibration}")
    logger.info(f"Camera device: {args.device}")
    logger.info(f"Alpha parameter: {args.alpha} (0.0=max crop, 1.0=min crop)")
    logger.info("=" * 60)

    try:
        success = run_undistortion_tests(
            mode=args.mode,
            calibration_file=args.calibration,
            device_id=args.device,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
        )

        if success:
            logger.info("✅ Undistortion testing completed successfully")
            exit(0)
        else:
            logger.error("❌ Undistortion testing failed")
            exit(1)

    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
