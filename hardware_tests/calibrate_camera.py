#!/usr/bin/env python3
"""DeepStream-based camera calibration for NVIDIA Jetson IMX219.

Camera calibration script using NVIDIA DeepStream for hardware-accelerated
capture with USB audio feedback for headless operation. Generates calibration
parameters for fisheye distortion correction.

Usage:
  # Run with default settings
  python hardware_tests/calibrate_camera.py

  # Custom checkerboard configuration
  python hardware_tests/calibrate_camera.py --cols 9 --rows 6 --square-size 25.0

  # Specify number of images to capture
  python hardware_tests/calibrate_camera.py --num-images 30

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
from typing import Dict, List, Optional, Tuple

import cv2
import gi
import numpy as np
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger("camera_calibration")
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


class DeepStreamCameraCalibrator:
    """Camera calibration using NVIDIA DeepStream hardware acceleration.

    Performs camera calibration using checkerboard pattern detection with
    hardware-accelerated image capture and processing. Generates camera
    intrinsics matrix and distortion coefficients for IMX219 fisheye camera.
    """

    def __init__(
        self,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 25.0,
        num_images: int = 30,
        save_dir: str = "hardware_tests/calibration_images",
        device_id: int = 0,
    ):
        """Initialize camera calibrator with DeepStream pipeline.

        Args:
            checkerboard_size: Tuple of (columns, rows) of inner corners
            square_size: Size of checkerboard square in mm
            num_images: Number of calibration images to capture
            save_dir: Directory to save calibration images
            device_id: Camera device ID
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.num_images = num_images
        self.save_dir = save_dir
        self.device_id = device_id

        # Initialize GStreamer
        Gst.init(None)

        # Initialize audio interface
        self.audio = USBAudioInterface()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(
            -1, 2
        )
        self.objp *= square_size

        # Arrays to store object points and image points
        self.objpoints: List[np.ndarray] = []  # 3D points in real world space
        self.imgpoints: List[np.ndarray] = []  # 2D points in image plane

        # Camera calibration results
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.img_size: Optional[Tuple[int, int]] = None

        logger.info("Camera calibrator initialized")
        logger.info(f"Checkerboard: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
        logger.info(f"Square size: {square_size}mm")
        logger.info(f"Target images: {num_images}")
        logger.info(f"Save directory: {save_dir}")

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
        # Use optimal settings for calibration (good balance of resolution and processing speed)
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

    def detect_checkerboard_corners(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect checkerboard corners in frame with subpixel accuracy.

        Args:
            frame: Input image frame

        Returns:
            Tuple of (success, corners) where corners has subpixel accuracy
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        if ret:
            # Check if checkerboard covers enough of the image area
            corners_flat = corners.reshape(-1, 2)
            x_spread = np.max(corners_flat[:, 0]) - np.min(corners_flat[:, 0])
            y_spread = np.max(corners_flat[:, 1]) - np.min(corners_flat[:, 1])

            # Require checkerboard to cover at least 30% of image width/height
            min_coverage = 0.3
            if x_spread < gray.shape[1] * min_coverage or y_spread < gray.shape[0] * min_coverage:
                logger.debug(
                    f"Checkerboard too small: coverage "
                    f"{x_spread/gray.shape[1]:.2f}x{y_spread/gray.shape[0]:.2f}"
                )
                return False, None

            # Refine corners for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners_refined

        return False, None

    def capture_calibration_images(self) -> bool:
        """Capture calibration images with interactive guidance.

        Returns:
            True if images captured successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING CALIBRATION IMAGE CAPTURE")
        logger.info("=" * 60)
        logger.info(f"Target: {self.num_images} images with detected checkerboard")
        logger.info(
            f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} inner corners"
        )
        logger.info("\nGUIDANCE:")
        logger.info("- Hold checkerboard in front of camera")
        logger.info("- Try different angles: straight, tilted left/right, top/bottom")
        logger.info("- Try different distances: close, medium, far")
        logger.info("- Cover all areas of the image")
        logger.info("- Press ENTER to capture when checkerboard detected")
        logger.info("- Press 'q' to quit early")
        logger.info("=" * 60)

        self.audio.speak("Starting calibration capture")

        # Create DeepStream pipeline
        try:
            pipeline = self.build_camera_pipeline()
            pipeline.set_state(Gst.State.PLAYING)
            time.sleep(2)  # Warm-up for auto-exposure and white balance
            logger.info("Camera pipeline started successfully")
        except Exception as e:
            logger.error(f"Failed to start camera pipeline: {e}")
            return False

        captured_count = 0
        attempt_count = 0
        width, height = 1640, 1232  # Calibration resolution

        # Suggested capture positions for good calibration coverage
        suggested_positions = [
            "Center, straight on",
            "Left side, tilted right",
            "Right side, tilted left",
            "Top, tilted down",
            "Bottom, tilted up",
            "Close distance",
            "Far distance",
            "45° angle from left",
            "45° angle from right",
            "Corner positions (4 corners)",
            "Various rotations",
        ]

        try:
            while captured_count < self.num_images:
                # Capture frame
                frame = self.capture_frame(pipeline, width, height)
                if frame is None:
                    continue

                if self.img_size is None:
                    self.img_size = (frame.shape[1], frame.shape[0])

                # Detect checkerboard corners
                corners_found, corners = self.detect_checkerboard_corners(frame)

                # Draw corners on frame for visualization
                if corners_found and corners is not None:
                    cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, True)

                # Save preview frame
                preview_path = os.path.join(self.save_dir, "preview.jpg")
                cv2.imwrite(preview_path, frame)

                # Status update
                attempt_count += 1
                status = "DETECTED ✓" if corners_found else "NOT DETECTED ✗"
                print(
                    f"\r[{captured_count}/{self.num_images}] Checkerboard: {status} | "
                    f"Attempt: {attempt_count} | Preview: {preview_path}",
                    end="",
                    flush=True,
                )

                # User interaction when checkerboard detected
                if corners_found:
                    self.audio.beep(1000, 0.05)  # Success beep
                    print(f"\n\n✓ Checkerboard detected! Preview saved to: {preview_path}")

                    # Show suggested position for this capture
                    if captured_count < len(suggested_positions):
                        logger.info(f"\nSuggested position: {suggested_positions[captured_count]}")

                    response = (
                        input("\nPress ENTER to save, 's' to skip, 'q' to quit: ").strip().lower()
                    )

                    if response == "q":
                        logger.info("Quitting calibration capture...")
                        break
                    elif response == "s":
                        logger.info("Skipping this frame...")
                        continue
                    elif response == "":
                        # Save calibration image and data
                        self.objpoints.append(self.objp)
                        self.imgpoints.append(corners)

                        # Save image with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(
                            self.save_dir, f"calib_{captured_count:02d}_{timestamp}.jpg"
                        )
                        cv2.imwrite(img_path, frame)

                        captured_count += 1
                        self.audio.speak(f"Image {captured_count} saved")
                        logger.info(f"✓ Saved: {img_path}")
                        logger.info("-" * 60)
                else:
                    # Check if checkerboard was detected but rejected for being too small
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_check, corners_check = cv2.findChessboardCorners(
                        gray, self.checkerboard_size, None
                    )
                    if ret_check:
                        corners_flat = corners_check.reshape(-1, 2)
                        x_spread = np.max(corners_flat[:, 0]) - np.min(corners_flat[:, 0])
                        y_spread = np.max(corners_flat[:, 1]) - np.min(corners_flat[:, 1])
                        if x_spread < gray.shape[1] * 0.3 or y_spread < gray.shape[0] * 0.3:
                            print(
                                "\r⚠ WARNING: Checkerboard too small or far away. "
                                "Move closer for better coverage.",
                                end="",
                                flush=True,
                            )
                            time.sleep(1)  # Show warning for longer

                    time.sleep(0.1)  # Brief pause before next attempt

        except KeyboardInterrupt:
            logger.info("\nCapture interrupted by user")
        finally:
            pipeline.set_state(Gst.State.NULL)

        logger.info("=" * 60)
        logger.info(f"CAPTURE COMPLETE! {captured_count} images saved")
        logger.info("=" * 60)
        self.audio.speak(f"Captured {captured_count} images")

        return captured_count > 0

    def calibrate_camera(self) -> float:
        """Perform camera calibration using captured images.

        Returns:
            Mean reprojection error in pixels

        Raises:
            ValueError: If no calibration images available
            RuntimeError: If calibration fails
        """
        if len(self.objpoints) == 0:
            raise ValueError(
                "No calibration images available. Run capture_calibration_images() first."
            )

        logger.info("=" * 60)
        logger.info("RUNNING CAMERA CALIBRATION")
        logger.info("=" * 60)
        logger.info(f"Processing {len(self.objpoints)} images...")
        self.audio.speak("Calibrating camera")

        # Perform camera calibration
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_size, None, None
        )

        if not ret:
            raise RuntimeError("Camera calibration failed!")

        # Calculate reprojection error
        total_error = 0.0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i],
                self.rvecs[i],
                self.tvecs[i],
                self.camera_matrix,
                self.dist_coeffs,
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(self.objpoints)

        logger.info("✓ Calibration successful!")
        logger.info(f"Mean reprojection error: {mean_error:.4f} pixels")
        logger.info(f"\nCamera Matrix:\n{self.camera_matrix}")
        logger.info(f"\nDistortion Coefficients:\n{self.dist_coeffs}")
        logger.info("=" * 60)

        self.audio.speak("Calibration complete")

        return mean_error

    def save_calibration(self, filename: str = "config/camera_calibration.yaml") -> None:
        """Save calibration parameters to YAML file.

        Args:
            filename: Path to save calibration file

        Raises:
            ValueError: If no calibration data available
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("No calibration data available. Run calibrate_camera() first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Prepare calibration data dictionary
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "image_width": self.img_size[0],
            "image_height": self.img_size[1],
            "calibration_date": datetime.now().isoformat(),
            "num_images": len(self.objpoints),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size,
            "device_id": self.device_id,
            "capture_resolution": f"{self.img_size[0]}x{self.img_size[1]}",
            "pipeline_type": "DeepStream_nvarguscamerasrc",
        }

        # Save to YAML file
        with open(filename, "w") as f:
            yaml.dump(calibration_data, f, default_flow_style=False, indent=2)

        logger.info(f"✓ Calibration saved to: {filename}")
        self.audio.speak("Calibration saved")

    def load_calibration(self, filename: str = "config/camera_calibration.yaml") -> Dict:
        """Load calibration parameters from YAML file.

        Args:
            filename: Path to calibration file

        Returns:
            Calibration data dictionary
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        self.camera_matrix = np.array(data["camera_matrix"])
        self.dist_coeffs = np.array(data["distortion_coefficients"])
        self.img_size = (data["image_width"], data["image_height"])

        logger.info(f"✓ Calibration loaded from: {filename}")
        return data


def run_calibration_workflow(
    checkerboard_cols: int = 9,
    checkerboard_rows: int = 6,
    square_size: float = 25.0,
    num_images: int = 25,
    device_id: int = 0,
    save_dir: str = "hardware_tests/calibration_images",
) -> bool:
    """Run complete camera calibration workflow.

    Args:
        checkerboard_cols: Number of inner corner columns
        checkerboard_rows: Number of inner corner rows
        square_size: Size of checkerboard square in mm
        num_images: Number of images to capture
        device_id: Camera device ID
        save_dir: Directory to save images

    Returns:
        True if calibration successful, False otherwise
    """
    try:
        # Create calibrator
        calibrator = DeepStreamCameraCalibrator(
            checkerboard_size=(checkerboard_cols, checkerboard_rows),
            square_size=square_size,
            num_images=num_images,
            save_dir=save_dir,
            device_id=device_id,
        )

        # Capture calibration images
        logger.info("Starting image capture phase...")
        success = calibrator.capture_calibration_images()

        if not success:
            logger.error("No images captured. Calibration failed.")
            return False

        # Perform calibration
        logger.info("Starting calibration computation...")
        mean_error = calibrator.calibrate_camera()

        # Validate calibration quality
        if mean_error > 1.0:
            logger.warning(f"High reprojection error ({mean_error:.4f} pixels)")
            logger.warning("Consider recapturing with:")
            logger.warning("  - Better lighting")
            logger.warning("  - More varied angles")
            logger.warning("  - Clearer checkerboard visibility")

            response = input("\nSave calibration anyway? [y/N]: ").strip().lower()
            if response != "y":
                logger.info("Calibration not saved. Exiting.")
                return False

        # Save calibration
        calibrator.save_calibration()

        logger.info("=" * 60)
        logger.info("CALIBRATION WORKFLOW COMPLETE!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Review calibration images in calibration_images/ directory")
        logger.info("2. Test undistortion with test_undistortion.py")
        logger.info("3. Use calibration data in perception pipeline")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Calibration workflow failed: {e}")
        return False


def main() -> None:
    """Main entry point with CLI argument parsing."""
    examples = (
        "Examples:\n"
        "  # Run with default settings (9x6 checkerboard, 30 images)\n"
        "  python hardware_tests/calibrate_camera.py\n\n"
        "  # Custom checkerboard configuration\n"
        "  python hardware_tests/calibrate_camera.py --cols 7 --rows 5 --square-size 30.0\n\n"
        "  # Capture more images for better accuracy\n"
        "  python hardware_tests/calibrate_camera.py --num-images 40\n\n"
        "  # Use different camera device\n"
        "  python hardware_tests/calibrate_camera.py --device 1\n"
    )

    parser = argparse.ArgumentParser(
        description="DeepStream camera calibration for NVIDIA Jetson IMX219",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of inner corner columns in checkerboard (default: 9)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of inner corner rows in checkerboard (default: 6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Size of checkerboard square in mm (default: 25.0)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of calibration images to capture (default: 30)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="hardware_tests/calibration_images",
        help="Directory to save calibration images (default: hardware_tests/calibration_images)",
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
    logger.info("JETSON IMX219 CAMERA CALIBRATION (DeepStream)")
    logger.info("=" * 60)
    logger.info(f"Checkerboard: {args.cols}x{args.rows} inner corners")
    logger.info(f"Square size: {args.square_size}mm")
    logger.info(f"Target images: {args.num_images}")
    logger.info(f"Camera device: {args.device}")
    logger.info("=" * 60)

    try:
        success = run_calibration_workflow(
            checkerboard_cols=args.cols,
            checkerboard_rows=args.rows,
            square_size=args.square_size,
            num_images=args.num_images,
            device_id=args.device,
            save_dir=args.save_dir,
        )

        if success:
            logger.info("✅ Camera calibration completed successfully")
            exit(0)
        else:
            logger.error("❌ Camera calibration failed")
            exit(1)

    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
