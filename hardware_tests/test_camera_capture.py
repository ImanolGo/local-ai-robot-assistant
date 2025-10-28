#!/usr/bin/env python3
"""DeepStream-based CSI camera validation for IMX219.

Tests multiple sensor modes with hardware-accelerated capture using NVIDIA DeepStream.
Captures frames at different resolutions and FOV settings, measures performance,
and saves sample images for inspection.

Usage:
  # Run full test suite
  python hardware_tests/test_camera_capture.py

  # Test specific sensor mode only
  python hardware_tests/test_camera_capture.py --mode 3

  # Custom output directory
  python hardware_tests/test_camera_capture.py --output-dir /path/to/save

This script follows the project's guidelines: validate hardware connections,
handle errors gracefully, and provide detailed performance metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import gi
import numpy as np
from PIL import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger("camera_test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)


class IMX219CameraTester:
    """Camera testing harness for NVIDIA Jetson CSI camera (IMX219).

    Tests multiple sensor modes with hardware acceleration using DeepStream.
    Measures performance and saves sample images for validation.
    """

    def __init__(self, output_dir: str = "hardware_tests/test_images"):
        """Initialize camera tester with output directory.

        Args:
            output_dir: Directory to save captured images
        """
        # Sensor mode information for IMX219
        self.sensor_modes: Dict[int, Dict] = {
            0: {
                "resolution": (3280, 2464),
                "fps": 21,
                "fov": "full",
                "description": "8MP full resolution (native)",
            },
            1: {
                "resolution": (3280, 1848),
                "fps": 28,
                "fov": "full",
                "description": "6MP wide (native aspect)",
            },
            2: {
                "resolution": (1920, 1080),
                "fps": 30,
                "fov": "cropped",
                "description": "2MP HD cropped center",
            },
            3: {
                "resolution": (1640, 1232),
                "fps": 30,
                "fov": "full",
                "description": "2MP full FOV",
            },
            4: {
                "resolution": (1280, 720),
                "fps": 60,
                "fov": "cropped",
                "description": "1MP HD cropped",
            },
            5: {
                "resolution": (820, 616),
                "fps": 60,
                "fov": "scaled",
                "description": "0.25x downscaled (quarter-resolution test)",
            },
        }

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        Gst.init(None)
        logger.info("Camera tester initialized with output dir: %s", self.output_dir)

    # ----------------------------------------------------------------------

    def build_pipeline(self, width: int, height: int, fps: int, device_id: int = 0) -> Gst.Pipeline:
        """Build a GPU-accelerated DeepStream pipeline for the CSI camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frame rate
            device_id: Camera device ID

        Returns:
            Configured GStreamer pipeline

        Raises:
            Exception: If pipeline creation fails
        """
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={device_id} ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},"
            f"framerate={fps}/1,format=NV12 ! "
            "nvvideoconvert ! "
            "video/x-raw,format=RGBA ! "
            "appsink name=sink emit-signals=True max-buffers=1 drop=True"
        )

        try:
            pipeline = Gst.parse_launch(pipeline_str)
            logger.debug("Created pipeline: %s", pipeline_str)
            return pipeline
        except Exception as e:
            logger.error("Failed to create pipeline: %s", e)
            raise

    # ----------------------------------------------------------------------

    def save_rgba_to_png(self, data: bytes, width: int, height: int, filename: str) -> bool:
        """Convert raw RGBA bytes into a PNG using Pillow.

        Args:
            data: Raw RGBA image data
            width: Image width
            height: Image height
            filename: Output filename

        Returns:
            True if save successful, False otherwise
        """
        try:
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
            img = Image.fromarray(arr[..., :3], mode="RGB")  # drop alpha channel
            img.save(filename)
            logger.debug("Saved image: %s", filename)
            return True
        except Exception as e:
            logger.error("Failed to save image %s: %s", filename, e)
            return False

    # ----------------------------------------------------------------------

    def capture_frames(
        self, pipeline: Gst.Pipeline, width: int, height: int, fps: int, n_frames: int = 5
    ) -> List[float]:
        """Capture frames from DeepStream pipeline and measure performance.

        Args:
            pipeline: GStreamer pipeline
            width: Frame width
            height: Frame height
            fps: Target frame rate
            n_frames: Number of frames to capture

        Returns:
            List of capture timestamps (latencies)
        """
        sink = pipeline.get_by_name("sink")
        timestamps = []

        logger.info("Capturing %d frames at %dx%d@%dfps", n_frames, width, height, fps)

        for i in range(n_frames):
            start = time.time()
            sample = sink.emit("try-pull-sample", Gst.SECOND * 5)
            end = time.time()

            if not sample:
                logger.warning("Timeout capturing frame %d at %dx%d@%dfps", i, width, height, fps)
                continue

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                logger.warning("Failed to map buffer for frame %d", i)
                continue

            png_path = os.path.join(self.output_dir, f"frame_{width}x{height}_{fps}fps_{i}.png")
            if self.save_rgba_to_png(map_info.data, width, height, png_path):
                logger.debug("Saved frame %d to %s", i, png_path)
            buffer.unmap(map_info)

            timestamps.append(end - start)

        logger.info("Captured %d/%d frames successfully", len(timestamps), n_frames)
        return timestamps

    # ----------------------------------------------------------------------

    def test_sensor_mode(self, mode: int, device_id: int = 0, n_frames: int = 5) -> Optional[Dict]:
        """Test a specific sensor mode and return performance results.

        Args:
            mode: Sensor mode to test
            device_id: Camera device ID
            n_frames: Number of frames to capture

        Returns:
            Performance results dictionary or None if failed
        """
        if mode not in self.sensor_modes:
            logger.error("Invalid sensor mode: %d", mode)
            return None

        info = self.sensor_modes[mode]
        w, h = info["resolution"]
        fps = info["fps"]

        logger.info("Testing Mode %d: %dx%d@%dfps (%s)", mode, w, h, fps, info["description"])

        try:
            pipeline = self.build_pipeline(w, h, fps, device_id)
            pipeline.set_state(Gst.State.PLAYING)
            time.sleep(2)  # warm-up for auto-exposure and white balance

            timestamps = self.capture_frames(pipeline, w, h, fps, n_frames)
            pipeline.set_state(Gst.State.NULL)

            if timestamps:
                avg_latency = sum(timestamps) / len(timestamps)
                fps_measured = 1 / avg_latency
                result = {
                    "mode": mode,
                    "resolution": f"{w}x{h}",
                    "requested_fps": fps,
                    "measured_fps": round(fps_measured, 2),
                    "avg_latency_ms": round(avg_latency * 1000, 2),
                    "description": info["description"],
                    "fov": info["fov"],
                    "success": True,
                }
                logger.info("Mode %d completed: %.2f fps measured", mode, fps_measured)
                return result
            else:
                logger.warning("Mode %d failed: no frames captured", mode)
                return {
                    "mode": mode,
                    "resolution": f"{w}x{h}",
                    "requested_fps": fps,
                    "measured_fps": 0,
                    "avg_latency_ms": None,
                    "description": info["description"],
                    "fov": info["fov"],
                    "success": False,
                }

        except Exception as e:
            logger.error("Mode %d failed with exception: %s", mode, e)
            return None

    # ----------------------------------------------------------------------

    def test_all_modes(self, device_id: int = 0, n_frames: int = 5) -> List[Dict]:
        """Test all sensor modes and return performance results.

        Args:
            device_id: Camera device ID
            n_frames: Number of frames to capture per mode

        Returns:
            List of performance results for each mode
        """
        results = []
        logger.info("Starting camera test suite for all %d sensor modes", len(self.sensor_modes))

        for mode in sorted(self.sensor_modes.keys()):
            result = self.test_sensor_mode(mode, device_id, n_frames)
            if result:
                results.append(result)

        self._print_summary(results)
        return results

    def _print_summary(self, results: List[Dict]) -> None:
        """Display camera test summary.

        Args:
            results: List of test results
        """
        logger.info("📊 DeepStream Camera Test Summary (IMX219):")
        for r in results:
            if r.get("success", False):
                logger.info(
                    "Mode %2d: %s @ %dfps → %.2ffps, %.2fms avg | %s FOV | %s",
                    r["mode"],
                    r["resolution"],
                    r["requested_fps"],
                    r["measured_fps"],
                    r["avg_latency_ms"],
                    r["fov"],
                    r["description"],
                )
            else:
                logger.warning(
                    "Mode %2d: %s @ %dfps → FAILED | %s FOV | %s",
                    r["mode"],
                    r["resolution"],
                    r["requested_fps"],
                    r["fov"],
                    r["description"],
                )

        successful_results = [r for r in results if r.get("success", False)]
        if successful_results:
            optimal = max(successful_results, key=lambda x: x["measured_fps"])
            logger.info(
                "🏁 Optimal: Mode %d → %s @ %dfps (%.2ffps measured)",
                optimal["mode"],
                optimal["resolution"],
                optimal["requested_fps"],
                optimal["measured_fps"],
            )
        else:
            logger.error(
                "❌ No successful tests - check camera connection and DeepStream installation"
            )


def run_camera_tests(
    device_id: int = 0,
    output_dir: str = "hardware_tests/test_images",
    mode: Optional[int] = None,
    n_frames: int = 5,
) -> bool:
    """Run camera tests with specified parameters.

    Args:
        device_id: Camera device ID
        output_dir: Directory to save images
        mode: Specific sensor mode to test (None for all)
        n_frames: Number of frames to capture per mode

    Returns:
        True if all tests successful, False otherwise
    """
    try:
        tester = IMX219CameraTester(output_dir)

        if mode is not None:
            result = tester.test_sensor_mode(mode, device_id, n_frames)
            return result is not None and result.get("success", False)
        else:
            results = tester.test_all_modes(device_id, n_frames)
            return all(r.get("success", False) for r in results)

    except Exception as e:
        logger.error("Camera test failed with exception: %s", e)
        return False


def main() -> None:
    """Main entry point with CLI argument parsing."""
    EXAMPLES = (
        "Examples:\n"
        "  # Test all sensor modes (full test suite)\n"
        "  python hardware_tests/test_camera_capture.py\n\n"
        "  # Test specific sensor mode only\n"
        "  python hardware_tests/test_camera_capture.py --mode 3\n\n"
        "  # Custom output directory\n"
        "  python hardware_tests/test_camera_capture.py --output-dir /tmp/camera_test\n\n"
        "  # Use different camera device\n"
        "  python hardware_tests/test_camera_capture.py --device 1\n\n"
        "  # Capture more frames per test\n"
        "  python hardware_tests/test_camera_capture.py --frames 10\n"
    )

    parser = argparse.ArgumentParser(
        description="DeepStream CSI camera validation for IMX219",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hardware_tests/test_images",
        help="Output directory for captured images (default: hardware_tests/test_images)",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=list(range(6)),  # modes 0-5
        help="Test specific sensor mode only (0-5). If omitted, all modes are tested.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to capture per mode (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Starting DeepStream camera test with device %d", args.device)

    try:
        success = run_camera_tests(
            device_id=args.device,
            output_dir=args.output_dir,
            mode=args.mode,
            n_frames=args.frames,
        )

        if success:
            logger.info("✅ All camera tests completed successfully")
            exit(0)
        else:
            logger.error("❌ Some camera tests failed")
            exit(1)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        exit(130)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        exit(1)


if __name__ == "__main__":
    main()
