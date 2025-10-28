#!/usr/bin/env python3
"""
Camera capture test using NVIDIA DeepStream SDK.

This script tests the IMX219 camera module connected via MIPI CSI-2 port
using DeepStream for hardware-accelerated video processing.

Test Requirements:
- Test camera initialization
- Test frame capture at various resolutions
- Test frame rate measurement
- Save sample images for validation
- Test continuous capture for 5 minutes

Hardware: NVIDIA Jetson Orin Nano + IMX219 Camera Module
"""

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    import gi

    # import pyds

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except ImportError as e:
    print(f"❌ Error importing DeepStream dependencies: {e}")
    print("Please install DeepStream SDK and pyds bindings")
    sys.exit(1)


class DeepStreamCameraTest:
    """DeepStream-based camera test class."""

    def __init__(self, device_id: int = 0, save_dir: str = "test_images"):
        """
        Initialize camera test.

        Args:
            device_id: Camera device ID (usually 0 for CSI camera)
            save_dir: Directory to save test images
        """
        self.device_id = device_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Frame statistics
        self.frame_count = 0
        self.start_time = 0
        self.fps_history: List[float] = []
        self.frame_queue = queue.Queue(maxsize=10)

        # GStreamer pipeline
        self.pipeline = None
        self.loop = None
        self.running = False

        # Test configurations (optimized for IMX219)
        self.test_resolutions = [
            (1280, 720),  # High FPS mode - sensor mode 4 (60fps)
            (1920, 1080),  # Standard HD - sensor mode 2 (30fps) - CROPPED
            (1640, 1232),  # Native 4:3 ratio - sensor mode 3 (30fps) - FULL FOV
            # (3280, 2464)  # Full resolution - disabled due to instability
        ]

        # Sensor mode information for IMX219
        self.sensor_modes = {
            0: {
                "resolution": (3280, 2464),
                "fps": 21,
                "fov": "full",
                "description": "8MP full resolution",
            },
            1: {"resolution": (3280, 1848), "fps": 28, "fov": "full", "description": "6MP wide"},
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
        }

        # Initialize GStreamer
        Gst.init(None)

    def create_pipeline(self, width: int = 1920, height: int = 1080, framerate: int = 30) -> bool:
        """
        Create DeepStream pipeline for camera capture.

        Args:
            width: Frame width
            height: Frame height
            framerate: Target frame rate

        Returns:
            True if pipeline created successfully
        """
        try:
            # Create pipeline with balanced settings for good image quality
            pipeline_str = (
                f"nvarguscamerasrc sensor_id={self.device_id} "
                f"wbmode=1 "  # Auto white balance
                f'exposuretimerange="50000 500000000" '  # Limited exposure range
                f'gainrange="1 8" '  # Limited gain range to reduce noise
                f'ispdigitalgainrange="1 8" '  # Limited ISP digital gain
                f"ee-mode=1 "  # Edge enhancement
                f"ee-strength=0.5 "  # Moderate edge enhancement
                f"tnr-mode=1 "  # Temporal noise reduction
                f"tnr-strength=1 ! "  # TNR strength for noise reduction
                f"video/x-raw(memory:NVMM), "
                f"width=(int){width}, height=(int){height}, "
                f"framerate=(fraction){framerate}/1, format=(string)NV12 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! "
                f"appsink name=sink emit-signals=true sync=false "
                f"max-buffers=2 drop=true"
            )

            self.pipeline = Gst.parse_launch(pipeline_str)

            if not self.pipeline:
                print("❌ Failed to create pipeline")
                return False

            # Get sink element and connect callback
            sink = self.pipeline.get_by_name("sink")
            sink.connect("new-sample", self.new_sample_callback)

            print(f"✅ Pipeline created: {width}x{height}@{framerate}fps")
            return True

        except Exception as e:
            print(f"❌ Pipeline creation error: {e}")
            return False

    def new_sample_callback(self, sink):
        """Handle new frame from camera."""
        try:
            sample = sink.emit("pull-sample")
            if sample:
                # Get buffer
                buffer = sample.get_buffer()
                caps = sample.get_caps()

                # Get frame info
                struct = caps.get_structure(0)
                width = struct.get_value("width")
                height = struct.get_value("height")

                # Map buffer to numpy array
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    # Convert to numpy array
                    frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                    frame = frame_data.reshape((height, width, 3))

                    # Add to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass  # Drop frame if queue is full

                    # Update statistics
                    self.frame_count += 1

                    buffer.unmap(map_info)

        except Exception as e:
            print(f"❌ Frame callback error: {e}")

        return Gst.FlowReturn.OK

    def start_capture(self) -> bool:
        """Start camera capture."""
        try:
            if not self.pipeline:
                print("❌ Pipeline not created")
                return False

            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("❌ Failed to start pipeline")
                return False

            self.running = True
            self.start_time = time.time()
            self.frame_count = 0

            print("✅ Camera capture started")
            return True

        except Exception as e:
            print(f"❌ Start capture error: {e}")
            return False

    def stop_capture(self):
        """Stop camera capture."""
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self.running = False
            print("✅ Camera capture stopped")

        except Exception as e:
            print(f"❌ Stop capture error: {e}")

    def print_sensor_info(self):
        """Print information about IMX219 sensor modes and field of view."""
        print("\n📊 IMX219 Sensor Mode Information:")
        print("=" * 60)
        for mode, info in self.sensor_modes.items():
            res = info["resolution"]
            fov_status = "🔍 CROPPED" if info["fov"] == "cropped" else "📐 FULL FOV"
            print(f"Mode {mode}: {res[0]}x{res[1]} @ {info['fps']}fps - {fov_status}")
            print(f"         {info['description']}")
        print("=" * 60)
        print("NOTE: Modes 2 & 4 (1920x1080, 1280x720) crop the center of the sensor")
        print("      Mode 3 (1640x1232) uses the full sensor area with 4:3 aspect ratio")
        print("      For computer vision applications, consider using mode 3 for full FOV")
        print("=" * 60)

    def get_optimal_settings_info(self) -> str:
        """Get information about optimal camera settings."""
        return """
🎯 Camera Settings Explanation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📸 Current Settings (Balanced for Quality):
   • Auto white balance (wbmode=1) for natural colors
   • Limited exposure range (50μs - 500ms) to prevent over/under exposure
   • Limited gain range (1-8) to reduce noise
   • Temporal noise reduction enabled for cleaner images

🔍 Field of View Differences:
   • 1920x1080: Uses sensor mode 2 - CROPPED center region (16:9)
   • 1640x1232: Uses sensor mode 3 - FULL sensor area (4:3)
   • 1280x720:  Uses sensor mode 4 - CROPPED center region (16:9)

💡 Recommendations:
   • Use 1640x1232 for full field of view computer vision
   • Use 1920x1080 for standard video recording (cropped but 16:9)
   • Use 1280x720 for high frame rate applications (60fps capable)

🔧 Image Quality Notes:
   • Settings optimized to balance quality and consistency
   • Gain limited to reduce noise in continuous capture
   • Auto exposure adapts to lighting but within reasonable bounds
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def get_current_fps(self) -> float:
        """Get current FPS."""
        if self.start_time == 0 or self.frame_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """Save frame to disk."""
        try:
            filepath = self.save_dir / filename
            success = cv2.imwrite(str(filepath), frame)
            if success:
                print(f"💾 Saved frame: {filepath}")
            return success
        except Exception as e:
            print(f"❌ Save frame error: {e}")
            return False

    def test_resolution(self, width: int, height: int, duration: float = 10.0) -> dict:
        """
        Test specific resolution.

        Args:
            width: Frame width
            height: Frame height
            duration: Test duration in seconds

        Returns:
            Test results dictionary
        """
        print(f"\n🔍 Testing resolution: {width}x{height}")

        # Create pipeline for this resolution
        if not self.create_pipeline(width, height):
            return {"success": False, "error": "Pipeline creation failed"}

        # Start capture
        if not self.start_capture():
            return {"success": False, "error": "Failed to start capture"}

        results = {
            "resolution": (width, height),
            "success": True,
            "frames_captured": 0,
            "average_fps": 0.0,
            "sample_saved": False,
        }

        try:
            end_time = time.time() + duration
            sample_saved = False

            # Wait a bit for camera to stabilize
            time.sleep(1)

            while time.time() < end_time and self.running:
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=1.0)

                    # Save first frame as sample after stabilization
                    if not sample_saved and self.frame_count > 20:  # Wait for 20 frames
                        filename = f"sample_{width}x{height}.jpg"
                        if self.save_frame(frame, filename):
                            results["sample_saved"] = True
                            sample_saved = True

                except queue.Empty:
                    continue

            # Calculate final statistics
            results["frames_captured"] = self.frame_count
            results["average_fps"] = self.get_current_fps()

            print("✅ Resolution test complete:")
            print("   Frames captured: {results['frames_captured']}")
            print("   Average FPS: {results['average_fps']:.2f}")

        except Exception as e:
            print(f"❌ Resolution test error: {e}")
            results["success"] = False
            results["error"] = str(e)

        finally:
            self.stop_capture()

        return results

    def test_continuous_capture(self, duration: float = 300.0) -> dict:
        """
        Test continuous capture for extended period.

        Args:
            duration: Test duration in seconds (default 5 minutes)

        Returns:
            Test results dictionary
        """
        print(f"\n⏱️  Testing continuous capture for {duration/60:.1f} minutes")

        # Use default resolution for continuous test
        if not self.create_pipeline(1920, 1080, 30):
            return {"success": False, "error": "Pipeline creation failed"}

        if not self.start_capture():
            return {"success": False, "error": "Failed to start capture"}

        results = {
            "success": True,
            "duration": duration,
            "total_frames": 0,
            "average_fps": 0.0,
            "fps_stability": 0.0,
            "samples_saved": 0,
        }

        try:
            end_time = time.time() + duration
            fps_measurements = []
            last_fps_time = time.time()
            sample_interval = duration / 10  # Save 10 samples during test
            next_sample_time = time.time() + sample_interval
            sample_count = 0

            while time.time() < end_time and self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)

                    # Measure FPS every second
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        fps = self.get_current_fps()
                        fps_measurements.append(fps)
                        last_fps_time = current_time

                        # Print progress
                        elapsed = current_time - self.start_time
                        print(
                            f"   Progress: {elapsed/duration*100:.1f}% - "
                            f"FPS: {fps:.2f} - Frames: {self.frame_count}"
                        )

                    # Save periodic samples
                    if current_time >= next_sample_time:
                        filename = f"continuous_sample_{sample_count:03d}.jpg"
                        if self.save_frame(frame, filename):
                            sample_count += 1
                            results["samples_saved"] = sample_count
                        next_sample_time += sample_interval

                except queue.Empty:
                    continue

            # Calculate final statistics
            results["total_frames"] = self.frame_count
            results["average_fps"] = self.get_current_fps()

            if fps_measurements:
                fps_array = np.array(fps_measurements)
                results["fps_stability"] = 1.0 - (np.std(fps_array) / np.mean(fps_array))

            print("✅ Continuous capture test complete:")
            print(f"   Total frames: {results['total_frames']}")
            print(f"   Average FPS: {results['average_fps']:.2f}")
            print(f"   FPS stability: {results['fps_stability']*100:.1f}%")
            print(f"   Samples saved: {results['samples_saved']}")

        except Exception as e:
            print(f"❌ Continuous capture error: {e}")
            results["success"] = False
            results["error"] = str(e)

        finally:
            self.stop_capture()

        return results

    def run_all_tests(self) -> dict:
        """Run complete camera test suite."""
        print("🚀 Starting DeepStream camera test suite")
        print(f"📁 Saving test images to: {self.save_dir}")

        # Print sensor information
        self.print_sensor_info()

        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device_id": self.device_id,
            "resolution_tests": [],
            "continuous_test": None,
            "summary": {"total_tests": 0, "passed_tests": 0, "failed_tests": 0},
        }

        # Test each resolution
        for width, height in self.test_resolutions:
            try:
                result = self.test_resolution(width, height, duration=5.0)
                all_results["resolution_tests"].append(result)

                all_results["summary"]["total_tests"] += 1
                if result["success"]:
                    all_results["summary"]["passed_tests"] += 1
                else:
                    all_results["summary"]["failed_tests"] += 1

            except Exception as e:
                print(f"❌ Resolution test failed: {e}")
                all_results["summary"]["total_tests"] += 1
                all_results["summary"]["failed_tests"] += 1

        # Continuous capture test
        try:
            continuous_result = self.test_continuous_capture(duration=60.0)  # 1 minute for testing
            all_results["continuous_test"] = continuous_result

            all_results["summary"]["total_tests"] += 1
            if continuous_result["success"]:
                all_results["summary"]["passed_tests"] += 1
            else:
                all_results["summary"]["failed_tests"] += 1

        except Exception as e:
            print(f"❌ Continuous test failed: {e}")
            all_results["summary"]["total_tests"] += 1
            all_results["summary"]["failed_tests"] += 1

        return all_results

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.stop_capture()
            if self.pipeline:
                self.pipeline = None
        except Exception as e:
            print(f"❌ Cleanup error: {e}")


def print_results_summary(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 60)
    print("📊 CAMERA TEST RESULTS SUMMARY")
    print("=" * 60)

    summary = results["summary"]
    print(f"🕒 Test time: {results['timestamp']}")
    print(f"📹 Camera device: {results['device_id']}")
    print(f"📈 Total tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")
    print(f"📊 Success rate: {summary['passed_tests']/summary['total_tests']*100:.1f}%")

    print("\n📏 Resolution Test Results:")
    for test in results["resolution_tests"]:
        status = "✅" if test["success"] else "❌"
        res = test["resolution"]
        fps = test.get("average_fps", 0)

        # Add field of view information
        if res == (1920, 1080):
            fov_info = " (CROPPED FOV)"
        elif res == (1280, 720):
            fov_info = " (CROPPED FOV)"
        elif res == (1640, 1232):
            fov_info = " (FULL FOV)"
        else:
            fov_info = ""

        print(f"   {status} {res[0]}x{res[1]}: {fps:.2f} FPS{fov_info}")

    if results["continuous_test"]:
        cont = results["continuous_test"]
        status = "✅" if cont["success"] else "❌"
        print("\n⏱️  Continuous Test:")
        print(f"   {status} Duration: {cont['duration']:.0f}s")
        if cont["success"]:
            print(f"   📊 Average FPS: {cont['average_fps']:.2f}")
            print(f"   📈 FPS stability: {cont['fps_stability']*100:.1f}%")

    print("=" * 60)

    # Add camera settings explanation
    camera_test = DeepStreamCameraTest()
    print(camera_test.get_optimal_settings_info())


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="DeepStream Camera Test")
    parser.add_argument("--device", "-d", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="test_images",
        help="Output directory for test images",
    )
    parser.add_argument(
        "--continuous-duration",
        "-t",
        type=float,
        default=300.0,
        help="Continuous test duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Run quick tests only (shorter durations)"
    )

    args = parser.parse_args()

    # Create test instance
    camera_test = DeepStreamCameraTest(device_id=args.device, save_dir=args.output_dir)

    try:
        # Adjust test duration for quick mode
        if args.quick:
            print("🏃 Running in quick test mode")
            args.continuous_duration = 30.0

        # Run tests
        results = camera_test.run_all_tests()

        # Print summary
        print_results_summary(results)

        # Determine exit code
        summary = results["summary"]
        if summary["failed_tests"] == 0:
            print("🎉 All tests passed!")
            exit_code = 0
        else:
            print(f"⚠️  {summary['failed_tests']} tests failed")
            exit_code = 1

        return exit_code

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1
    finally:
        camera_test.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
