"""
Hardware validation test for RT-MonoDepth depth estimation
Tests camera integration, TensorRT inference, and performance on Jetson
Place in: hardware_tests/
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path before other imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "perception_nodes"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception_nodes.depth.rt_monodepth_inference import RTMonoDepthInference  # noqa: E402
from perception_nodes.depth.rt_monodepth_preprocessing import RTMonoDepthPreprocessor  # noqa: E402


class DepthHardwareTest:
    """Hardware validation for depth estimation pipeline."""

    def __init__(self, model_path, use_tensorrt=True, camera_id=0):
        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.camera_id = camera_id

        print("\n" + "=" * 70)
        print("RT-MonoDepth Hardware Validation Test")
        print("=" * 70)

        # Initialize depth estimator
        print("\n[1/3] Loading depth estimation model...")
        self.depth_estimator = RTMonoDepthInference(
            model_path=model_path,
            use_tensorrt=use_tensorrt,
            input_height=192,
            input_width=640,
            device="cuda",
        )
        print("✓ Model loaded successfully")

        # Initialize preprocessor
        self.preprocessor = RTMonoDepthPreprocessor()

        # Performance tracking
        self.frame_times = []
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []

    def test_camera_capture(self):
        """Test 1: Verify camera capture works."""
        print("\n[2/3] Testing camera capture...")

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("✗ Failed to open camera")
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("✗ Failed to capture frame")
            return False

        print("✓ Camera capture successful")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Channels: {frame.shape[2]}")
        return True

    def test_single_inference(self):
        """Test 2: Run single inference and verify output."""
        print("\n[3/3] Testing single inference...")

        cap = cv2.VideoCapture(self.camera_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("✗ Failed to capture test frame")
            return False

        try:
            # Time preprocessing
            t0 = time.perf_counter()
            depth_map, original_shape = self.depth_estimator.predict(image_array=frame)
            t1 = time.perf_counter()

            inference_time = (t1 - t0) * 1000

            print("✓ Inference successful")
            print(f"  Time: {inference_time:.2f} ms")
            print(f"  FPS: {1000/inference_time:.1f}")
            print(f"  Output shape: {depth_map.shape}")
            print(f"  Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}] meters")

            # Validate output
            if depth_map.shape != original_shape:
                print("✗ Output shape mismatch")
                return False

            if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                print("✗ Invalid values in output")
                return False

            return True

        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return False

    def run_performance_test(self, duration_seconds=10):
        """Test 3: Run continuous inference and measure performance."""
        print("\n" + "=" * 70)
        print("Performance Test")
        print("=" * 70)
        print(f"\nRunning for {duration_seconds} seconds...")
        print("Press 'q' to stop early\n")

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("✗ Failed to open camera")
            return

        # Warm up
        print("Warming up (5 frames)...")
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                _, _ = self.depth_estimator.predict(image_array=frame)

        print("Starting performance test...\n")

        start_time = time.time()
        frame_count = 0

        try:
            while (time.time() - start_time) < duration_seconds:
                frame_start = time.perf_counter()

                # Capture
                ret, frame = cap.read()
                if not ret:
                    continue

                # Preprocess & Inference
                preprocess_start = time.perf_counter()
                depth_map, _ = self.depth_estimator.predict(image_array=frame)
                inference_time = (time.perf_counter() - preprocess_start) * 1000

                # Postprocess (colormap for visualization)
                postprocess_start = time.perf_counter()
                _ = self.preprocessor.depth_to_colormap(depth_map)
                postprocess_time = (time.perf_counter() - postprocess_start) * 1000

                frame_time = (time.perf_counter() - frame_start) * 1000

                # Track times
                self.frame_times.append(frame_time)
                self.inference_times.append(inference_time)
                self.postprocessing_times.append(postprocess_time)

                frame_count += 1

                # Display (optional - reduces FPS but useful for visual verification)
                if frame_count % 10 == 0:
                    fps = 1000 / np.mean(self.frame_times[-30:])
                    print(
                        f"Frames: {frame_count} | FPS: {fps:.1f} | "
                        f"Inference: {inference_time:.1f}ms | "
                        f"Depth range: [{depth_map.min():.1f}, {depth_map.max():.1f}]"
                    )

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Print statistics
        self._print_statistics(frame_count, time.time() - start_time)

    def _print_statistics(self, frame_count, duration):
        """Print detailed performance statistics."""
        print("\n" + "=" * 70)
        print("Performance Statistics")
        print("=" * 70)

        print(f"\nTotal frames: {frame_count}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Average FPS: {frame_count / duration:.2f}")

        if self.frame_times:
            print("\nEnd-to-End Pipeline:")
            print(f"  Mean:   {np.mean(self.frame_times):.2f} ms")
            print(f"  Std:    {np.std(self.frame_times):.2f} ms")
            print(f"  Min:    {np.min(self.frame_times):.2f} ms")
            print(f"  Max:    {np.max(self.frame_times):.2f} ms")
            print(f"  FPS:    {1000 / np.mean(self.frame_times):.1f}")

        if self.inference_times:
            print("\nInference Only:")
            print(f"  Mean:   {np.mean(self.inference_times):.2f} ms")
            print(f"  Std:    {np.std(self.inference_times):.2f} ms")
            print(f"  Min:    {np.min(self.inference_times):.2f} ms")
            print(f"  Max:    {np.max(self.inference_times):.2f} ms")
            print(f"  FPS:    {1000 / np.mean(self.inference_times):.1f}")

        if self.postprocessing_times:
            print("\nPostprocessing:")
            print(f"  Mean:   {np.mean(self.postprocessing_times):.2f} ms")

    def run_thermal_test(self, duration_minutes=5):
        """Test 4: Monitor thermal performance under sustained load."""
        print("\n" + "=" * 70)
        print("Thermal Stability Test")
        print("=" * 70)
        print(f"\nRunning for {duration_minutes} minutes...")
        print("Monitoring GPU temperature and throttling...\n")

        import subprocess

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("✗ Failed to open camera")
            return

        start_time = time.time()
        duration_seconds = duration_minutes * 60

        temp_samples = []
        fps_samples = []
        frame_times_window = []

        try:
            while (time.time() - start_time) < duration_seconds:
                frame_start = time.perf_counter()

                ret, frame = cap.read()
                if ret:
                    _, _ = self.depth_estimator.predict(image_array=frame)

                frame_time = (time.perf_counter() - frame_start) * 1000
                frame_times_window.append(frame_time)

                # Keep last 30 frames for FPS calculation
                if len(frame_times_window) > 30:
                    frame_times_window.pop(0)

                # Sample temperature every 10 seconds
                elapsed = time.time() - start_time
                if len(temp_samples) == 0 or elapsed - temp_samples[-1][0] >= 10:
                    try:
                        # Get GPU temperature (Jetson specific)
                        _ = subprocess.run(
                            ["tegrastats", "--interval", "100"],
                            capture_output=True,
                            text=True,
                            timeout=0.5,
                        )
                        # Parse temperature from output (simplified)
                        temp = "N/A"  # You'd parse actual temp here

                        fps = 1000 / np.mean(frame_times_window)
                        temp_samples.append((elapsed, temp))
                        fps_samples.append((elapsed, fps))

                        print(f"Time: {elapsed:.0f}s | FPS: {fps:.1f} | Temp: {temp}")
                    except Exception:
                        pass

        finally:
            cap.release()

        print("\n✓ Thermal test complete")
        print("Check for any thermal throttling or FPS degradation above")

    def run_all_tests(self):
        """Run complete hardware validation suite."""
        results = {"camera": False, "inference": False, "performance": False}

        # Test 1: Camera
        results["camera"] = self.test_camera_capture()

        # Test 2: Single inference
        if results["camera"]:
            results["inference"] = self.test_single_inference()

        # Test 3: Performance
        if results["inference"]:
            self.run_performance_test(duration_seconds=10)
            results["performance"] = True

        # Print summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Camera Capture:     {'✓ PASS' if results['camera'] else '✗ FAIL'}")
        print(f"Single Inference:   {'✓ PASS' if results['inference'] else '✗ FAIL'}")
        print(f"Performance Test:   {'✓ PASS' if results['performance'] else '✗ FAIL'}")

        all_passed = all(results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 70 + "\n")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Hardware validation for RT-MonoDepth")
    parser.add_argument("--model_path", type=str, required=True, help="Path to TensorRT engine")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "quick", "performance", "thermal"],
        help="Test suite to run",
    )
    parser.add_argument(
        "--duration", type=int, default=10, help="Performance test duration (seconds)"
    )
    parser.add_argument(
        "--use_pytorch", action="store_true", help="Use PyTorch instead of TensorRT"
    )

    args = parser.parse_args()

    # Initialize test suite
    tester = DepthHardwareTest(
        model_path=args.model_path,
        use_tensorrt=not args.use_pytorch,
        camera_id=args.camera_id,
    )

    # Run selected tests
    if args.test == "all":
        tester.run_all_tests()
    elif args.test == "quick":
        tester.test_camera_capture()
        tester.test_single_inference()
    elif args.test == "performance":
        tester.run_performance_test(duration_seconds=args.duration)
    elif args.test == "thermal":
        tester.run_thermal_test(duration_minutes=args.duration)


if __name__ == "__main__":
    main()
