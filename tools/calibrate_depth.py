#!/usr/bin/env python3
"""
Depth Calibration Tool for SLAM.

Helps calibrate the monocular depth estimation (Depth Anything V2) against
real-world distances. This is critical for RTAB-Map SLAM accuracy.

Usage:
    1. Place the robot at a known distance from a flat wall
    2. Run: python3 tools/calibrate_depth.py
    3. Follow the prompts to record measurements
    4. The tool outputs depth_scale and depth_offset parameters

The calibrated parameters go into:
    - config/perception_config.yaml (depth_estimation.calibration section)
    - Or as ROS2 node parameters: depth_scale, depth_offset
"""

import sys
import time
from typing import List, Tuple

import numpy as np

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    print("ERROR: ROS2 packages not found. Source your ROS2 environment first.")
    print("  source ros2_venv.sh")
    sys.exit(1)


class DepthCalibrator(Node):
    """Collects depth measurements and computes calibration parameters."""

    def __init__(self):
        super().__init__("depth_calibrator")

        self.bridge = CvBridge()
        self.latest_depth: np.ndarray = None

        # Subscribe to depth topic
        self.depth_sub = self.create_subscription(
            Image, "/perception/depth", self._depth_callback, 10
        )
        self.get_logger().info("Waiting for depth data on /perception/depth...")

    def _depth_callback(self, msg: Image) -> None:
        """Store latest depth frame."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def get_center_depth(self, roi_fraction: float = 0.1) -> float:
        """Get average depth in the center region of the image.

        Args:
            roi_fraction: Fraction of image width/height for the center ROI.

        Returns:
            Average depth value in meters (from the depth node's current scaling).
        """
        if self.latest_depth is None:
            return -1.0

        h, w = self.latest_depth.shape[:2]
        margin_h = int(h * roi_fraction / 2)
        margin_w = int(w * roi_fraction / 2)
        cy, cx = h // 2, w // 2

        roi = self.latest_depth[cy - margin_h : cy + margin_h, cx - margin_w : cx + margin_w]

        # Filter out zeros and invalid values
        valid = roi[(roi > 0.01) & (roi < 100.0)]
        if len(valid) == 0:
            return -1.0

        return float(np.median(valid))


def collect_measurements(calibrator: DepthCalibrator) -> List[Tuple[float, float]]:
    """Interactive measurement collection.

    Returns:
        List of (actual_distance_m, measured_depth_value) tuples.
    """
    measurements = []
    suggested_distances = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("\n" + "=" * 60)
    print("  DEPTH CALIBRATION TOOL")
    print("=" * 60)
    print("\nInstructions:")
    print("  1. Point the robot camera at a flat wall")
    print("  2. Place the robot at each distance and press Enter")
    print("  3. Measure the real distance with a tape measure")
    print("  4. Enter the actual distance when prompted")
    print("  5. At least 3 measurements needed, 5+ recommended")
    print("\nSuggested distances:", suggested_distances)
    print("Type 'done' when finished, 'skip' to skip a measurement.\n")

    for i, dist in enumerate(suggested_distances):
        response = input(
            f"\n[{i + 1}/{len(suggested_distances)}] "
            f"Place robot {dist}m from wall, press Enter (or 'done'/'skip'): "
        )

        if response.strip().lower() == "done":
            break
        if response.strip().lower() == "skip":
            continue

        # Wait for fresh depth data
        print("  Collecting depth samples (2 seconds)...")
        samples = []
        start = time.time()
        while time.time() - start < 2.0:
            rclpy.spin_once(calibrator, timeout_sec=0.1)
            d = calibrator.get_center_depth()
            if d > 0:
                samples.append(d)

        if not samples:
            print("  ⚠ No valid depth data received! Check the perception pipeline.")
            continue

        measured = float(np.median(samples))
        std = float(np.std(samples))

        # Ask for actual distance
        actual_str = input(
            f"  Measured depth: {measured:.3f}m (std: {std:.3f}m)"
            f"\n  Enter ACTUAL distance in meters [{dist}]: "
        ).strip()
        actual = float(actual_str) if actual_str else dist

        measurements.append((actual, measured))
        print(f"  ✓ Recorded: actual={actual:.2f}m, measured={measured:.3f}m")

    return measurements


def compute_calibration(
    measurements: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Fit a linear model: actual = measured * scale + offset.

    Args:
        measurements: List of (actual, measured) pairs.

    Returns:
        (scale, offset) calibration parameters.
    """
    if len(measurements) < 2:
        print("ERROR: Need at least 2 measurements for calibration.")
        return 1.0, 0.0

    actuals = np.array([m[0] for m in measurements])
    measured = np.array([m[1] for m in measurements])

    # Linear regression: actual = measured * scale + offset
    A = np.vstack([measured, np.ones(len(measured))]).T
    result = np.linalg.lstsq(A, actuals, rcond=None)
    scale, offset = result[0]

    # Calculate residuals
    predicted = measured * scale + offset
    residuals = actuals - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_error = float(np.max(np.abs(residuals)))

    print("\n" + "=" * 60)
    print("  CALIBRATION RESULTS")
    print("=" * 60)
    print(f"\n  depth_scale:  {scale:.4f}")
    print(f"  depth_offset: {offset:.4f} meters")
    print(f"\n  RMSE:      {rmse:.3f} m")
    print(f"  Max Error: {max_error:.3f} m")
    print("\n  Measurements:")
    for actual, meas in measurements:
        pred = meas * scale + offset
        err = actual - pred
        print(
            f"    actual={actual:.2f}m, measured={meas:.3f}m, "
            f"predicted={pred:.3f}m, error={err:+.3f}m"
        )

    print("\n  Add to config/perception_config.yaml:")
    print("    calibration:")
    print(f"      depth_scale: {scale:.4f}")
    print(f"      depth_offset: {offset:.4f}")

    print("\n  Or set as ROS2 parameters:")
    print(f"    ros2 param set /depth_estimation_node depth_scale {scale:.4f}")
    print(f"    ros2 param set /depth_estimation_node depth_offset {offset:.4f}")
    print("=" * 60)

    return scale, offset


def main():
    rclpy.init()
    calibrator = DepthCalibrator()

    # Wait for first depth message
    print("Waiting for depth data...")
    timeout = 10.0
    start = time.time()
    while calibrator.latest_depth is None and time.time() - start < timeout:
        rclpy.spin_once(calibrator, timeout_sec=0.5)

    if calibrator.latest_depth is None:
        print("ERROR: No depth data received after 10 seconds.")
        print("  Make sure the perception pipeline is running:")
        print("  ros2 launch perception_nodes depth_estimation_launch.py")
        calibrator.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print(f"✓ Receiving depth data (shape: {calibrator.latest_depth.shape})")

    try:
        measurements = collect_measurements(calibrator)
        if measurements:
            compute_calibration(measurements)
        else:
            print("No measurements collected.")
    except KeyboardInterrupt:
        print("\nCalibration cancelled.")
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
