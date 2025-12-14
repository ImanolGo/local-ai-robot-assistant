#!/usr/bin/env python3
"""
Manual Test Script for Depth Estimation Accuracy
------------------------------------------------
This script allows manual verification of the depth estimation node's accuracy.

Usage:
    python3 test_depth_accuracy.py --engine <path_to_engine> [--image <path_to_image>] \
        [--distance <true_distance_m>]

Features:
- Loads TensorRT engine
- Runs inference on image (file or camera)
- Visualizes depth map
- Calculates error if true distance is provided
- interactive mode for multiple measurements
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent / "src" / "perception_nodes"))
try:
    from perception_nodes.depth_anything_v2_trt import DepthAnythingV2TRT
except ImportError:
    print("Error: Could not import perception_nodes. Make sure you have built the package.")
    sys.exit(1)


def calculate_metrics(predicted_depth, true_distance, roi_size=20):
    """Calculate error metrics for center region."""
    h, w = predicted_depth.shape
    cy, cx = h // 2, w // 2

    # Extract center region
    center_roi = predicted_depth[cy - roi_size : cy + roi_size, cx - roi_size : cx + roi_size]
    avg_pred = np.mean(center_roi)

    error = avg_pred - true_distance
    abs_error = abs(error)
    rel_error = abs_error / true_distance if true_distance > 0 else 0

    return {
        "predicted": avg_pred,
        "true": true_distance,
        "error": error,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Accuracy Test")
    parser.add_argument(
        "--engine",
        default="models/depth_trt/depth_anything_v2_small.trt",
        help="Path to TensorRT engine",
    )
    parser.add_argument("--image", help="Path to test image (default: use camera)")
    parser.add_argument("--distance", type=float, help="True distance to object in center (meters)")
    parser.add_argument(
        "--max-depth", type=float, default=10.0, help="Maximum depth range (meters)"
    )

    args = parser.parse_args()

    # Check engine
    if not Path(args.engine).exists():
        print(f"Error: Engine not found at {args.engine}")
        return

    # Load model
    print(f"Loading model from {args.engine}...")
    try:
        model = DepthAnythingV2TRT(args.engine)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get image
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image {args.image}")
            return
    else:
        print("Opening camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Could not read frame")
            return

    print(f"Processing image {frame.shape}...")

    # Run inference
    start = time.time()
    raw_depth = model.infer(frame)
    inference_time = (time.time() - start) * 1000

    # Convert to metric (approximate)
    # Note: This matches the node's logic
    depth_normalized = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-8)
    metric_depth = depth_normalized * args.max_depth

    # Visualize
    depth_vis = model.visualize_depth(raw_depth)

    # Show results
    print(f"\nInference Time: {inference_time:.1f} ms")
    print(f"Depth Range: {metric_depth.min():.2f}m - {metric_depth.max():.2f}m")

    if args.distance:
        metrics = calculate_metrics(metric_depth, args.distance)
        print("\nAccuracy Metrics (Center Region):")
        print(f"  True Distance:      {metrics['true']:.3f} m")
        print(f"  Predicted Distance: {metrics['predicted']:.3f} m")
        print(f"  Absolute Error:     {metrics['abs_error']:.3f} m")
        print(f"  Relative Error:     {metrics['rel_error']*100:.1f}%")

        if metrics["rel_error"] < 0.15:
            print("\n[PASS] Accuracy within 15% tolerance")
        else:
            print("\n[FAIL] Accuracy outside 15% tolerance")

    # Display
    cv2.imshow("RGB", frame)
    cv2.imshow("Depth", depth_vis)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    model.cleanup()


if __name__ == "__main__":
    main()
