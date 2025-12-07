#!/usr/bin/env python3
"""
Test script for Depth Anything V2 pipeline implementation.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add the module path
sys.path.append(
    "/home/imanolgo/repos/local-ai-robot-assistant/src/perception_nodes/perception_nodes"
)

from depth_estimation_node_pipeline import DepthAnythingV2Pipeline  # noqa E402


def test_depth_pipeline():
    """Test the depth estimation pipeline."""
    print("🔬 Testing Depth Anything V2 Pipeline Implementation")

    # Initialize pipeline
    model_dir = "/home/imanolgo/repos/local-ai-robot-assistant/models/depth_trt"

    try:
        print(f"📁 Loading model from: {model_dir}")
        depth_estimator = DepthAnythingV2Pipeline(model_dir)
        print("✅ Pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return False

    # Test with dummy image
    print("\n🖼️ Testing with dummy image...")

    # Create test image (640x480 RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")

    # Run inference
    start_time = time.time()
    depth_map = depth_estimator.predict(test_image)
    inference_time = time.time() - start_time

    print(f"✅ Inference completed in {inference_time:.3f}s")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Test performance with multiple frames
    print("\n⚡ Testing performance with 10 frames...")
    times = []

    for i in range(10):
        # Create slightly different image each time
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        depth_map = depth_estimator.predict(test_image)
        inference_time = time.time() - start_time
        times.append(inference_time)

        print(f"Frame {i+1}: {inference_time:.3f}s")

    # Calculate statistics
    avg_time = np.mean(times)
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0

    print("\n📊 Performance Summary:")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")

    # Get pipeline statistics
    stats = depth_estimator.get_stats()
    print("\n📈 Pipeline Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\n✅ All tests completed successfully!")
    return True


def test_with_real_image():
    """Test with a real image if available."""
    print("\n🖼️ Testing with real image (if available)...")

    # Try to capture from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("📷 No camera available, skipping real image test")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("📷 Failed to capture image from camera")
        return

    print(f"📷 Captured image: {frame.shape}")

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize pipeline
    model_dir = "/home/imanolgo/repos/local-ai-robot-assistant/models/depth_trt"
    depth_estimator = DepthAnythingV2Pipeline(model_dir)

    # Run inference
    start_time = time.time()
    depth_map = depth_estimator.predict(rgb_frame)
    inference_time = time.time() - start_time

    print(f"✅ Real image inference: {inference_time:.3f}s")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Save results if possible
    try:
        output_dir = Path("/home/imanolgo/repos/local-ai-robot-assistant/test_output")
        output_dir.mkdir(exist_ok=True)

        # Save input image
        cv2.imwrite(str(output_dir / "test_input.jpg"), frame)

        # Save depth map (normalized for visualization)
        depth_vis = (
            (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        ).astype(np.uint8)
        cv2.imwrite(str(output_dir / "test_depth.jpg"), depth_vis)

        print(f"💾 Results saved to: {output_dir}")

    except Exception as e:
        print(f"⚠️ Could not save results: {e}")


def main():
    """Main test function."""
    print("🚀 Starting Depth Anything V2 Pipeline Tests\n")

    # Test 1: Basic pipeline functionality
    if not test_depth_pipeline():
        print("❌ Basic pipeline test failed")
        return 1

    # Test 2: Real image test (optional)
    try:
        test_with_real_image()
    except Exception as e:
        print(f"⚠️ Real image test failed: {e}")

    print("\n🎉 All tests completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
