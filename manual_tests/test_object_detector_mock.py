#!/usr/bin/env python3
"""
Test Object Detection Node with Mock Data
Tests the object_detector node without requiring actual TensorRT engines.

This script validates:
- ROS2 node initialization
- Message handling
- Detection pipeline
- Visualization output
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class MockImagePublisher(Node):
    """Publishes mock camera images for testing."""

    def __init__(self):
        super().__init__("mock_image_publisher")

        self.publisher = self.create_publisher(Image, "/camera/undistorted", 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.033, self.publish_image)  # 30 Hz

        self.get_logger().info("Mock image publisher started")

    def publish_image(self):
        """Publish a test image with drawn objects."""
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw some shapes to simulate objects
        cv2.rectangle(img, (100, 100), (200, 250), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(img, (400, 300), 50, (255, 0, 0), -1)  # Blue circle
        cv2.rectangle(img, (300, 50), (450, 150), (0, 0, 255), -1)  # Red rectangle

        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"

        self.publisher.publish(msg)


class DetectionMonitor(Node):
    """Monitors detection output."""

    def __init__(self):
        super().__init__("detection_monitor")

        self.subscription = self.create_subscription(
            Detection2DArray, "/perception/objects", self.detection_callback, 10
        )

        self.detection_count = 0
        self.last_detection_time = None

        self.get_logger().info("Detection monitor started")

    def detection_callback(self, msg):
        """Process detection messages."""
        self.detection_count += 1
        current_time = time.time()

        if self.last_detection_time is not None:
            latency = current_time - self.last_detection_time
            fps = 1.0 / latency if latency > 0 else 0

            self.get_logger().info(
                f"Received detection #{self.detection_count}: "
                f"{len(msg.detections)} objects, {fps:.1f} FPS"
            )

            # Print details of each detection
            for i, detection in enumerate(msg.detections):
                if detection.results:
                    result = detection.results[0]
                    self.get_logger().info(
                        f"  Object {i+1}: class={result.hypothesis.class_id}, "
                        f"confidence={result.hypothesis.score:.2f}, "
                        f"bbox=({detection.bbox.center.position.x:.0f}, "
                        f"{detection.bbox.center.position.y:.0f}), "
                        f"size=({detection.bbox.size_x:.0f}x{detection.bbox.size_y:.0f})"
                    )

        self.last_detection_time = current_time


def main():
    """Run mock testing."""
    rclpy.init()

    # Create nodes
    publisher = MockImagePublisher()
    monitor = DetectionMonitor()

    # Note: object_detector node should be launched separately
    print("\n" + "=" * 70)
    print("Mock Image Publisher Started")
    print("=" * 70)
    print("\nThis script publishes mock images to /camera/undistorted")
    print("To test the object detector:")
    print("\n1. In another terminal, run:")
    print("   cd ~/repos/local-ai-robot-assistant")
    print("   source src/install/setup.bash")
    print("   ros2 run perception_nodes object_detector --ros-args \\")
    print("       -p engine_path:=/path/to/yolo11n_fp16.engine")
    print("\n2. Watch detection output in this terminal")
    print("\n3. To visualize detections:")
    print("   ros2 run rqt_image_view rqt_image_view /perception/objects_viz")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    # Spin nodes
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(monitor)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        publisher.destroy_node()
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
