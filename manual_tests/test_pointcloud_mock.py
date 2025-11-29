#!/usr/bin/env python3
"""
Test Point Cloud Generator with Mock Data
Tests the pointcloud_generator node without requiring actual camera/depth data.

This script validates:
- ROS2 node initialization
- Message synchronization
- Point cloud generation
- Camera calibration loading
"""

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


class MockDepthPublisher(Node):
    """Publishes mock depth maps and RGB images for testing."""

    def __init__(self):
        super().__init__("mock_depth_publisher")

        self.depth_pub = self.create_publisher(Image, "/perception/depth", 10)
        self.rgb_pub = self.create_publisher(Image, "/camera/undistorted", 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, "/camera_info", 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_data)  # 10 Hz

        self.get_logger().info("Mock depth publisher started")

    def publish_data(self):
        """Publish synchronized depth, RGB, and camera info."""
        stamp = self.get_clock().now().to_msg()

        # Create mock depth map (gradient from near to far)
        depth = np.linspace(0.5, 5.0, 480 * 640).reshape(480, 640).astype(np.float32)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = "camera_link"

        # Create mock RGB image (colored gradient)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Blue gradient
        rgb[:, :, 1] = np.linspace(0, 255, 480)[:, np.newaxis].astype(np.uint8)  # Green gradient
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="bgr8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "camera_link"

        # Create camera info
        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = "camera_link"
        camera_info.width = 640
        camera_info.height = 480
        # Camera matrix K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

        # Publish all
        self.depth_pub.publish(depth_msg)
        self.rgb_pub.publish(rgb_msg)
        self.camera_info_pub.publish(camera_info)


class PointCloudMonitor(Node):
    """Monitors point cloud output."""

    def __init__(self):
        super().__init__("pointcloud_monitor")

        self.subscription = self.create_subscription(
            PointCloud2, "/perception/pointcloud", self.pointcloud_callback, 10
        )

        self.pointcloud_count = 0
        self.last_pointcloud_time = None

        self.get_logger().info("Point cloud monitor started")

    def pointcloud_callback(self, msg):
        """Process point cloud messages."""
        self.pointcloud_count += 1
        current_time = time.time()

        if self.last_pointcloud_time is not None:
            latency = current_time - self.last_pointcloud_time
            fps = 1.0 / latency if latency > 0 else 0

            self.get_logger().info(
                f"Received point cloud #{self.pointcloud_count}: "
                f"{msg.width} points, {fps:.1f} Hz, "
                f"frame: {msg.header.frame_id}"
            )

            # Print point cloud info
            field_names = [field.name for field in msg.fields]
            self.get_logger().info(f'  Fields: {", ".join(field_names)}')
            self.get_logger().info(f"  Size: {len(msg.data)} bytes")

        self.last_pointcloud_time = current_time


def main():
    """Run mock testing."""
    rclpy.init()

    # Create nodes
    publisher = MockDepthPublisher()
    monitor = PointCloudMonitor()

    # Note: pointcloud_generator node should be launched separately
    print("\n" + "=" * 70)
    print("Mock Depth/RGB Publisher Started")
    print("=" * 70)
    print("\nThis script publishes mock data to:")
    print("  - /perception/depth (depth maps)")
    print("  - /camera/undistorted (RGB images)")
    print("  - /camera_info (camera calibration)")
    print("\nTo test the point cloud generator:")
    print("\n1. In another terminal, run:")
    print("   cd ~/repos/local-ai-robot-assistant")
    print("   source src/install/setup.bash")
    print("   ros2 run perception_nodes pointcloud_generator")
    print("\n2. Watch point cloud output in this terminal")
    print("\n3. To visualize point cloud:")
    print("   rviz2")
    print("   Then add PointCloud2 display for /perception/pointcloud")
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
