#!/usr/bin/env python3
"""
Unit tests for Point Cloud Generator Node.

Tests:
- Camera intrinsics loading
- Point cloud generation from depth maps
- RGB color mapping
- Depth filtering and validation
- PointCloud2 message creation
- Performance benchmarking
"""

import unittest

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, PointCloud2


class TestPointCloudGenerator(unittest.TestCase):
    """Test cases for Point Cloud Generator node."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test environment."""
        self.test_depth = self._create_test_depth()
        self.test_rgb = self._create_test_rgb()
        self.camera_info = self._create_camera_info()

    def _create_test_depth(self) -> np.ndarray:
        """Create a test depth map."""
        # Create depth map with gradient (closer at top, farther at bottom)
        depth = np.linspace(0.5, 5.0, 480 * 640).reshape(480, 640).astype(np.float32)
        return depth

    def _create_test_rgb(self) -> np.ndarray:
        """Create a test RGB image."""
        # Create colored image (blue gradient)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Blue channel
        return rgb

    def _create_camera_info(self) -> CameraInfo:
        """Create test camera info message."""
        msg = CameraInfo()
        msg.header.frame_id = "camera_link"
        msg.width = 640
        msg.height = 480

        # Camera matrix K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        msg.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]

        return msg

    def test_node_initialization(self):
        """Test node initialization."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()

        # Check node was created
        self.assertIsNotNone(node)
        self.assertEqual(node.get_name(), "pointcloud_generator")

        # Check parameters
        self.assertIsNotNone(node.depth_min)
        self.assertIsNotNone(node.depth_max)
        self.assertIsNotNone(node.downsample)

        # Clean up
        node.destroy_node()

    def test_camera_info_callback(self):
        """Test camera intrinsics extraction."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()

        # Initially, intrinsics should be None
        self.assertIsNone(node.fx)

        # Send camera info
        node.camera_info_callback(self.camera_info)

        # Check intrinsics were extracted
        self.assertEqual(node.fx, 500.0)
        self.assertEqual(node.fy, 500.0)
        self.assertEqual(node.cx, 320.0)
        self.assertEqual(node.cy, 240.0)
        self.assertEqual(node.camera_frame, "camera_link")

        # Clean up
        node.destroy_node()

    def test_pointcloud_generation(self):
        """Test 3D point cloud generation."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()

        # Set camera intrinsics
        node.camera_info_callback(self.camera_info)

        # Generate point cloud
        points = node.generate_pointcloud(self.test_depth, self.test_rgb)

        # Check point cloud shape (N x 6: x, y, z, r, g, b)
        self.assertEqual(points.shape[1], 6)
        self.assertGreater(points.shape[0], 0)

        # Check all points are within depth range
        z_values = points[:, 2]
        self.assertTrue(np.all(z_values >= node.depth_min))
        self.assertTrue(np.all(z_values <= node.depth_max))

        # Check RGB values are in valid range [0, 255]
        rgb_values = points[:, 3:6]
        self.assertTrue(np.all(rgb_values >= 0))
        self.assertTrue(np.all(rgb_values <= 255))

        # Clean up
        node.destroy_node()

    def test_depth_filtering(self):
        """Test depth range filtering."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()
        node.depth_min = 1.0
        node.depth_max = 3.0

        # Set camera intrinsics
        node.camera_info_callback(self.camera_info)

        # Create depth with values outside range
        depth = np.ones((100, 100), dtype=np.float32)
        depth[0:30, :] = 0.5  # Too close
        depth[30:70, :] = 2.0  # In range
        depth[70:100, :] = 5.0  # Too far

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        # Generate point cloud
        points = node.generate_pointcloud(depth, rgb)

        # Check all points are in valid range
        z_values = points[:, 2]
        self.assertTrue(np.all(z_values >= 1.0))
        self.assertTrue(np.all(z_values <= 3.0))

        # Clean up
        node.destroy_node()

    def test_downsampling(self):
        """Test point cloud downsampling."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        # Create nodes with different downsample factors
        node1 = PointCloudGenerator()
        node1.downsample = 1  # No downsampling
        node1.camera_info_callback(self.camera_info)

        node2 = PointCloudGenerator()
        node2.downsample = 2  # 2x downsampling
        node2.camera_info_callback(self.camera_info)

        # Generate point clouds
        points1 = node1.generate_pointcloud(self.test_depth, self.test_rgb)
        points2 = node2.generate_pointcloud(self.test_depth, self.test_rgb)

        # Downsampled version should have fewer points
        self.assertLess(points2.shape[0], points1.shape[0])

        # Should be approximately 4x fewer points (2x in each dimension)
        ratio = points1.shape[0] / points2.shape[0]
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 5.0)

        # Clean up
        node1.destroy_node()
        node2.destroy_node()

    def test_pointcloud2_message_creation(self):
        """Test PointCloud2 message creation."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()
        node.camera_info_callback(self.camera_info)

        # Generate point cloud
        points = node.generate_pointcloud(self.test_depth, self.test_rgb)

        # Create header
        from std_msgs.msg import Header

        header = Header()
        header.frame_id = "camera_link"
        header.stamp = node.get_clock().now().to_msg()

        # Create PointCloud2 message
        msg = node.create_pointcloud2_msg(points, header)

        # Check message properties
        self.assertEqual(msg.header.frame_id, "camera_link")
        self.assertEqual(msg.height, 1)
        self.assertEqual(msg.width, len(points))
        self.assertFalse(msg.is_dense)

        # Check fields (should have x, y, z, rgb)
        field_names = [field.name for field in msg.fields]
        self.assertIn("x", field_names)
        self.assertIn("y", field_names)
        self.assertIn("z", field_names)
        self.assertIn("rgb", field_names)

        # Clean up
        node.destroy_node()

    def test_synchronized_callback(self):
        """Test synchronized depth and RGB callback."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()
        node.camera_info_callback(self.camera_info)

        # Create ROS messages
        bridge = CvBridge()
        depth_msg = bridge.cv2_to_imgmsg(self.test_depth, encoding="32FC1")
        rgb_msg = bridge.cv2_to_imgmsg(self.test_rgb, encoding="bgr8")

        # Set up subscriber to capture published point clouds
        received_pointclouds = []

        def pointcloud_callback(msg):
            received_pointclouds.append(msg)

        test_sub = node.create_subscription(
            PointCloud2, "/perception/pointcloud", pointcloud_callback, 10
        )

        # Process synchronized messages
        node.synchronized_callback(depth_msg, rgb_msg)

        # Spin to process callbacks
        rclpy.spin_once(node, timeout_sec=1.0)

        # Check point cloud was published
        self.assertEqual(len(received_pointclouds), 1)

        # Clean up
        node.destroy_subscription(test_sub)
        node.destroy_node()

    def test_rgb_disabled(self):
        """Test point cloud generation without RGB."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()
        node.enable_rgb = False
        node.camera_info_callback(self.camera_info)

        # Generate point cloud
        points = node.generate_pointcloud(self.test_depth, self.test_rgb)

        # Should only have XYZ (3 columns)
        self.assertEqual(points.shape[1], 3)

        # Clean up
        node.destroy_node()


class TestPointCloudGeometry(unittest.TestCase):
    """Test geometric correctness of point cloud generation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if rclpy.ok():
            rclpy.shutdown()

    def test_pinhole_projection(self):
        """Test pinhole camera projection math."""
        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()

        # Set known camera intrinsics
        camera_info = CameraInfo()
        camera_info.k = [
            100.0,
            0.0,
            50.0,  # fx, 0, cx
            0.0,
            100.0,
            50.0,  # 0, fy, cy
            0.0,
            0.0,
            1.0,
        ]
        node.camera_info_callback(camera_info)

        # Create simple depth map with known depth
        depth = np.ones((100, 100), dtype=np.float32) * 2.0  # 2 meters
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        # Generate point cloud
        points = node.generate_pointcloud(depth, rgb)

        # Check z-coordinates (should all be ~2.0)
        z_values = points[:, 2]
        self.assertTrue(np.allclose(z_values, 2.0, atol=0.01))

        # Check center pixel projects to (0, 0, 2)
        # Pixel (50, 50) with depth 2.0 should project to origin
        center_points = points[
            np.isclose(points[:, 0], 0, atol=0.1) & np.isclose(points[:, 1], 0, atol=0.1)
        ]
        self.assertGreater(len(center_points), 0)

        # Clean up
        node.destroy_node()


class TestPerformance(unittest.TestCase):
    """Performance benchmarking tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if rclpy.ok():
            rclpy.shutdown()

    def test_generation_performance(self):
        """Benchmark point cloud generation speed."""
        import time

        from perception_nodes.pointcloud_generator import PointCloudGenerator

        node = PointCloudGenerator()

        # Set camera intrinsics
        camera_info = CameraInfo()
        camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        node.camera_info_callback(camera_info)

        # Create test data
        depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Benchmark generation
        num_iterations = 50
        start = time.time()

        for _ in range(num_iterations):
            _ = node.generate_pointcloud(depth, rgb)

        elapsed = time.time() - start
        avg_time = (elapsed / num_iterations) * 1000  # ms

        # Point cloud generation should be reasonably fast (<100ms)
        # Target is 10 Hz, so <100ms is acceptable
        self.assertLess(avg_time, 100.0, f"Point cloud generation too slow: {avg_time:.2f}ms")

        # Clean up
        node.destroy_node()


if __name__ == "__main__":
    unittest.main()
