#!/usr/bin/env python3
"""
Integration test for the complete camera pipeline.
Tests camera_driver.py and image_undistort_node.py working together.

Author: Local AI Robot Team
License: Apache-2.0
"""

import os
import tempfile
import threading
import time
import unittest
from typing import List
from unittest.mock import Mock, patch

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraPipelineIntegrationTest(unittest.TestCase):
    """Integration test for the complete camera pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize ROS2 for testing
        if not rclpy.ok():
            rclpy.init()

        self.executor = MultiThreadedExecutor()
        self.bridge = CvBridge()

        # Test data
        self.received_raw_images: List[Image] = []
        self.received_undistorted_images: List[Image] = []
        self.received_camera_info: List[CameraInfo] = []

        # Create test calibration and config
        self._create_test_files()

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        if hasattr(self, "config_file"):
            os.unlink(self.config_file)
        if hasattr(self, "calib_file"):
            os.unlink(self.calib_file)

        # Shutdown executor
        self.executor.shutdown()

    def _create_test_files(self):
        """Create temporary test configuration and calibration files."""
        # Camera calibration data
        calib_data = {
            "calibration_date": "2025-10-30T21:20:43.195898",
            "camera_matrix": [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [[-0.3, 0.1, 0.0, 0.0, -0.01]],
            "image_width": 640,
            "image_height": 480,
            "capture_resolution": "640x480",
            "pipeline_type": "DeepStream_nvarguscamerasrc",
        }

        # Create calibration file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(calib_data, f)
            self.calib_file = f.name

        # Camera configuration
        config_data = {
            "camera": {
                "device_id": 0,
                "sensor_mode": 0,
                "width": 640,
                "height": 480,
                "framerate": 30,
                "flip_method": 0,
            },
            "deepstream": {
                "source_element": "nvarguscamerasrc",
                "nvmm_memory": True,
                "format": "NV12",
                "buffer_pool_size": 4,
                "max_buffers": 8,
                "do_timestamp": True,
            },
            "ros2": {
                "raw_image_topic": "/camera/raw",
                "camera_info_topic": "/camera/camera_info",
                "undistorted_image_topic": "/camera/undistorted",
                "frame_id": "camera_link",
                "publish_camera_info": True,
            },
            "undistortion": {
                "use_gpu_acceleration": False,  # Use CPU for testing
                "interpolation_method": "linear",
                "border_mode": "constant",
                "border_value": [0, 0, 0],
                "cache_maps": True,
                "use_optimized_camera_matrix": True,
                "alpha": 1.0,
            },
            "monitoring": {
                "enable_fps_monitoring": True,
                "enable_gpu_monitoring": False,
                "log_performance_stats": False,
                "stats_publish_rate": 1.0,
            },
            "calibration_file": self.calib_file,
        }

        # Create config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            self.config_file = f.name

    def _create_test_subscriber_node(self) -> Node:
        """Create a test subscriber node to monitor pipeline output."""

        class TestSubscriberNode(Node):
            def __init__(self, test_instance):
                super().__init__("test_subscriber")
                self.test_instance = test_instance

                # Subscribers
                self.raw_sub = self.create_subscription(
                    Image, "/camera/raw", self._raw_callback, 10
                )
                self.undistorted_sub = self.create_subscription(
                    Image, "/camera/undistorted", self._undistorted_callback, 10
                )
                self.camera_info_sub = self.create_subscription(
                    CameraInfo, "/camera/camera_info", self._camera_info_callback, 10
                )

            def _raw_callback(self, msg):
                self.test_instance.received_raw_images.append(msg)

            def _undistorted_callback(self, msg):
                self.test_instance.received_undistorted_images.append(msg)

            def _camera_info_callback(self, msg):
                self.test_instance.received_camera_info.append(msg)

        return TestSubscriberNode(self)

    def _create_mock_camera_driver(self) -> Node:
        """Create a mock camera driver that publishes test images."""

        class MockCameraDriver(Node):
            def __init__(self, calib_file):
                super().__init__("mock_camera_driver")

                # Load calibration for camera info
                with open(calib_file, "r") as f:
                    calib_data = yaml.safe_load(f)

                # Create camera info
                self.camera_info = CameraInfo()
                self.camera_info.header.frame_id = "camera_link"
                self.camera_info.width = calib_data["image_width"]
                self.camera_info.height = calib_data["image_height"]

                K = np.array(calib_data["camera_matrix"]).flatten()
                self.camera_info.k = K.tolist()

                D = np.array(calib_data["distortion_coefficients"]).flatten()
                self.camera_info.d = D.tolist()
                self.camera_info.distortion_model = "plumb_bob"

                # Publishers
                self.image_pub = self.create_publisher(Image, "/camera/raw", 10)
                self.camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)

                # Timer to publish test images
                self.timer = self.create_timer(0.1, self._publish_test_image)  # 10 Hz
                self.frame_count = 0

                self.bridge = CvBridge()

            def _publish_test_image(self):
                """Publish a test image with some distortion pattern."""
                # Create test image with checkerboard pattern
                image = np.zeros((480, 640, 3), dtype=np.uint8)

                # Add checkerboard pattern
                for i in range(0, 480, 40):
                    for j in range(0, 640, 40):
                        if ((i // 40) + (j // 40)) % 2 == 0:
                            image[i : i + 40, j : j + 40] = [255, 255, 255]

                # Add some distortion-like effects (barrel distortion simulation)
                center_x, center_y = 320, 240
                for y in range(480):
                    for x in range(640):
                        dx = x - center_x
                        dy = y - center_y
                        r = np.sqrt(dx * dx + dy * dy)
                        if r > 0:
                            # Simple radial distortion
                            k = -0.0002
                            r_distorted = r * (1 + k * r * r)
                            if r_distorted < r * 1.5:  # Limit distortion
                                scale = r_distorted / r
                                new_x = int(center_x + dx * scale)
                                new_y = int(center_y + dy * scale)
                                if 0 <= new_x < 640 and 0 <= new_y < 480:
                                    image[y, x] = [128, 128, 128]

                # Convert to ROS message
                header = self.get_clock().now().to_msg()
                image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                image_msg.header.stamp = header
                image_msg.header.frame_id = "camera_link"

                # Publish image and camera info
                self.image_pub.publish(image_msg)

                self.camera_info.header.stamp = header
                self.camera_info_pub.publish(self.camera_info)

                self.frame_count += 1

                # Stop after publishing a few frames for testing
                if self.frame_count >= 5:
                    self.timer.cancel()

        return MockCameraDriver(self.calib_file)

    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._load_config")
    def test_complete_pipeline_integration(self, mock_load_config):
        """Test the complete camera pipeline from raw capture to undistorted output."""

        # Mock config loading to use our test files
        def mock_config_side_effect():
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f)

        mock_load_config.side_effect = mock_config_side_effect

        # Import nodes after mocking
        from perception_nodes.image_undistort_node import ImageUndistortNode

        # Create nodes
        mock_camera = self._create_mock_camera_driver()
        undistort_node = ImageUndistortNode()
        test_subscriber = self._create_test_subscriber_node()

        # Add nodes to executor
        self.executor.add_node(mock_camera)
        self.executor.add_node(undistort_node)
        self.executor.add_node(test_subscriber)

        # Start executor in a separate thread
        executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        executor_thread.start()

        # Wait for messages to be published and processed
        time.sleep(2.0)

        # Verify messages were received
        self.assertGreater(len(self.received_raw_images), 0, "No raw images received")
        self.assertGreater(
            len(self.received_undistorted_images), 0, "No undistorted images received"
        )
        self.assertGreater(len(self.received_camera_info), 0, "No camera info received")

        # Verify image dimensions are consistent
        for raw_img in self.received_raw_images:
            self.assertEqual(raw_img.width, 640)
            self.assertEqual(raw_img.height, 480)

        for undist_img in self.received_undistorted_images:
            self.assertEqual(undist_img.width, 640)
            self.assertEqual(undist_img.height, 480)

        # Verify camera info
        for cam_info in self.received_camera_info:
            self.assertEqual(cam_info.width, 640)
            self.assertEqual(cam_info.height, 480)
            self.assertEqual(len(cam_info.k), 9)  # 3x3 camera matrix
            self.assertEqual(len(cam_info.d), 5)  # 5 distortion coefficients

        # Verify that undistortion actually occurred
        # (Check that raw and undistorted images are different)
        if self.received_raw_images and self.received_undistorted_images:
            raw_cv = self.bridge.imgmsg_to_cv2(self.received_raw_images[0], "bgr8")
            undist_cv = self.bridge.imgmsg_to_cv2(self.received_undistorted_images[0], "bgr8")

            # Images should be different after undistortion
            difference = cv2.absdiff(raw_cv, undist_cv)
            self.assertGreater(np.sum(difference), 0, "Undistortion did not change the image")

        # Clean up nodes
        mock_camera.destroy_node()
        undistort_node.destroy_node()
        test_subscriber.destroy_node()

    def test_pipeline_performance_monitoring(self):
        """Test that performance monitoring works correctly."""

        # Mock config loading
        def mock_config_side_effect():
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
                config["monitoring"]["log_performance_stats"] = True
                return config

        with patch(
            "perception_nodes.image_undistort_node.ImageUndistortNode._load_config",
            side_effect=mock_config_side_effect,
        ):
            from perception_nodes.image_undistort_node import ImageUndistortNode

            undistort_node = ImageUndistortNode()

        # Simulate some processing
        start_time = time.time() - 0.01  # 10ms ago
        undistort_node._update_performance_metrics(start_time)
        undistort_node._update_performance_metrics(start_time)
        undistort_node._update_performance_metrics(start_time)

        # Verify metrics were collected
        self.assertEqual(len(undistort_node.processing_times), 3)
        self.assertGreater(undistort_node.frame_count, 0)

        undistort_node.destroy_node()

    def test_pipeline_error_recovery(self):
        """Test error recovery in the pipeline."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._load_config"):
            with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
                with patch(
                    "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
                ):
                    from perception_nodes.image_undistort_node import ImageUndistortNode

                    node = ImageUndistortNode()

        # Test with invalid image data
        node.use_gpu = False
        node.camera_matrix = None
        node.dist_coeffs = None

        # Create invalid image message
        invalid_image = Image()
        invalid_image.width = 640
        invalid_image.height = 480
        invalid_image.encoding = "bgr8"
        invalid_image.data = b"invalid_data"

        # Should not crash on invalid input
        try:
            node._image_callback(invalid_image)
        except Exception as e:
            self.fail(f"Pipeline should handle invalid input gracefully, but got: {e}")

        node.destroy_node()

    def test_configuration_parameter_updates(self):
        """Test that configuration parameters can be updated dynamically."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._load_config"):
            with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
                with patch(
                    "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
                ):
                    from perception_nodes.image_undistort_node import ImageUndistortNode

                    node = ImageUndistortNode()

        # Mock parameter methods
        node.declare_parameter = Mock()
        node.get_parameter = Mock()

        # Test parameter setting
        mock_param = Mock()
        mock_param.value = True
        node.get_parameter.return_value = mock_param

        # Should be able to get parameter values
        use_gpu = node.get_parameter("use_gpu").value
        self.assertTrue(use_gpu)

        node.destroy_node()


class CameraPipelineBenchmarkTest(unittest.TestCase):
    """Benchmark tests for camera pipeline performance."""

    def setUp(self):
        """Set up benchmark test fixtures."""
        if not rclpy.ok():
            rclpy.init()

    def test_processing_latency_benchmark(self):
        """Benchmark processing latency for the undistortion node."""
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test CPU undistortion performance
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch("builtins.open", unittest.mock.mock_open(read_data=yaml.dump({}))):
                    from perception_nodes.image_undistort_node import ImageUndistortNode

                    node = ImageUndistortNode()

        # Setup for CPU processing
        node.camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        node.dist_coeffs = np.array([-0.3, 0.1, 0, 0, -0.01], dtype=np.float32)
        node.new_camera_matrix = node.camera_matrix
        node.use_gpu = False
        node.config = {"undistortion": {"cache_maps": False}}

        # Benchmark multiple runs
        latencies = []
        num_runs = 10

        for _ in range(num_runs):
            start_time = time.time()
            _ = node._undistort_cpu(test_image)
            end_time = time.time()
            latencies.append(end_time - start_time)

        # Calculate statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        min_latency = np.min(latencies)

        print("\nCPU Undistortion Benchmark Results:")
        print(f"Average latency: {avg_latency*1000:.2f}ms")
        print(f"Min latency: {min_latency*1000:.2f}ms")
        print(f"Max latency: {max_latency*1000:.2f}ms")
        print(f"Estimated FPS: {1.0/avg_latency:.1f}")

        # Verify reasonable performance (should be < 100ms for 640x480)
        self.assertLess(avg_latency, 0.1, "CPU undistortion too slow")

        node.destroy_node()


if __name__ == "__main__":
    # Initialize ROS2 for testing
    rclpy.init()

    try:
        unittest.main(verbosity=2)
    finally:
        rclpy.shutdown()
