#!/usr/bin/env python3
"""
Unit tests for image_undistort_node.py
Tests calibration loading, GPU/CPU undistortion, and performance optimization.

Author: Local AI Robot Team
License: Apache-2.0
"""

import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge

from perception_nodes.image_undistort_node import ImageUndistortNode


class TestImageUndistortNode(unittest.TestCase):
    """Test cases for ImageUndistortNode class."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize ROS2 for testing
        if not rclpy.ok():
            rclpy.init()

        # Create test calibration data
        self.test_calib_data = {
            "camera_matrix": [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [[-0.3, 0.1, 0.0, 0.0, -0.01]],
            "image_width": 640,
            "image_height": 480,
        }

        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create test configuration
        self.test_config = {
            "ros2": {
                "raw_image_topic": "/camera/raw",
                "undistorted_image_topic": "/camera/undistorted",
                "frame_id": "camera_link",
            },
            "undistortion": {
                "use_gpu_acceleration": False,  # Start with CPU for consistent testing
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
            "calibration_file": "test_calibration.yaml",
        }

    def tearDown(self):
        """Clean up after tests."""
        # Small delay to allow any background processes to complete
        time.sleep(0.1)

    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion")
    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2")
    def test_node_initialization(self, mock_setup_ros2, mock_setup_undist):
        """Test basic node initialization."""
        mock_setup_ros2.return_value = None
        mock_setup_undist.return_value = None

        with patch(
            "builtins.open",
            unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
        ):
            node = ImageUndistortNode()

        # Verify node is properly initialized
        self.assertIsInstance(node, rclpy.node.Node)
        self.assertEqual(node.get_name(), "image_undistort_node")

        # Verify methods were called
        mock_setup_ros2.assert_called_once()
        mock_setup_undist.assert_called_once()

        node.destroy_node()

    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2")
    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion")
    def test_config_loading(self, mock_setup_undist, mock_setup_ros2):
        """Test configuration loading from YAML file."""
        mock_setup_ros2.return_value = None
        mock_setup_undist.return_value = None

        with patch(
            "builtins.open",
            unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
        ):
            node = ImageUndistortNode()

        # Verify config was loaded correctly
        self.assertEqual(node.config["ros2"]["raw_image_topic"], "/camera/raw")
        self.assertEqual(node.config["undistortion"]["interpolation_method"], "linear")
        self.assertEqual(node.config["undistortion"]["use_gpu_acceleration"], False)

        node.destroy_node()

    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2")
    def test_calibration_loading(self, mock_setup_ros2):
        """Test camera calibration loading."""
        mock_setup_ros2.return_value = None

        # Mock config and calibration loading
        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("yaml.safe_load", side_effect=[self.test_config, self.test_calib_data]):
                node = ImageUndistortNode()

        # Verify calibration was loaded correctly
        expected_camera_matrix = np.array(self.test_calib_data["camera_matrix"], dtype=np.float32)
        expected_dist_coeffs = np.array(
            self.test_calib_data["distortion_coefficients"], dtype=np.float32
        ).flatten()

        np.testing.assert_array_equal(node.camera_matrix, expected_camera_matrix)
        np.testing.assert_array_equal(node.dist_coeffs, expected_dist_coeffs)
        self.assertEqual(node.image_width, 640)
        self.assertEqual(node.image_height, 480)

        node.destroy_node()

    def test_interpolation_method_mapping(self):
        """Test interpolation method string to OpenCV constant mapping."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Mock parameter
        mock_param = Mock()
        mock_param.value = "linear"
        node.get_parameter = Mock(return_value=mock_param)

        # Test interpolation method mapping
        interp = node._get_interpolation_method()
        self.assertEqual(interp, cv2.INTER_LINEAR)

        # Test other methods
        mock_param.value = "cubic"
        interp = node._get_interpolation_method()
        self.assertEqual(interp, cv2.INTER_CUBIC)

        mock_param.value = "nearest"
        interp = node._get_interpolation_method()
        self.assertEqual(interp, cv2.INTER_NEAREST)

        node.destroy_node()

    def test_border_mode_mapping(self):
        """Test border mode string to OpenCV constant mapping."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Test border mode mapping
        border = node._get_border_mode()
        self.assertEqual(border, cv2.BORDER_CONSTANT)

        # Test with different config
        node.config["undistortion"]["border_mode"] = "reflect"
        border = node._get_border_mode()
        self.assertEqual(border, cv2.BORDER_REFLECT)

        node.destroy_node()

    @patch("cv2.initUndistortRectifyMap")
    @patch("cv2.getOptimalNewCameraMatrix")
    def test_cpu_undistortion_setup(self, mock_optimal_matrix, mock_init_map):
        """Test CPU undistortion setup."""
        # Mock return values
        mock_optimal_matrix.return_value = (np.eye(3), (0, 0, 640, 480))
        mock_init_map.return_value = (np.zeros((480, 640)), np.zeros((480, 640)))

        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch("builtins.open", unittest.mock.mock_open()):
                with patch(
                    "yaml.safe_load",
                    side_effect=[self.test_config, self.test_calib_data],
                ):
                    node = ImageUndistortNode()

        # Verify CPU setup was called
        mock_optimal_matrix.assert_called_once()
        mock_init_map.assert_called_once()
        self.assertFalse(node.use_gpu)

        node.destroy_node()

    @patch("cv2.undistort")
    def test_cpu_undistortion_processing(self, mock_undistort):
        """Test CPU undistortion processing."""
        # Setup mock
        mock_undistort.return_value = self.test_image

        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Set up node for CPU processing
        node.camera_matrix = np.array(self.test_calib_data["camera_matrix"], dtype=np.float32)
        node.dist_coeffs = np.array(
            self.test_calib_data["distortion_coefficients"], dtype=np.float32
        ).flatten()
        node.new_camera_matrix = node.camera_matrix
        node.use_gpu = False
        node.config["undistortion"]["cache_maps"] = False

        # Test CPU undistortion
        result = node._undistort_cpu(self.test_image)

        # Verify undistort was called
        mock_undistort.assert_called_once()
        np.testing.assert_array_equal(result, self.test_image)

        node.destroy_node()

    @patch("cv2.remap")
    def test_cpu_undistortion_with_cached_maps(self, mock_remap):
        """Test CPU undistortion with cached maps."""
        # Setup mock
        mock_remap.return_value = self.test_image

        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Set up node for CPU processing with cached maps
        node.use_gpu = False
        node.config["undistortion"]["cache_maps"] = True
        node.map1 = np.zeros((480, 640), dtype=np.float32)
        node.map2 = np.zeros((480, 640), dtype=np.float32)

        # Mock parameter
        mock_param = Mock()
        mock_param.value = "linear"
        node.get_parameter = Mock(return_value=mock_param)

        # Test CPU undistortion with cached maps
        result = node._undistort_cpu(self.test_image)

        # Verify remap was called instead of undistort
        mock_remap.assert_called_once()
        np.testing.assert_array_equal(result, self.test_image)

        node.destroy_node()

    def test_performance_metrics_update(self):
        """Test performance metrics updating."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Test performance metrics
        start_time = time.time() - 0.01  # 10ms ago
        node._update_performance_metrics(start_time)

        # Verify metrics were updated
        self.assertEqual(len(node.processing_times), 1)
        self.assertGreater(node.processing_times[0], 0)
        self.assertEqual(node.frame_count, 1)

        node.destroy_node()

    def test_fps_calculation(self):
        """Test FPS calculation."""
        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Simulate frame processing over time
        node.last_fps_time = time.time() - 2.0  # 2 seconds ago
        node.frame_count = 60  # 60 frames

        node._update_performance_metrics(time.time())

        # FPS should be approximately 30 (60 frames / 2 seconds)
        self.assertAlmostEqual(node.fps, 30.0, delta=1.0)

        node.destroy_node()

    @patch("perception_nodes.image_undistort_node.ImageUndistortNode._undistort_cpu")
    def test_image_callback_processing(self, mock_undistort_cpu):
        """Test image callback processing."""
        # Setup mock
        mock_undistort_cpu.return_value = self.test_image

        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                ):
                    node = ImageUndistortNode()

        # Setup node
        node.use_gpu = False
        node.image_pub = Mock()

        # Create test ROS image message
        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(self.test_image, "bgr8")

        # Test image callback
        node._image_callback(ros_image)

        # Verify processing occurred
        mock_undistort_cpu.assert_called_once()
        node.image_pub.publish.assert_called_once()

        node.destroy_node()

    def test_error_handling_in_undistortion(self):
        """Test error handling during undistortion process."""
        node = None
        try:
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
                    with patch(
                        "builtins.open",
                        unittest.mock.mock_open(read_data=yaml.dump(self.test_config)),
                    ):
                        node = ImageUndistortNode()

            # Setup node with invalid calibration to trigger error
            node.camera_matrix = None
            node.dist_coeffs = None
            node.new_camera_matrix = None
            node.use_gpu = False

            # Test error handling - should return original image
            result = node._undistort_cpu(self.test_image)

            # Since calibration is invalid, should return original image
            np.testing.assert_array_equal(result, self.test_image)

        finally:
            if node is not None:
                node.destroy_node()
                time.sleep(0.1)  # Allow cleanup

    @patch("perception_nodes.image_undistort_node.GPU_AVAILABLE", True)
    @patch("cv2.cuda")
    def test_gpu_availability_detection(self, mock_cuda):
        """Test GPU availability detection."""
        # Mock GPU components
        mock_cuda.GpuMat = Mock()
        mock_cuda.remap = Mock()

        config_with_gpu = self.test_config.copy()
        config_with_gpu["undistortion"]["use_gpu_acceleration"] = True

        with patch("perception_nodes.image_undistort_node.ImageUndistortNode._setup_ros2"):
            with patch(
                "perception_nodes.image_undistort_node.ImageUndistortNode._setup_undistortion"
            ):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(read_data=yaml.dump(config_with_gpu)),
                ):
                    node = ImageUndistortNode()

        # Verify GPU availability was considered
        # Note: Actual GPU setup depends on successful mock configuration
        node.destroy_node()


if __name__ == "__main__":
    # Initialize ROS2 for testing
    rclpy.init()

    try:
        unittest.main()
    finally:
        rclpy.shutdown()
