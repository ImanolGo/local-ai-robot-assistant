#!/usr/bin/env python3
"""
Unit tests for camera_driver.py
Tests DeepStream pipeline initialization, frame publishing, and performance optimization.

Author: Local AI Robot Team
License: Apache-2.0
"""

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import rclpy
import yaml
from rclpy.node import Node

# Mock GStreamer before importing camera_driver
mock_gst = MagicMock()
mock_gst.init = MagicMock()
mock_gst.parse_launch = MagicMock()
mock_gst.State = MagicMock()
mock_gst.State.PLAYING = 1
mock_gst.State.NULL = 0
mock_gst.StateChangeReturn = MagicMock()
mock_gst.StateChangeReturn.FAILURE = 0
mock_gst.StateChangeReturn.SUCCESS = 1
mock_gst.MessageType = MagicMock()
mock_gst.FlowReturn = MagicMock()
mock_gst.FlowReturn.OK = 0
mock_gst.FlowReturn.ERROR = 1
mock_gst.MapFlags = MagicMock()
mock_gst.MapFlags.READ = 1

# Create a mock bus that returns None after first call to prevent infinite loop
mock_bus = MagicMock()
mock_bus.timed_pop_filtered.return_value = None
mock_gst.Element = MagicMock()
mock_gst.Element.get_bus.return_value = mock_bus

with patch.dict(
    "sys.modules", {"gi.repository.Gst": mock_gst, "gi.repository.GObject": MagicMock()}
):
    from perception_nodes.camera_driver import CameraDriver


class TestCameraDriver(unittest.TestCase):
    """Test cases for CameraDriver class."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize ROS2 for testing
        if not rclpy.ok():
            rclpy.init()

        # Create temporary config file
        self.temp_config = self._create_test_config()
        self.temp_calib = self._create_test_calibration()

    def tearDown(self):
        """Clean up after tests."""
        # Clean up temporary files
        if hasattr(self, "temp_config"):
            os.unlink(self.temp_config)
        if hasattr(self, "temp_calib"):
            os.unlink(self.temp_calib)

        # Ensure ROS2 cleanup
        time.sleep(0.1)  # Small delay to allow cleanup

    def _create_test_config(self) -> str:
        """Create a temporary camera configuration file."""
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
                "frame_id": "camera_link",
                "publish_camera_info": True,
                "qos_profile": {
                    "reliability": "best_effort",
                    "durability": "volatile",
                    "history": "keep_last",
                    "depth": 1,
                },
            },
            "monitoring": {
                "enable_fps_monitoring": True,
                "enable_gpu_monitoring": True,
                "log_performance_stats": True,
                "stats_publish_rate": 1.0,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            return f.name

    def _create_test_calibration(self) -> str:
        """Create a temporary calibration file."""
        calib_data = {
            "calibration_date": "2025-10-30T21:20:43.195898",
            "camera_matrix": [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [[-0.1, 0.05, 0.0, 0.0, -0.01]],
            "image_width": 640,
            "image_height": 480,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(calib_data, f)
            return f.name

    @patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline")
    @patch("perception_nodes.camera_driver.CameraDriver._start_pipeline")
    @patch("perception_nodes.camera_driver.CameraDriver._load_calibration")
    @patch("perception_nodes.camera_driver.CameraDriver._setup_ros2")
    def test_node_initialization(self, mock_setup_ros2, mock_load_calib, mock_start, mock_setup):
        """Test basic node initialization."""
        # Mock methods to prevent actual pipeline creation
        mock_load_calib.return_value = None
        mock_setup.return_value = None
        mock_start.return_value = None
        mock_setup_ros2.return_value = None

        node = None
        test_config = {
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
                "frame_id": "camera_link",
                "publish_camera_info": True,
            },
            "monitoring": {
                "enable_fps_monitoring": True,
                "enable_gpu_monitoring": True,
                "log_performance_stats": True,
                "stats_publish_rate": 1.0,
            },
        }

        try:
            # Patch the config loading and set config manually
            with patch.object(CameraDriver, "_load_config"):
                node = CameraDriver()
                node.config = test_config  # Set the config manually after creation

            # Verify node is properly initialized
            self.assertIsInstance(node, Node)
            self.assertEqual(node.get_name(), "camera_driver")

            # Verify methods were called
            mock_load_calib.assert_called_once()
            mock_setup.assert_called_once()
            mock_start.assert_called_once()
            mock_setup_ros2.assert_called_once()

        finally:
            if node is not None:
                node.destroy_node()
                time.sleep(0.1)  # Allow cleanup

    @patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline")
    @patch("perception_nodes.camera_driver.CameraDriver._start_pipeline")
    def test_config_loading(self, mock_start, mock_setup):
        """Test configuration loading from YAML file."""
        mock_setup.return_value = None
        mock_start.return_value = None

        # Test with custom config file
        test_config = {
            "camera": {"device_id": 1, "width": 1920, "height": 1080},
            "ros2": {"raw_image_topic": "/test/camera"},
            "monitoring": {"enable_fps_monitoring": False},
        }

        with patch("builtins.open", unittest.mock.mock_open(read_data=yaml.dump(test_config))):
            with patch("perception_nodes.camera_driver.CameraDriver._load_calibration"):
                node = CameraDriver()

        # Verify config was loaded correctly
        self.assertEqual(node.config["camera"]["device_id"], 1)
        self.assertEqual(node.config["camera"]["width"], 1920)
        self.assertEqual(node.config["ros2"]["raw_image_topic"], "/test/camera")

        node.destroy_node()

    @patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline")
    @patch("perception_nodes.camera_driver.CameraDriver._start_pipeline")
    def test_calibration_loading(self, mock_start, mock_setup):
        """Test camera calibration loading."""
        mock_setup.return_value = None
        mock_start.return_value = None

        calib_data = {
            "camera_matrix": [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [[-0.1, 0.05, 0.0, 0.0, -0.01]],
            "image_width": 640,
            "image_height": 480,
        }

        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("yaml.safe_load", side_effect=[{}, calib_data]):  # Config, then calibration
                node = CameraDriver()

        # Verify calibration was loaded
        self.assertIsNotNone(node.camera_info)
        self.assertEqual(node.camera_info.width, 640)
        self.assertEqual(node.camera_info.height, 480)
        self.assertEqual(len(node.camera_info.k), 9)  # 3x3 camera matrix flattened
        self.assertEqual(len(node.camera_info.d), 5)  # 5 distortion coefficients

        node.destroy_node()

    def test_default_config(self):
        """Test default configuration fallback."""
        with patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline"):
            with patch("perception_nodes.camera_driver.CameraDriver._start_pipeline"):
                with patch("perception_nodes.camera_driver.CameraDriver._load_calibration"):
                    with patch("builtins.open", side_effect=FileNotFoundError):
                        node = CameraDriver()

        # Verify default config is used
        default_config = node._get_default_config()
        self.assertEqual(node.config["camera"]["device_id"], default_config["camera"]["device_id"])
        self.assertEqual(
            node.config["ros2"]["raw_image_topic"],
            default_config["ros2"]["raw_image_topic"],
        )

        node.destroy_node()

    @patch("perception_nodes.camera_driver.CameraDriver._load_config")
    @patch("perception_nodes.camera_driver.CameraDriver._setup_ros2")
    @patch("perception_nodes.camera_driver.CameraDriver._load_calibration")
    @patch("perception_nodes.camera_driver.CameraDriver._start_pipeline")
    def test_deepstream_pipeline_creation(self, mock_start, mock_calib, mock_ros2, mock_config):
        """Test DeepStream pipeline creation."""
        # Setup mocks
        mock_config.return_value = None
        mock_ros2.return_value = None
        mock_calib.return_value = None
        mock_start.return_value = None

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_gst.parse_launch.return_value = mock_pipeline
        mock_appsink = MagicMock()
        mock_pipeline.get_by_name.return_value = mock_appsink

        node = CameraDriver()
        node.config = node._get_default_config()

        # Add mock parameters
        node.get_parameter = MagicMock()
        node.get_parameter.return_value.value = (
            0  # device_id, width, height, framerate, flip_method
        )

        # Call pipeline setup
        node._setup_deepstream_pipeline()

        # Verify pipeline was created
        mock_gst.parse_launch.assert_called_once()
        mock_pipeline.get_by_name.assert_called_once_with("appsink")
        mock_appsink.connect.assert_called_once()

        node.destroy_node()

    def test_performance_metrics_update(self):
        """Test performance metrics updating."""
        with patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline"):
            with patch("perception_nodes.camera_driver.CameraDriver._start_pipeline"):
                with patch("perception_nodes.camera_driver.CameraDriver._load_calibration"):
                    with patch(
                        "builtins.open",
                        unittest.mock.mock_open(read_data=yaml.dump({})),
                    ):
                        node = CameraDriver()

        # Test performance metrics
        start_time = time.time() - 0.01  # 10ms ago
        node._update_performance_metrics(start_time)

        # Verify metrics were updated
        self.assertEqual(len(node.processing_times), 1)
        self.assertGreater(node.processing_times[0], 0)
        self.assertEqual(node.frame_count, 1)

        node.destroy_node()

    @patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline")
    @patch("perception_nodes.camera_driver.CameraDriver._start_pipeline")
    def test_publishers_created(self, mock_start, mock_setup):
        """Test that ROS2 publishers are created correctly."""
        mock_setup.return_value = None
        mock_start.return_value = None

        with patch("builtins.open", unittest.mock.mock_open(read_data=yaml.dump({}))):
            with patch("perception_nodes.camera_driver.CameraDriver._load_calibration"):
                node = CameraDriver()

        # Verify publishers exist
        self.assertTrue(hasattr(node, "image_pub"))

        # If camera info publishing is enabled, verify camera info publisher
        if node.config["ros2"]["publish_camera_info"]:
            self.assertTrue(hasattr(node, "camera_info_pub"))

        node.destroy_node()

    def test_fps_calculation(self):
        """Test FPS calculation."""
        with patch("perception_nodes.camera_driver.CameraDriver._setup_deepstream_pipeline"):
            with patch("perception_nodes.camera_driver.CameraDriver._start_pipeline"):
                with patch("perception_nodes.camera_driver.CameraDriver._load_calibration"):
                    with patch(
                        "builtins.open",
                        unittest.mock.mock_open(read_data=yaml.dump({})),
                    ):
                        node = CameraDriver()

        # Simulate frame processing over time
        node.last_fps_time = time.time() - 2.0  # 2 seconds ago
        node.frame_count = 60  # 60 frames

        node._update_performance_metrics(time.time())

        # FPS should be approximately 30 (60 frames / 2 seconds)
        self.assertAlmostEqual(node.fps, 30.0, delta=1.0)

        node.destroy_node()


if __name__ == "__main__":
    # Initialize ROS2 for testing
    rclpy.init()

    try:
        unittest.main()
    finally:
        rclpy.shutdown()
