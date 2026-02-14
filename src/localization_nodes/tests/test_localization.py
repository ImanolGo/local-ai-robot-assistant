"""
Tests for localization and SLAM configuration.

Tests verify:
  - EKF configuration validity
  - RTAB-Map config consistency
  - TF frame naming conventions
  - SLAM health monitor node instantiation
  - Config file loading and parameter validation
"""

import os
import unittest

import yaml


class TestLocalizationConfig(unittest.TestCase):
    """Test localization configuration files are valid and consistent."""

    @classmethod
    def setUpClass(cls):
        """Find config files relative to the package."""
        # Search up from the test file to find the config directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.dirname(test_dir)
        cls.config_dir = os.path.join(pkg_dir, "config")

        cls.ekf_config_path = os.path.join(cls.config_dir, "localization_config.yaml")
        cls.rtabmap_config_path = os.path.join(cls.config_dir, "rtabmap_config.yaml")

    def test_ekf_config_exists(self):
        """EKF config file must exist."""
        self.assertTrue(
            os.path.isfile(self.ekf_config_path),
            f"Missing EKF config: {self.ekf_config_path}",
        )

    def test_rtabmap_config_exists(self):
        """RTAB-Map config file must exist."""
        self.assertTrue(
            os.path.isfile(self.rtabmap_config_path),
            f"Missing RTAB-Map config: {self.rtabmap_config_path}",
        )

    def test_ekf_config_parseable(self):
        """EKF config YAML must be parseable."""
        with open(self.ekf_config_path) as f:
            config = yaml.safe_load(f)
        self.assertIsNotNone(config)
        self.assertIn("ekf_filter_node", config)

    def test_rtabmap_config_parseable(self):
        """RTAB-Map config YAML must be parseable."""
        with open(self.rtabmap_config_path) as f:
            config = yaml.safe_load(f)
        self.assertIsNotNone(config)
        self.assertIn("rtabmap", config)

    def test_tf_frame_consistency(self):
        """TF frame names must be consistent across EKF and RTAB-Map configs."""
        with open(self.ekf_config_path) as f:
            ekf = yaml.safe_load(f)
        with open(self.rtabmap_config_path) as f:
            rtab = yaml.safe_load(f)

        ekf_params = ekf["ekf_filter_node"]["ros__parameters"]
        rtab_params = rtab["rtabmap"]["ros__parameters"]

        # Both must agree on frame names
        self.assertEqual(ekf_params["odom_frame"], rtab_params["odom_frame_id"])
        self.assertEqual(ekf_params["base_link_frame"], rtab_params["frame_id"])
        self.assertEqual(ekf_params["map_frame"], rtab_params["map_frame_id"])

    def test_ekf_imu_input_configured(self):
        """EKF must have IMU input configured."""
        with open(self.ekf_config_path) as f:
            config = yaml.safe_load(f)
        params = config["ekf_filter_node"]["ros__parameters"]

        self.assertEqual(params["imu0"], "/imu/data")
        self.assertIn("imu0_config", params)
        self.assertEqual(len(params["imu0_config"]), 15)

    def test_ekf_visual_odom_configured(self):
        """EKF must have visual odometry input configured (not commented out)."""
        with open(self.ekf_config_path) as f:
            config = yaml.safe_load(f)
        params = config["ekf_filter_node"]["ros__parameters"]

        self.assertEqual(params["odom0"], "/rtabmap/odom")
        self.assertIn("odom0_config", params)
        self.assertEqual(len(params["odom0_config"]), 15)

    def test_rtabmap_subscribe_mode(self):
        """RTAB-Map must use direct depth subscription (not rgbd bundle)."""
        with open(self.rtabmap_config_path) as f:
            config = yaml.safe_load(f)
        params = config["rtabmap"]["ros__parameters"]

        self.assertTrue(params["subscribe_depth"])
        self.assertFalse(params["subscribe_rgbd"])

    def test_rtabmap_odom_no_tf_publish(self):
        """Visual odometry must NOT publish TF (EKF handles that)."""
        with open(self.rtabmap_config_path) as f:
            config = yaml.safe_load(f)
        odom_params = config["rgbd_odometry"]["ros__parameters"]

        self.assertFalse(odom_params["publish_tf"])

    def test_rtabmap_memory_limit(self):
        """RTAB-Map working memory must be limited for 8GB Jetson."""
        with open(self.rtabmap_config_path) as f:
            config = yaml.safe_load(f)
        mem_params = config["rtabmap"]["ros__parameters"]["Mem"]

        wm_size = int(mem_params["WorkingMemorySize"])
        self.assertLessEqual(wm_size, 200, "Working memory too large for 8GB Jetson")
        self.assertGreaterEqual(wm_size, 50, "Working memory too small for useful SLAM")


class TestSLAMHealthMonitorNode(unittest.TestCase):
    """Test SLAM health monitor node can be instantiated.

    These tests require rclpy (ROS2) to be available.
    They will be skipped if rclpy is not importable.
    """

    @classmethod
    def setUpClass(cls):
        """Check if rclpy is available."""
        try:
            import rclpy  # noqa: F401

            cls.rclpy_available = True
        except ImportError:
            cls.rclpy_available = False

    def setUp(self):
        if not self.rclpy_available:
            self.skipTest("rclpy not available")
        import rclpy

        rclpy.init()
        self.node = None

    def tearDown(self):
        if not self.rclpy_available:
            return
        import rclpy

        if self.node:
            self.node.destroy_node()
        rclpy.shutdown()

    def test_node_instantiation(self):
        """SLAM health monitor node must instantiate without errors."""
        from localization_nodes.slam_node import SLAMHealthMonitorNode

        self.node = SLAMHealthMonitorNode()
        self.assertEqual(self.node.get_name(), "slam_health_monitor")

    def test_node_has_status_publisher(self):
        """Node must have a /slam/status publisher."""
        from localization_nodes.slam_node import SLAMHealthMonitorNode

        self.node = SLAMHealthMonitorNode()

        topic_names = [t[0] for t in self.node.get_topic_names_and_types()]
        self.assertIn("/slam/status", topic_names)

    def test_node_default_params(self):
        """Node must have reasonable default parameters."""
        from localization_nodes.slam_node import SLAMHealthMonitorNode

        self.node = SLAMHealthMonitorNode()

        self.assertGreater(self.node.check_interval, 0)
        self.assertGreater(self.node.odom_timeout, 0)
        self.assertGreater(self.node.depth_timeout, 0)


if __name__ == "__main__":
    unittest.main()
