#!/usr/bin/env python3
"""Unit tests for UART IMU Node.

Tests all functionality of the IMU node including:
- Serial communication (mocked)
- IMU data parsing and validation
- ROS2 message conversion
- Coordinate frame transformations
- Error handling and recovery

Author: Local AI Robot Team
License: Apache-2.0
"""

import json
import math
import unittest
from unittest.mock import MagicMock, patch

import rclpy
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Imu

# Import the node under test
from localization_nodes.uart_imu_node import UARTIMUNode


class TestUARTIMUNode(unittest.TestCase):
    """Test suite for UART IMU Node."""

    @classmethod
    def setUpClass(cls):
        """Set up ROS2 for all tests."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Cleanup ROS2 after all tests."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures before each test."""
        # Mock serial to avoid hardware dependency
        self.serial_patcher = patch("localization_nodes.uart_imu_node.serial.Serial")
        self.mock_serial_class = self.serial_patcher.start()
        self.mock_serial = MagicMock()
        self.mock_serial_class.return_value = self.mock_serial
        self.mock_serial.is_open = True
        self.mock_serial.in_waiting = 0

        # Create node with mocked serial
        self.node = UARTIMUNode()

    def tearDown(self):
        """Clean up after each test."""
        self.node.destroy_node()
        self.serial_patcher.stop()

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        self.assertEqual(self.node.get_name(), "uart_imu_node")
        self.assertIsNotNone(self.node.uart_config)
        self.assertIsNotNone(self.node.imu_config)
        self.assertEqual(self.node._imu_sequence, 0)

    def test_serial_connection(self):
        """Test serial connection establishment."""
        self.mock_serial_class.assert_called_once()
        self.mock_serial.flushInput.assert_called_once()
        self.mock_serial.flushOutput.assert_called_once()

    def test_send_imu_query(self):
        """Test sending IMU query command."""
        command = {"T": 126}
        result = self.node._send_command(command)

        self.assertTrue(result)
        expected_json = json.dumps(command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_send_command_serial_error(self):
        """Test command sending with serial error."""
        self.mock_serial.write.side_effect = Exception("Serial error")

        command = {"T": 126}
        result = self.node._send_command(command)

        self.assertFalse(result)

    def test_valid_imu_data_parsing(self):
        """Test parsing of valid IMU data."""
        # Sample IMU data from Wave Rover
        sample_data = {
            "roll": 5.2,
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 0.98,
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        # Test validation
        is_valid = self.node._validate_imu_data(sample_data)
        self.assertTrue(is_valid)

    def test_invalid_imu_data_missing_fields(self):
        """Test rejection of IMU data with missing fields."""
        # Missing required fields
        invalid_data = {
            "roll": 5.2,
            "pitch": -2.1,
            # Missing yaw and other fields
        }

        is_valid = self.node._validate_imu_data(invalid_data)
        self.assertFalse(is_valid)

    def test_invalid_imu_data_out_of_range(self):
        """Test rejection of IMU data with out-of-range values."""
        # Angle out of range
        invalid_data = {
            "roll": 500.0,  # Way out of range
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 0.98,
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        is_valid = self.node._validate_imu_data(invalid_data)
        self.assertFalse(is_valid)

    def test_invalid_imu_data_extreme_acceleration(self):
        """Test rejection of IMU data with extreme acceleration."""
        # Extreme acceleration value
        invalid_data = {
            "roll": 5.2,
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 100.0,  # Extreme acceleration
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        is_valid = self.node._validate_imu_data(invalid_data)
        self.assertFalse(is_valid)

    def test_degrees_to_radians_conversion(self):
        """Test angle conversion from degrees to radians."""
        # Test known conversions
        self.assertAlmostEqual(self.node._degrees_to_radians(0), 0, places=6)
        self.assertAlmostEqual(self.node._degrees_to_radians(90), math.pi / 2, places=6)
        self.assertAlmostEqual(self.node._degrees_to_radians(180), math.pi, places=6)
        self.assertAlmostEqual(self.node._degrees_to_radians(-90), -math.pi / 2, places=6)

    def test_euler_to_quaternion_conversion(self):
        """Test Euler angle to quaternion conversion."""
        # Test zero rotation
        q = self.node._euler_to_quaternion(0, 0, 0)
        self.assertAlmostEqual(q.w, 1.0, places=6)
        self.assertAlmostEqual(q.x, 0.0, places=6)
        self.assertAlmostEqual(q.y, 0.0, places=6)
        self.assertAlmostEqual(q.z, 0.0, places=6)

        # Test 90 degree yaw rotation
        q = self.node._euler_to_quaternion(0, 0, 90)
        self.assertAlmostEqual(q.w, math.cos(math.pi / 4), places=6)
        self.assertAlmostEqual(q.z, math.sin(math.pi / 4), places=6)

        # Test quaternion normalization (should have unit length)
        q = self.node._euler_to_quaternion(30, 45, 60)
        magnitude = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        self.assertAlmostEqual(magnitude, 1.0, places=6)

    def test_create_imu_message(self):
        """Test creation of ROS2 IMU message from parsed data."""
        sample_data = {
            "roll": 5.2,
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 0.98,
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        imu_msg = self.node._create_imu_message(sample_data)

        # Check message type
        self.assertIsInstance(imu_msg, Imu)

        # Check header
        self.assertEqual(imu_msg.header.frame_id, "imu_link")
        self.assertIsNotNone(imu_msg.header.stamp)

        # Check orientation quaternion
        self.assertIsInstance(imu_msg.orientation, Quaternion)

        # Check angular velocity (should be converted to rad/s)
        expected_gyro_x = self.node._degrees_to_radians(sample_data["GyroX"])
        self.assertAlmostEqual(imu_msg.angular_velocity.x, expected_gyro_x, places=6)

        # Check linear acceleration (should be converted to m/s²)
        g = 9.80665
        expected_acc_x = sample_data["AccX"] * g
        self.assertAlmostEqual(imu_msg.linear_acceleration.x, expected_acc_x, places=6)

        # Check covariance matrices are properly set
        self.assertEqual(len(imu_msg.orientation_covariance), 9)
        self.assertEqual(len(imu_msg.angular_velocity_covariance), 9)
        self.assertEqual(len(imu_msg.linear_acceleration_covariance), 9)

    def test_read_response_valid_json(self):
        """Test reading valid JSON response from serial."""
        sample_response = {
            "roll": 5.2,
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 0.98,
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        # Mock serial data
        self.mock_serial.in_waiting = 100
        json_str = json.dumps(sample_response) + "\n"
        self.mock_serial.readline.return_value = json_str.encode()

        data = self.node._read_response()

        self.assertIsNotNone(data)
        self.assertEqual(data, sample_response)

    def test_read_response_invalid_json(self):
        """Test handling of invalid JSON response."""
        # Mock invalid JSON
        self.mock_serial.in_waiting = 50
        self.mock_serial.readline.return_value = b"invalid json data\n"

        data = self.node._read_response()

        self.assertIsNone(data)

    def test_read_response_no_data(self):
        """Test handling when no data is available."""
        # Mock no data available
        self.mock_serial.in_waiting = 0

        data = self.node._read_response()

        self.assertIsNone(data)

    def test_read_response_serial_error(self):
        """Test handling of serial read errors."""
        # Mock serial read error
        self.mock_serial.in_waiting = 50
        self.mock_serial.readline.side_effect = Exception("Serial read error")

        data = self.node._read_response()

        self.assertIsNone(data)

    def test_validation_disabled(self):
        """Test behavior when data validation is disabled."""
        # Disable validation
        self.node.imu_config.validate_data = False

        # Even invalid data should pass validation
        invalid_data = {"incomplete": "data"}
        is_valid = self.node._validate_imu_data(invalid_data)

        self.assertTrue(is_valid)

    def test_parameter_loading(self):
        """Test that parameters are loaded correctly."""
        # Check default parameters
        self.assertEqual(self.node.uart_config.port, "/dev/ttyTHS1")
        self.assertEqual(self.node.uart_config.baudrate, 115200)
        self.assertEqual(self.node.imu_config.query_rate, 20.0)
        self.assertEqual(self.node.imu_config.query_command, 126)
        self.assertEqual(self.node.frame_id, "imu_link")

    def test_imu_timer_callback(self):
        """Test periodic IMU query timer."""
        # Clear previous calls
        self.mock_serial.write.reset_mock()

        # Trigger timer callback
        self.node._imu_timer_callback()

        # Should have sent IMU query
        expected_command = {"T": 126}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_process_data_callback_valid_data(self):
        """Test processing of valid IMU data."""
        # Mock valid response
        sample_data = {
            "roll": 5.2,
            "pitch": -2.1,
            "yaw": 45.8,
            "AccX": 0.98,
            "AccY": 0.05,
            "AccZ": 0.12,
            "GyroX": 0.02,
            "GyroY": -0.01,
            "GyroZ": 0.15,
        }

        self.mock_serial.in_waiting = 100
        json_str = json.dumps(sample_data) + "\n"
        self.mock_serial.readline.return_value = json_str.encode()

        # Mock publisher
        mock_publisher = MagicMock()
        self.node.imu_pub = mock_publisher

        # Process data
        self.node._process_data_callback()

        # Should have published IMU message
        mock_publisher.publish.assert_called_once()

        # Check that sequence number increased
        with self.node._data_lock:
            self.assertEqual(self.node._imu_sequence, 1)
            self.assertIsNotNone(self.node._last_imu_data)

    def test_process_data_callback_invalid_data(self):
        """Test processing of invalid IMU data."""
        # Mock invalid response
        invalid_data = {"incomplete": "data"}

        self.mock_serial.in_waiting = 50
        json_str = json.dumps(invalid_data) + "\n"
        self.mock_serial.readline.return_value = json_str.encode()

        # Mock publisher
        mock_publisher = MagicMock()
        self.node.imu_pub = mock_publisher

        # Process data
        self.node._process_data_callback()

        # Should not have published
        mock_publisher.publish.assert_not_called()

        # Sequence should not have changed
        with self.node._data_lock:
            self.assertEqual(self.node._imu_sequence, 0)

    def test_process_data_callback_no_data(self):
        """Test processing when no data is available."""
        # Mock no data
        self.mock_serial.in_waiting = 0

        # Mock publisher
        mock_publisher = MagicMock()
        self.node.imu_pub = mock_publisher

        # Process data
        self.node._process_data_callback()

        # Should not have published
        mock_publisher.publish.assert_not_called()

    def test_serial_connection_failure(self):
        """Test handling of serial connection failure."""
        # Simulate closed serial port
        self.mock_serial.is_open = False

        # Try to send command
        result = self.node._send_command({"T": 126})

        # Should return False
        self.assertFalse(result)

    def test_coordinate_frame_consistency(self):
        """Test that coordinate frames are consistent."""
        # Create IMU message
        sample_data = {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "AccX": 0.0,
            "AccY": 0.0,
            "AccZ": 1.0,  # 1g upward
            "GyroX": 0.0,
            "GyroY": 0.0,
            "GyroZ": 0.0,
        }

        imu_msg = self.node._create_imu_message(sample_data)

        # Check frame ID
        self.assertEqual(imu_msg.header.frame_id, "imu_link")

        # Check that gravity is properly oriented (1g ≈ 9.8 m/s²)
        self.assertAlmostEqual(imu_msg.linear_acceleration.z, 9.80665, places=3)


if __name__ == "__main__":
    unittest.main()
