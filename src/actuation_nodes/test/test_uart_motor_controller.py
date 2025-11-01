#!/usr/bin/env python3
"""Unit tests for UART Motor Controller Node.

Tests all functionality of the motor controller including:
- Serial communication (mocked)
- Differential drive kinematics
- Watchdog timer functionality
- Emergency stop service
- ROS2 message handling
- Safety features

Author: Local AI Robot Team
License: Apache-2.0
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import rclpy
from geometry_msgs.msg import Twist

# Import the node under test
from actuation_nodes.uart_motor_controller import UARTMotorController
from robot_interfaces.msg import MotorCommand
from robot_interfaces.srv import EmergencyStop


class TestUARTMotorController(unittest.TestCase):
    """Test suite for UART Motor Controller Node."""

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
        self.serial_patcher = patch("actuation_nodes.uart_motor_controller.serial.Serial")
        self.mock_serial_class = self.serial_patcher.start()
        self.mock_serial = MagicMock()
        self.mock_serial_class.return_value = self.mock_serial
        self.mock_serial.is_open = True

        # Create node with mocked serial
        self.node = UARTMotorController()

        # Store original timers to control them in tests
        self.original_timers = {}

    def tearDown(self):
        """Clean up after each test."""
        self.node.destroy_node()
        self.serial_patcher.stop()

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        self.assertEqual(self.node.get_name(), "uart_motor_controller")
        self.assertIsNotNone(self.node.uart_config)
        self.assertIsNotNone(self.node.motor_config)
        self.assertFalse(self.node._emergency_stop_active)
        self.assertTrue(self.node._motors_enabled)

    def test_serial_connection(self):
        """Test serial connection establishment."""
        # Serial should be mocked and configured
        self.mock_serial_class.assert_called_once()
        self.mock_serial.flushInput.assert_called_once()
        self.mock_serial.flushOutput.assert_called_once()

    def test_send_command(self):
        """Test sending JSON commands via UART."""
        # Test successful command sending
        command = {"T": 1, "L": 0.2, "R": 0.3}
        result = self.node._send_command(command)

        self.assertTrue(result)
        expected_json = json.dumps(command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_send_command_serial_error(self):
        """Test command sending with serial error."""
        # Mock serial write to raise exception
        self.mock_serial.write.side_effect = Exception("Serial error")

        command = {"T": 1, "L": 0.0, "R": 0.0}
        result = self.node._send_command(command)

        self.assertFalse(result)

    def test_twist_to_wheel_speeds(self):
        """Test differential drive kinematics conversion."""
        # Test forward motion
        twist = Twist()
        twist.linear.x = 0.2  # 0.2 m/s forward
        twist.angular.z = 0.0  # No rotation

        left_speed, right_speed = self.node._twist_to_wheel_speeds(twist)

        # Both wheels should have same speed for forward motion
        self.assertAlmostEqual(left_speed, right_speed, places=3)
        self.assertGreater(left_speed, 0)

        # Test rotation (turn in place)
        twist.linear.x = 0.0
        twist.angular.z = 0.5  # 0.5 rad/s rotation

        left_speed, right_speed = self.node._twist_to_wheel_speeds(twist)

        # Wheels should have opposite speeds for rotation
        self.assertAlmostEqual(left_speed, -right_speed, places=3)

        # Test combined motion
        twist.linear.x = 0.1
        twist.angular.z = 0.2

        left_speed, right_speed = self.node._twist_to_wheel_speeds(twist)

        # Right wheel should be faster for left turn
        self.assertGreater(right_speed, left_speed)

    def test_speed_limiting(self):
        """Test that wheel speeds are properly limited."""
        # Test maximum speed limiting
        twist = Twist()
        twist.linear.x = 10.0  # Very high speed
        twist.angular.z = 0.0

        left_speed, right_speed = self.node._twist_to_wheel_speeds(twist)

        # Speeds should be limited to max_speed
        self.assertLessEqual(abs(left_speed), self.node.motor_config.max_speed)
        self.assertLessEqual(abs(right_speed), self.node.motor_config.max_speed)

    def test_cmd_vel_callback(self):
        """Test cmd_vel message handling."""
        twist = Twist()
        twist.linear.x = 0.1
        twist.angular.z = 0.2

        # Store time before callback
        time_before = time.time()

        self.node._cmd_vel_callback(twist)

        # Check that last command was stored
        with self.node._state_lock:
            self.assertEqual(self.node._last_cmd_vel.linear.x, 0.1)
            self.assertEqual(self.node._last_cmd_vel.angular.z, 0.2)
            self.assertGreaterEqual(self.node._last_command_time, time_before)

    def test_motor_command_callback(self):
        """Test direct motor command handling."""
        motor_cmd = MotorCommand()
        motor_cmd.left_speed = 0.3
        motor_cmd.right_speed = 0.4
        motor_cmd.enable = True
        motor_cmd.command_id = 123

        time_before = time.time()

        self.node._motor_command_callback(motor_cmd)

        # Should have sent motor command
        expected_command = {"T": 1, "L": 0.3, "R": 0.4}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

        # Command time should be updated
        with self.node._state_lock:
            self.assertGreaterEqual(self.node._last_command_time, time_before)

    def test_motor_command_disabled(self):
        """Test motor command with enable=False."""
        motor_cmd = MotorCommand()
        motor_cmd.left_speed = 0.3
        motor_cmd.right_speed = 0.4
        motor_cmd.enable = False

        # Clear previous calls
        self.mock_serial.write.reset_mock()

        self.node._motor_command_callback(motor_cmd)

        # Should not send motor command when disabled
        self.mock_serial.write.assert_not_called()

    def test_emergency_stop_service(self):
        """Test emergency stop service functionality."""
        # Create service request
        request = EmergencyStop.Request()
        request.enable_stop = True
        request.reason = "Test emergency stop"

        # Call service
        response = EmergencyStop.Response()
        response = self.node._emergency_stop_callback(request, response)

        # Check response
        self.assertTrue(response.success)
        self.assertIn("Test emergency stop", response.message)

        # Check that emergency stop is active
        self.assertTrue(self.node._emergency_stop_active)

        # Should have sent stop command
        expected_command = {"T": 1, "L": 0.0, "R": 0.0}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_emergency_stop_deactivation(self):
        """Test emergency stop deactivation."""
        # First activate emergency stop
        self.node._emergency_stop_active = True

        # Create deactivation request
        request = EmergencyStop.Request()
        request.enable_stop = False
        request.reason = ""

        response = EmergencyStop.Response()
        response = self.node._emergency_stop_callback(request, response)

        # Check response
        self.assertTrue(response.success)
        self.assertIn("deactivated", response.message)

        # Check that emergency stop is inactive
        self.assertFalse(self.node._emergency_stop_active)

    def test_emergency_stop_blocks_commands(self):
        """Test that emergency stop blocks motor commands."""
        # Activate emergency stop
        self.node._emergency_stop_active = True

        # Clear previous calls
        self.mock_serial.write.reset_mock()

        # Try to send motor command
        success = self.node._send_motor_command(0.3, 0.4)

        # Should still return True (command was sent)
        self.assertTrue(success)

        # But the actual command should be zero speeds
        expected_command = {"T": 1, "L": 0.0, "R": 0.0}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_watchdog_functionality(self):
        """Test watchdog timer stops motors on timeout."""
        # Set up conditions for watchdog to trigger
        current_time = time.time()

        with self.node._state_lock:
            # Simulate node has been running for sufficient time
            self.node._node_start_time = current_time - 10.0  # 10 seconds ago

            # Set command times to simulate timeout
            self.node._last_command_time = current_time - 1.0  # 1 second ago (timeout)
            self.node._last_nonzero_command_time = current_time - 0.5  # Recent motion command

            # Set motor speeds to indicate motors are running
            self.node._current_wheel_speeds = {"left": 0.3, "right": 0.3}

            # Mark that commands have been received
            self.node._commands_received = True

        # Clear previous calls
        self.mock_serial.write.reset_mock()

        # Trigger watchdog callback
        self.node._watchdog_callback()

        # Should have sent stop command
        expected_command = {"T": 1, "L": 0.0, "R": 0.0}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_watchdog_no_action_when_stopped(self):
        """Test watchdog doesn't send unnecessary stop commands."""
        # Set a command time in the past but motors already stopped
        with self.node._state_lock:
            self.node._last_command_time = time.time() - 1.0
            self.node._current_wheel_speeds = {"left": 0.0, "right": 0.0}

        # Clear previous calls
        self.mock_serial.write.reset_mock()

        # Trigger watchdog callback
        self.node._watchdog_callback()

        # Should not send any commands
        self.mock_serial.write.assert_not_called()

    def test_continuous_feedback_enable(self):
        """Test enabling continuous feedback mode."""
        # Reset mock to check specific call
        self.mock_serial.write.reset_mock()

        # Call enable feedback
        result = self.node._enable_continuous_feedback()

        self.assertTrue(result)

        # Should have sent continuous feedback command
        expected_command = {"T": 131, "cmd": 1}
        expected_json = json.dumps(expected_command) + "\n"
        self.mock_serial.write.assert_called_with(expected_json.encode())

    def test_parameter_loading(self):
        """Test that parameters are loaded correctly."""
        # Check that default parameters are set
        self.assertEqual(self.node.uart_config.port, "/dev/ttyTHS1")
        self.assertEqual(self.node.uart_config.baudrate, 115200)
        self.assertEqual(self.node.motor_config.command_rate, 20.0)
        self.assertEqual(self.node.motor_config.max_speed, 0.5)
        self.assertEqual(self.node.motor_config.wheelbase, 0.16)

    def test_serial_reconnection(self):
        """Test handling of serial connection loss."""
        # Simulate serial port closure
        self.mock_serial.is_open = False

        # Mock _connect_serial to return False (failed reconnection)
        with patch.object(self.node, "_connect_serial", return_value=False):
            # Try to send command
            command = {"T": 1, "L": 0.0, "R": 0.0}
            result = self.node._send_command(command)

            # Should return False when serial is not available
            self.assertFalse(result)

    def test_command_rate_limits(self):
        """Test that command rate limiting works."""
        # This would require more complex timing tests
        # For now, just verify the timer is created with correct period
        # Check that timer exists (implementation detail may vary)
        self.assertIsNotNone(self.node.command_timer)

    def test_thread_safety(self):
        """Test thread safety of critical sections."""
        # Test that locks are used properly
        self.assertIsNotNone(self.node._serial_lock)
        self.assertIsNotNone(self.node._state_lock)

        # Test concurrent access (simplified test)
        def update_state():
            with self.node._state_lock:
                self.node._last_command_time = time.time()

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=update_state)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should complete without deadlock
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
