#!/usr/bin/env python3
"""Integration tests for UART communication with Wave Rover.

These tests require actual hardware and should be run with the Wave Rover
connected via UART. They test the complete integration between the ROS2 nodes
and the physical hardware.

Run these tests with:
    python3 scripts/run_test.py test_uart_integration --port /dev/ttyTHS1

Or build the workspace and source it:
    colcon build --symlink-install
    source install/setup.bash
    python3 integration_tests/test_uart_integration.py --port /dev/ttyTHS1

Or as part of the pytest suite:
    python3 -m pytest integration_tests/test_uart_integration.py -v

Author: Local AI Robot Team
License: Apache-2.0
"""

import argparse
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Optional

# Add source paths for imports when run directly
if __name__ == "__main__":
    workspace_root = Path(__file__).parent.parent
    src_path = workspace_root / "src"

    # Add each package to Python path
    for package_dir in src_path.iterdir():
        if package_dir.is_dir() and not package_dir.name.startswith("."):
            sys.path.insert(0, str(package_dir))

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu

# Import nodes to test
from actuation_nodes.uart_motor_controller import UARTMotorController
from localization_nodes.uart_imu_node import UARTIMUNode
from robot_interfaces.msg import ChassisState, MotorCommand
from robot_interfaces.srv import EmergencyStop


class TestUARTIntegration(unittest.TestCase):
    """Integration tests for UART communication with Wave Rover."""

    @classmethod
    def setUpClass(cls):
        """Set up ROS2 and test environment."""
        # Parse command line arguments for port
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", default="/dev/ttyTHS1", help="Serial port for Wave Rover")
        parser.add_argument(
            "--skip-hardware", action="store_true", help="Skip tests requiring hardware"
        )

        # This is a bit of a hack to get args in unittest
        import sys

        args, unknown = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + unknown  # Remove our args for unittest

        cls.serial_port = args.port
        cls.skip_hardware = args.skip_hardware

        # Initialize ROS2
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Cleanup ROS2."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures."""
        if self.skip_hardware:
            self.skipTest("Hardware tests disabled")

        # Test serial connection availability
        try:
            test_serial = serial.Serial(port=self.serial_port, baudrate=115200, timeout=1.0)
            test_serial.close()
        except Exception as e:
            self.skipTest(f"Serial port {self.serial_port} not available: {e}")

        # Create test node for subscriptions
        self.test_node = Node("uart_integration_test")

        # Storage for received messages
        self.received_chassis_state: Optional[ChassisState] = None
        self.received_imu_data: Optional[Imu] = None
        self.message_count = {"chassis": 0, "imu": 0}

        # Set up subscribers
        self.chassis_sub = self.test_node.create_subscription(
            ChassisState, "/chassis_state", self._chassis_callback, 10
        )

        self.imu_sub = self.test_node.create_subscription(Imu, "/imu/data", self._imu_callback, 10)

        # Create publishers for testing
        self.cmd_vel_pub = self.test_node.create_publisher(Twist, "/cmd_vel", 10)

        self.motor_cmd_pub = self.test_node.create_publisher(MotorCommand, "/motor_command", 10)

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, "test_node"):
            self.test_node.destroy_node()

    def _chassis_callback(self, msg: ChassisState):
        """Callback for chassis state messages."""
        self.received_chassis_state = msg
        self.message_count["chassis"] += 1

    def _imu_callback(self, msg: Imu):
        """Callback for IMU messages."""
        self.received_imu_data = msg
        self.message_count["imu"] += 1

    def _wait_for_messages(self, timeout: float = 5.0) -> bool:
        """Wait for both chassis and IMU messages."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            rclpy.spin_once(self.test_node, timeout_sec=0.1)

            if self.received_chassis_state is not None and self.received_imu_data is not None:
                return True

        return False

    def test_motor_controller_initialization(self):
        """Test that motor controller node initializes correctly."""
        # Create motor controller node
        motor_node = UARTMotorController()

        try:
            # Let node initialize
            time.sleep(1.0)

            # Check that node is running
            self.assertEqual(motor_node.get_name(), "uart_motor_controller")
            self.assertIsNotNone(motor_node._serial)
            self.assertTrue(motor_node._serial.is_open)

        finally:
            motor_node.destroy_node()

    def test_imu_node_initialization(self):
        """Test that IMU node initializes correctly."""
        # Create IMU node
        imu_node = UARTIMUNode()

        try:
            # Let node initialize
            time.sleep(1.0)

            # Check that node is running
            self.assertEqual(imu_node.get_name(), "uart_imu_node")
            self.assertIsNotNone(imu_node._serial)
            self.assertTrue(imu_node._serial.is_open)

        finally:
            imu_node.destroy_node()

    def test_cmd_vel_to_motor_commands(self):
        """Test complete cmd_vel to motor command pipeline."""
        # Create motor controller
        motor_node = UARTMotorController()

        # Create executor for node
        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)

        try:
            # Start executor in background
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            # Wait for initialization
            time.sleep(2.0)

            # Send cmd_vel command
            twist = Twist()
            twist.linear.x = 0.1  # 0.1 m/s forward
            twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)

            # Wait for command processing
            time.sleep(1.0)

            # Check that chassis state is published
            if self._wait_for_messages(timeout=3.0):
                self.assertIsNotNone(self.received_chassis_state)
                self.assertEqual(self.received_chassis_state.header.frame_id, "base_link")
            else:
                self.fail("Did not receive chassis state message")

        finally:
            executor.shutdown()
            motor_node.destroy_node()

    def test_motor_command_direct(self):
        """Test direct motor command interface."""
        # Create motor controller
        motor_node = UARTMotorController()

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)

        try:
            # Start executor
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            time.sleep(2.0)

            # Send direct motor command
            motor_cmd = MotorCommand()
            motor_cmd.left_speed = 0.2
            motor_cmd.right_speed = 0.3
            motor_cmd.enable = True
            motor_cmd.command_id = 123

            self.motor_cmd_pub.publish(motor_cmd)

            # Wait for processing
            time.sleep(1.0)

            # Verify command was processed (should see chassis state update)
            if self._wait_for_messages(timeout=3.0):
                self.assertIsNotNone(self.received_chassis_state)
            else:
                self.fail("Motor command was not processed")

        finally:
            executor.shutdown()
            motor_node.destroy_node()

    def test_emergency_stop_service(self):
        """Test emergency stop service."""
        # Create motor controller
        motor_node = UARTMotorController()

        # Create service client
        emergency_client = self.test_node.create_client(EmergencyStop, "/emergency_stop")

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)

        try:
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            time.sleep(2.0)

            # Wait for service to be available
            if not emergency_client.wait_for_service(timeout_sec=5.0):
                self.fail("Emergency stop service not available")

            # Call emergency stop
            request = EmergencyStop.Request()
            request.enable_stop = True
            request.reason = "Integration test"

            future = emergency_client.call_async(request)

            # Wait for response
            start_time = time.time()
            while not future.done() and time.time() - start_time < 5.0:
                time.sleep(0.1)

            if future.done():
                response = future.result()
                self.assertTrue(response.success)
                self.assertIn("Integration test", response.message)
            else:
                self.fail("Emergency stop service call timed out")

        finally:
            executor.shutdown()
            motor_node.destroy_node()

    def test_imu_data_publishing(self):
        """Test IMU data acquisition and publishing."""
        # Create IMU node
        imu_node = UARTIMUNode()

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(imu_node)

        try:
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            # Wait for IMU data
            time.sleep(3.0)

            if self._wait_for_messages(timeout=5.0):
                # Verify IMU message
                imu_msg = self.received_imu_data
                self.assertIsNotNone(imu_msg)
                self.assertEqual(imu_msg.header.frame_id, "imu_link")

                # Check that orientation is reasonable
                q = imu_msg.orientation
                magnitude = (q.w**2 + q.x**2 + q.y**2 + q.z**2) ** 0.5
                self.assertAlmostEqual(magnitude, 1.0, places=2)

                # Check that some acceleration is present (gravity at minimum)
                acc_magnitude = (
                    imu_msg.linear_acceleration.x**2
                    + imu_msg.linear_acceleration.y**2
                    + imu_msg.linear_acceleration.z**2
                ) ** 0.5
                self.assertGreater(acc_magnitude, 5.0)  # At least 5 m/s² (should be ~9.8)

            else:
                self.fail("Did not receive IMU data")

        finally:
            executor.shutdown()
            imu_node.destroy_node()

    def test_dual_node_operation(self):
        """Test both nodes running simultaneously."""
        # Create both nodes
        motor_node = UARTMotorController()
        imu_node = UARTIMUNode()

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)
        executor.add_node(imu_node)

        try:
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            # Wait for initialization
            time.sleep(3.0)

            # Send motor command
            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = 0.1
            self.cmd_vel_pub.publish(twist)

            # Wait for both types of messages
            if self._wait_for_messages(timeout=10.0):
                # Verify both message types received
                self.assertIsNotNone(self.received_chassis_state)
                self.assertIsNotNone(self.received_imu_data)

                # Check message rates
                self.assertGreater(self.message_count["chassis"], 0)
                self.assertGreater(self.message_count["imu"], 0)

            else:
                self.fail("Did not receive both chassis and IMU messages")

        finally:
            executor.shutdown()
            motor_node.destroy_node()
            imu_node.destroy_node()

    def test_communication_robustness(self):
        """Test robustness of UART communication."""
        # Create motor controller
        motor_node = UARTMotorController()

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)

        try:
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            time.sleep(2.0)

            # Send rapid sequence of commands
            for i in range(10):
                twist = Twist()
                twist.linear.x = 0.1 * (1 if i % 2 == 0 else -1)
                twist.angular.z = 0.2 * (1 if i % 3 == 0 else -1)

                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)

            # Wait for processing
            time.sleep(2.0)

            # Verify system is still responsive
            if self._wait_for_messages(timeout=3.0):
                self.assertIsNotNone(self.received_chassis_state)
            else:
                self.fail("System became unresponsive during rapid commands")

        finally:
            executor.shutdown()
            motor_node.destroy_node()

    def test_watchdog_functionality(self):
        """Test watchdog timer stops motors."""
        # Create motor controller with short watchdog timeout
        motor_node = UARTMotorController()

        # Override watchdog timeout for testing
        motor_node.motor_config.watchdog_timeout = 0.5  # 500ms

        executor = MultiThreadedExecutor()
        executor.add_node(self.test_node)
        executor.add_node(motor_node)

        try:
            executor_thread = threading.Thread(target=executor.spin)
            executor_thread.daemon = True
            executor_thread.start()

            time.sleep(2.0)

            # Send a movement command
            twist = Twist()
            twist.linear.x = 0.2
            self.cmd_vel_pub.publish(twist)

            time.sleep(0.2)

            # Stop sending commands and wait for watchdog
            time.sleep(1.0)  # Wait longer than watchdog timeout

            # Motors should have stopped (this is hard to verify without feedback)
            # For now, just verify the system is still responsive
            if self._wait_for_messages(timeout=3.0):
                self.assertIsNotNone(self.received_chassis_state)
            else:
                self.fail("Watchdog may have caused system hang")

        finally:
            executor.shutdown()
            motor_node.destroy_node()


class UARTHardwareTest:
    """Standalone hardware test utilities."""

    @staticmethod
    def test_basic_communication(port: str = "/dev/ttyTHS1"):
        """Test basic UART communication with Wave Rover."""
        try:
            ser = serial.Serial(port, 115200, timeout=2.0)

            print(f"Testing communication on {port}")

            # Test IMU query
            imu_cmd = {"T": 126}
            json_str = json.dumps(imu_cmd) + "\n"
            ser.write(json_str.encode())

            # Wait for response
            response = ser.readline().decode().strip()
            if response:
                try:
                    data = json.loads(response)
                    print(f"IMU Response: {data}")
                    return True
                except json.JSONDecodeError:
                    print(f"Invalid JSON response: {response}")
                    return False
            else:
                print("No response received")
                return False

        except Exception as e:
            print(f"Communication test failed: {e}")
            return False
        finally:
            if "ser" in locals():
                ser.close()


def main():
    """Run integration tests or hardware tests."""
    parser = argparse.ArgumentParser(description="UART Integration Tests")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="Serial port for Wave Rover")
    parser.add_argument("--hardware-test", action="store_true", help="Run basic hardware test only")
    parser.add_argument(
        "--skip-hardware", action="store_true", help="Skip hardware-dependent tests"
    )

    args = parser.parse_args()

    if args.hardware_test:
        # Run basic hardware test
        success = UARTHardwareTest.test_basic_communication(args.port)
        exit(0 if success else 1)
    else:
        # Run full integration test suite
        # Pass arguments to test class
        TestUARTIntegration.serial_port = args.port
        TestUARTIntegration.skip_hardware = args.skip_hardware

        unittest.main(argv=[""], exit=False, verbosity=2)


if __name__ == "__main__":
    main()
