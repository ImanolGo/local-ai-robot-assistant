#!/usr/bin/env python3
"""UART IMU Node for Wave Rover 9-axis IMU.

This node handles IMU data acquisition from the Wave Rover chassis via UART.
It periodically queries the IMU (20 Hz) and publishes sensor_msgs/Imu messages
with proper coordinate frame transformations and data validation.

Features:
- Periodic IMU data queries at configurable rate
- Data validation and error checking
- Proper ROS2 IMU message formatting
- Coordinate frame transformations
- Robust error handling and recovery

Author: Local AI Robot Team
License: Apache-2.0
"""

import json
import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import rclpy
import serial
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# ROS2 messages
from sensor_msgs.msg import Imu
from std_msgs.msg import Header


@dataclass
class UARTConfig:
    """Configuration parameters for UART communication."""

    port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    timeout: float = 1.0
    write_timeout: float = 1.0


@dataclass
class IMUConfig:
    """Configuration parameters for IMU data processing."""

    query_rate: float = 20.0
    query_command: int = 126
    data_timeout: float = 0.1
    validate_data: bool = True
    angle_range: tuple = (-180, 180)
    acceleration_limit: float = 50.0


class UARTIMUNode(Node):
    """UART IMU Node for Wave Rover 9-axis IMU data acquisition."""

    def __init__(self):
        super().__init__("uart_imu_node")

        # Initialize parameters
        self._declare_parameters()
        self._load_config()

        # Initialize state variables
        self._last_imu_data: Optional[Dict[str, Any]] = None
        self._imu_sequence = 0

        # Thread safety
        self._serial_lock = threading.Lock()
        self._data_lock = threading.Lock()

        # Initialize serial connection
        self._serial: Optional[serial.Serial] = None
        self._connect_serial()

        # Set up QoS profile for sensor data
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=50,
        )

        # Publishers
        self.imu_pub = self.create_publisher(Imu, "/imu/data", qos_sensor)

        # Timer for periodic IMU queries
        self.imu_timer = self.create_timer(
            1.0 / self.imu_config.query_rate, self._imu_timer_callback
        )

        # Timer for processing received data
        self.process_timer = self.create_timer(
            0.01, self._process_data_callback  # Check for new data every 10ms
        )

        self.get_logger().info(f"UART IMU Node initialized on {self.uart_config.port}")
        self.get_logger().info(f"IMU query rate: {self.imu_config.query_rate} Hz")

        # Enable continuous feedback at startup
        self._enable_continuous_feedback()

    def _enable_continuous_feedback(self):
        """Enable continuous feedback mode and disable echo."""
        self.get_logger().info('Disabling serial echo ({"T": 143, "cmd": 0})...')
        self._send_command({"T": 143, "cmd": 0})

        self.get_logger().info('Enabling continuous feedback ({"T": 131, "cmd": 1})...')
        command = {"T": 131, "cmd": 1}
        if self._send_command(command):
            self.get_logger().info("Continuous feedback enabled")
        else:
            self.get_logger().warn("Failed to enable continuous feedback")

    def _declare_parameters(self):
        """Declare ROS2 parameters with default values."""
        # UART parameters
        self.declare_parameter("uart.port", "/dev/ttyTHS1")
        self.declare_parameter("uart.baudrate", 115200)
        self.declare_parameter("uart.timeout", 1.0)
        self.declare_parameter("uart.write_timeout", 1.0)

        # IMU parameters
        self.declare_parameter("imu.query_rate", 20.0)
        self.declare_parameter("imu.query_command", 126)
        self.declare_parameter("imu.data_timeout", 0.1)
        self.declare_parameter("imu.validate_data", True)
        self.declare_parameter("imu.acceleration_limit", 50.0)

        # Frame ID
        self.declare_parameter("frame_id", "imu_link")

    def _load_config(self):
        """Load configuration from parameters."""
        try:
            # Try to find config file in common locations
            import os

            from robot_interfaces.config_utils import ConfigError, ROS2ConfigLoader

            workspace_root = os.path.expanduser("~/repos/local-ai-robot-assistant")
            config_path = os.path.join(workspace_root, "config")

            loader = ROS2ConfigLoader(self, config_dir=config_path)

            try:
                # Load and declare parameters from config file
                _ = loader.load_and_declare_parameters(
                    "uart_config.yaml",
                    parameter_mapping={
                        "uart_config.port": "uart.port",
                        "uart_config.baudrate": "uart.baudrate",
                        "uart_config.imu_node.query_rate": "imu.query_rate",
                        # Add other mappings as needed or use defaults
                    },
                )
                self.get_logger().info("Loaded configuration from uart_config.yaml")
            except ConfigError as e:
                self.get_logger().warn(
                    f"Could not load uart_config.yaml: {e}. Using defaults/launch params."
                )

        except ImportError:
            self.get_logger().warn(
                "robot_interfaces.config_utils not found. Using defaults/launch params."
            )

        # Load from parameters (which may have been set by launch file or ConfigLoader)
        self.uart_config = UARTConfig(
            port=self.get_parameter("uart.port").get_parameter_value().string_value,
            baudrate=self.get_parameter("uart.baudrate").get_parameter_value().integer_value,
            timeout=self.get_parameter("uart.timeout").get_parameter_value().double_value,
            write_timeout=self.get_parameter("uart.write_timeout")
            .get_parameter_value()
            .double_value,
        )

        self.imu_config = IMUConfig(
            query_rate=self.get_parameter("imu.query_rate").get_parameter_value().double_value,
            query_command=self.get_parameter("imu.query_command")
            .get_parameter_value()
            .integer_value,
            data_timeout=self.get_parameter("imu.data_timeout").get_parameter_value().double_value,
            validate_data=self.get_parameter("imu.validate_data").get_parameter_value().bool_value,
            acceleration_limit=self.get_parameter("imu.acceleration_limit")
            .get_parameter_value()
            .double_value,
        )

        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

    def _connect_serial(self) -> bool:
        """Establish serial connection to Wave Rover."""
        try:
            self._serial = serial.Serial(
                port=self.uart_config.port,
                baudrate=self.uart_config.baudrate,
                timeout=self.uart_config.timeout,
                write_timeout=self.uart_config.write_timeout,
            )

            # Flush any existing data
            self._serial.flushInput()
            self._serial.flushOutput()

            self.get_logger().info(f"Serial connection established: {self.uart_config.port}")
            return True

        except Exception as e:
            self.get_logger().error(
                f"Failed to connect to serial port {self.uart_config.port}: {e}"
            )
            return False

    def _send_command(self, command: Dict[str, Any]) -> bool:
        """Send JSON command via UART with error handling."""
        if not self._serial or not self._serial.is_open:
            self.get_logger().warn("Serial port not available")
            return False

        try:
            with self._serial_lock:
                json_str = json.dumps(command) + "\n"
                self._serial.write(json_str.encode())
                self.get_logger().debug(f"Sent IMU query: {command}")
                return True

        except Exception as e:
            self.get_logger().error(f"UART send error: {e}")
            return False

    def _read_response(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON response from UART."""
        if not self._serial or not self._serial.is_open:
            return None

        try:
            with self._serial_lock:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        try:
                            data = json.loads(line)
                            # Ignore echoes
                            if "T" in data and data["T"] in [
                                self.imu_config.query_command,
                                131,
                                143,
                            ]:
                                return None
                            self.get_logger().debug(f"Received data: {data}")
                            return data
                        except json.JSONDecodeError:
                            self.get_logger().debug(f"Invalid JSON received: {line}")
                            return None
        except Exception as e:
            self.get_logger().error(f"UART read error: {e}")
            return None

        return None

    def _validate_imu_data(self, data: Dict[str, Any]) -> bool:
        """Validate received IMU data."""
        if not self.imu_config.validate_data:
            return True

        try:
            if "T" not in data:
                return False

            cmd_type = data["T"]

            # Support both Chassis Feedback (1001) and IMU Data (1002/126)
            if cmd_type == 1001:
                # Chassis feedback only has roll, pitch, yaw
                required = ["r", "p", "y"]
            elif cmd_type in [126, 1002]:
                # IMU raw data should have more
                required = ["r", "p", "y", "ax", "ay", "az", "gx", "gy", "gz"]
            else:
                return False

            for field in required:
                if field not in data:
                    return False
                if not isinstance(data[field], (int, float)):
                    return False

            # Validate angle ranges
            for angle_field in ["r", "p", "y"]:
                angle = data[angle_field]
                if not (self.imu_config.angle_range[0] <= angle <= self.imu_config.angle_range[1]):
                    self.get_logger().debug(f"Angle out of range {angle_field}: {angle}")
                    return False

            return True

        except Exception as e:
            self.get_logger().debug(f"Data validation error: {e}")
            self.get_logger().warn(f"Validation failed for data: {data}")
            return False

    def _degrees_to_radians(self, degrees: float) -> float:
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles (in degrees) to quaternion.

        Uses the standard ZYX (aerospace) convention, matching the
        uart_motor_controller implementation.
        """
        roll_rad = self._degrees_to_radians(roll)
        pitch_rad = self._degrees_to_radians(pitch)
        yaw_rad = self._degrees_to_radians(yaw)

        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    def _create_imu_message(self, data: Dict[str, Any]) -> Imu:
        """Create ROS2 IMU message from parsed data."""
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.frame_id

        # Orientation (present in both types)
        imu_msg.orientation = self._euler_to_quaternion(data["r"], data["p"], data["y"])
        imu_msg.orientation_covariance = [
            0.01,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.01,
        ]

        # Handle angular velocity
        if "gx" in data and "gy" in data and "gz" in data:
            imu_msg.angular_velocity.x = self._degrees_to_radians(data["gx"])
            imu_msg.angular_velocity.y = self._degrees_to_radians(data["gy"])
            imu_msg.angular_velocity.z = self._degrees_to_radians(data["gz"])
            imu_msg.angular_velocity_covariance = [
                0.001,
                0.0,
                0.0,
                0.0,
                0.001,
                0.0,
                0.0,
                0.0,
                0.001,
            ]
        else:
            # Set high covariance (or -1.0 according to some conventions,
            # but robot_localization prefers large positive values if it should ignore it)
            # Actually, standard is to set the first element to -1 if not available.
            imu_msg.angular_velocity_covariance[0] = -1.0

        # Handle linear acceleration
        if "ax" in data and "ay" in data and "az" in data:
            # Wave Rover sends mg
            g_to_ms2 = 9.80665 / 1000.0
            imu_msg.linear_acceleration.x = data["ax"] * g_to_ms2
            imu_msg.linear_acceleration.y = data["ay"] * g_to_ms2
            imu_msg.linear_acceleration.z = data["az"] * g_to_ms2
            imu_msg.linear_acceleration_covariance = [
                0.01,
                0.0,
                0.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.0,
                0.01,
            ]
        else:
            imu_msg.linear_acceleration_covariance[0] = -1.0

        return imu_msg

    def _imu_timer_callback(self):
        """Periodic IMU data query."""
        self._send_command({"T": self.imu_config.query_command})

    def _process_data_callback(self):
        """Process all available data from UART and publish IMU messages."""
        while self._serial and self._serial.is_open and self._serial.in_waiting > 0:
            data = self._read_response()
            if data is None:
                continue

            if not self._validate_imu_data(data):
                if self._imu_sequence % 50 == 0:
                    self.get_logger().info(f"Ignoring message of type {data.get('T')}")
                continue

            try:
                imu_msg = self._create_imu_message(data)
                self.imu_pub.publish(imu_msg)

                with self._data_lock:
                    self._last_imu_data = data
                    self._imu_sequence += 1

                if self._imu_sequence % 100 == 0:
                    self.get_logger().info(f"Published {self._imu_sequence} IMU messages")

            except Exception as e:
                self.get_logger().error(f"Error processing IMU data: {e}")

    def destroy_node(self):
        """Cleanup on node shutdown."""
        if self._serial and self._serial.is_open:
            self.get_logger().info("Closing serial connection")
            self._serial.close()

        super().destroy_node()


def main(args=None):
    """Main entry point for the UART IMU node."""
    rclpy.init(args=args)

    try:
        node = UARTIMUNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in IMU node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
