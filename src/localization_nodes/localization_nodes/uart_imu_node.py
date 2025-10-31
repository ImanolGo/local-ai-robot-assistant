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
                    line = self._serial.readline().decode("utf-8").strip()
                    if line:
                        try:
                            data = json.loads(line)
                            self.get_logger().debug(f"Received IMU data: {data}")
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
            # Check for required fields (based on Wave Rover protocol)
            required_fields = [
                "roll",
                "pitch",
                "yaw",
                "AccX",
                "AccY",
                "AccZ",
                "GyroX",
                "GyroY",
                "GyroZ",
            ]

            for field in required_fields:
                if field not in data:
                    self.get_logger().debug(f"Missing required field: {field}")
                    return False

                value = data[field]
                if not isinstance(value, (int, float)):
                    self.get_logger().debug(f"Invalid data type for {field}: {type(value)}")
                    return False

            # Validate angle ranges
            for angle_field in ["roll", "pitch", "yaw"]:
                angle = data[angle_field]
                if not (self.imu_config.angle_range[0] <= angle <= self.imu_config.angle_range[1]):
                    self.get_logger().debug(f"Angle out of range {angle_field}: {angle}")
                    return False

            # Validate acceleration values
            for acc_field in ["AccX", "AccY", "AccZ"]:
                acc = data[acc_field]
                if abs(acc) > self.imu_config.acceleration_limit:
                    self.get_logger().debug(f"Acceleration out of range {acc_field}: {acc}")
                    return False

            return True

        except Exception as e:
            self.get_logger().debug(f"Data validation error: {e}")
            return False

    def _degrees_to_radians(self, degrees: float) -> float:
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles (in degrees) to quaternion.

        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees

        Returns:
            geometry_msgs/Quaternion
        """
        # Convert to radians
        roll_rad = self._degrees_to_radians(roll)
        pitch_rad = self._degrees_to_radians(pitch)
        yaw_rad = self._degrees_to_radians(yaw)

        # Compute quaternion
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        q = Quaternion()
        q.w = cy * cp * cr + sy * sp * sr
        q.x = cy * cp * sr - sy * sp * cr
        q.y = sy * cp * sr + cy * sp * cr
        q.z = sy * cp * cr - cy * sp * sr

        return q

    def _create_imu_message(self, data: Dict[str, Any]) -> Imu:
        """Create ROS2 IMU message from parsed data."""
        imu_msg = Imu()

        # Header
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.frame_id

        # Orientation (quaternion from Euler angles)
        imu_msg.orientation = self._euler_to_quaternion(data["roll"], data["pitch"], data["yaw"])

        # Orientation covariance (diagonal matrix with moderate uncertainty)
        # Since we're using Euler angles, there's some uncertainty
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

        # Angular velocity (rad/s)
        imu_msg.angular_velocity.x = self._degrees_to_radians(data["GyroX"])
        imu_msg.angular_velocity.y = self._degrees_to_radians(data["GyroY"])
        imu_msg.angular_velocity.z = self._degrees_to_radians(data["GyroZ"])

        # Angular velocity covariance
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

        # Linear acceleration (m/s²)
        # Note: Wave Rover provides acceleration in g's, convert to m/s²
        g = 9.80665  # Standard gravity
        imu_msg.linear_acceleration.x = data["AccX"] * g
        imu_msg.linear_acceleration.y = data["AccY"] * g
        imu_msg.linear_acceleration.z = data["AccZ"] * g

        # Linear acceleration covariance
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

        return imu_msg

    def _imu_timer_callback(self):
        """Periodic IMU data query."""
        # Send IMU query command
        command = {"T": self.imu_config.query_command}
        success = self._send_command(command)

        if not success:
            self.get_logger().debug("Failed to send IMU query")

    def _process_data_callback(self):
        """Process received data and publish IMU messages."""
        # Try to read response
        data = self._read_response()

        if data is None:
            return

        # Validate data
        if not self._validate_imu_data(data):
            self.get_logger().debug("Invalid IMU data received")
            return

        try:
            # Create and publish IMU message
            imu_msg = self._create_imu_message(data)
            self.imu_pub.publish(imu_msg)

            # Store last valid data
            with self._data_lock:
                self._last_imu_data = data
                self._imu_sequence += 1

            self.get_logger().debug(f"Published IMU data #{self._imu_sequence}")

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
