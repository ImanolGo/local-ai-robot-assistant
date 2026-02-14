#!/usr/bin/env python3
"""UART Motor Controller Node for Wave Rover Chassis.

This node handles motor control communication with the Wave Rover chassis via UART.
It subscribes to cmd_vel topics, converts to differential drive commands, and manages
safety features like watchdog timers and emergency stop.

Features:
- Differential drive kinematics conversion
- Watchdog timer for safety
- Emergency stop service
- Continuous feedback monitoring
- Motor status publishing
- Robust error handling and recovery

Author: Local AI Robot Team
License: Apache-2.0
"""

import json

# ROS2 messages
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rclpy
import serial
from geometry_msgs.msg import Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

# Custom messages and services
from robot_interfaces.msg import ChassisState, MotorCommand
from robot_interfaces.srv import EmergencyStop, SetMode


@dataclass
class UARTConfig:
    """Configuration parameters for UART communication."""

    port: str = "/dev/ttyTHS1"
    baudrate: int = 115200
    timeout: float = 1.0
    write_timeout: float = 1.0


@dataclass
class MotorConfig:
    """Configuration parameters for motor control."""

    command_rate: float = 20.0
    watchdog_timeout: float = 0.5
    max_speed: float = 0.5
    wheelbase: float = 0.16
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 1.0
    emergency_stop_enabled: bool = True
    acceleration_limit: float = 0.5
    deceleration_limit: float = 1.0


class UARTMotorController(Node):
    """UART Motor Controller Node for Wave Rover chassis communication."""

    def __init__(self):
        super().__init__("uart_motor_controller")

        # Initialize parameters
        self._declare_parameters()
        self._load_config()

        # Initialize state variables
        self._emergency_stop_active = False
        self._motors_enabled = True
        self._last_command_time = time.time()
        self._last_cmd_vel = Twist()
        self._current_wheel_speeds = {"left": 0.0, "right": 0.0}
        self._commands_received = False  # Track if any commands have been received
        self._last_nonzero_command_time = (
            0.0  # Track when we last received non-zero motion commands
        )
        self._node_start_time = time.time()  # Track when node started

        # Thread safety
        self._serial_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Response handling
        self._reader_thread = None
        self._stop_reader = threading.Event()

        # Response health tracking
        self._last_response_time: Optional[float] = None
        self._response_count = 0
        self._last_no_response_warning_time = 0.0

        # Initialize serial connection
        self._serial: Optional[serial.Serial] = None
        self._connect_serial()

        # Set up QoS profiles - use BEST_EFFORT for sensor data to avoid compatibility issues
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Changed to BEST_EFFORT for compatibility
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Publishers
        self.motor_status_pub = self.create_publisher(ChassisState, "/motor_status", qos_reliable)

        self.chassis_state_pub = self.create_publisher(ChassisState, "/chassis_state", qos_reliable)

        self.odom_raw_pub = self.create_publisher(Odometry, "/odom_raw", qos_sensor)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", qos_sensor)

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, qos_reliable
        )

        self.motor_command_sub = self.create_subscription(
            MotorCommand, "/motor_command", self._motor_command_callback, qos_reliable
        )

        # Services
        self.emergency_stop_srv = self.create_service(
            EmergencyStop, "/emergency_stop", self._emergency_stop_callback
        )

        self.set_mode_srv = self.create_service(SetMode, "/set_mode", self._set_mode_callback)

        # Timers
        self.command_timer = self.create_timer(
            1.0 / self.motor_config.command_rate, self._command_timer_callback
        )

        self.watchdog_timer = self.create_timer(0.1, self._watchdog_callback)  # Check every 100ms

        self.status_timer = self.create_timer(
            0.1, self._status_timer_callback  # Publish status at 10Hz
        )

        # Add chassis feedback request timer (reduced frequency since continuous
        # feedback provides data)
        self.chassis_feedback_timer = self.create_timer(
            1.0, self._chassis_feedback_callback
        )  # 1Hz chassis requests

        # Add IMU request timer (reduced since continuous feedback includes IMU)
        self.imu_timer = self.create_timer(0.5, self._imu_timer_callback)  # 2Hz IMU requests

        # Enable continuous feedback mode
        if self._serial and self._serial.is_open:
            self._start_response_reader()  # Start reader FIRST
            time.sleep(0.1)  # Brief delay to ensure reader is ready
            self._enable_continuous_feedback()

        self.get_logger().info(f"UART Motor Controller initialized on {self.uart_config.port}")

    def _declare_parameters(self):
        """Declare ROS2 parameters with default values."""
        # UART parameters
        self.declare_parameter("uart.port", "/dev/ttyTHS1")
        self.declare_parameter("uart.baudrate", 115200)
        self.declare_parameter("uart.timeout", 1.0)
        self.declare_parameter("uart.write_timeout", 1.0)

        # Motor parameters
        self.declare_parameter("motor.command_rate", 20.0)
        self.declare_parameter("motor.watchdog_timeout", 0.5)
        self.declare_parameter("motor.max_speed", 0.5)
        self.declare_parameter("motor.wheelbase", 0.16)
        self.declare_parameter("motor.max_linear_velocity", 0.3)
        self.declare_parameter("motor.max_angular_velocity", 1.0)
        self.declare_parameter("motor.emergency_stop_enabled", True)
        self.declare_parameter("motor.acceleration_limit", 0.5)
        self.declare_parameter("motor.deceleration_limit", 1.0)

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

        self.motor_config = MotorConfig(
            command_rate=self.get_parameter("motor.command_rate")
            .get_parameter_value()
            .double_value,
            watchdog_timeout=self.get_parameter("motor.watchdog_timeout")
            .get_parameter_value()
            .double_value,
            max_speed=self.get_parameter("motor.max_speed").get_parameter_value().double_value,
            wheelbase=self.get_parameter("motor.wheelbase").get_parameter_value().double_value,
            max_linear_velocity=self.get_parameter("motor.max_linear_velocity")
            .get_parameter_value()
            .double_value,
            max_angular_velocity=self.get_parameter("motor.max_angular_velocity")
            .get_parameter_value()
            .double_value,
            emergency_stop_enabled=self.get_parameter("motor.emergency_stop_enabled")
            .get_parameter_value()
            .bool_value,
            acceleration_limit=self.get_parameter("motor.acceleration_limit")
            .get_parameter_value()
            .double_value,
            deceleration_limit=self.get_parameter("motor.deceleration_limit")
            .get_parameter_value()
            .double_value,
        )

    def _connect_serial(self) -> bool:
        """Establish serial connection to Wave Rover."""
        try:
            self._serial = serial.Serial(
                port=self.uart_config.port,
                baudrate=self.uart_config.baudrate,
                timeout=self.uart_config.timeout,
                write_timeout=self.uart_config.write_timeout,
            )

            # Critical: Disable hardware flow control as per Wave Rover docs
            self._serial.setRTS(False)
            self._serial.setDTR(False)

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
            self.get_logger().warn("Serial port not available, attempting reconnection")
            if not self._connect_serial():
                return False

        try:
            with self._serial_lock:
                json_str = json.dumps(command) + "\n"
                self._serial.write(json_str.encode())
                self._serial.flush()  # Ensure data is sent
                self.get_logger().debug(f"Sent command: {command}")
                return True

        except serial.SerialException as e:
            self.get_logger().error(f"UART send error: {e}")
            # Attempt to reconnect on serial errors
            self._serial = None
            return False
        except Exception as e:
            self.get_logger().error(f"Unexpected error sending command: {e}")
            return False

    def _enable_continuous_feedback(self) -> bool:
        """Enable continuous feedback mode on Wave Rover."""
        command = {"T": 131, "cmd": 1}
        success = self._send_command(command)
        if success:
            self.get_logger().info("Continuous feedback mode enabled")
        else:
            self.get_logger().error("Failed to enable continuous feedback")
        return success

    def _start_response_reader(self):
        """Start background thread to read serial responses."""
        if self._reader_thread and self._reader_thread.is_alive():
            return

        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()

    def _read_responses(self):
        """Background thread to continuously read serial responses."""
        while not self._stop_reader.is_set() and self._serial and self._serial.is_open:
            try:
                if self._serial.in_waiting > 0:
                    line = self._serial.readline()
                    if line:
                        text = ""  # Initialize text variable
                        try:
                            # Try UTF-8 decode with error handling
                            text = line.decode("utf-8", errors="ignore").strip()

                            # Wave Rover sends mixed binary/JSON data, extract JSON portion
                            if text and "{" in text and "}" in text:
                                # Find the JSON part within the mixed data
                                start_idx = text.find("{")
                                end_idx = text.rfind("}") + 1
                                if start_idx >= 0 and end_idx > start_idx:
                                    json_part = text[start_idx:end_idx]
                                    try:
                                        response = json.loads(json_part)
                                        self._process_response(response)
                                    except json.JSONDecodeError:
                                        self.get_logger().debug(
                                            f"Invalid JSON extracted: {json_part}"
                                        )
                            elif text and not any(ord(c) < 32 for c in text if c not in "\r\n\t"):
                                # Log clean text that isn't JSON
                                self.get_logger().debug(f"Non-JSON text: {text}")

                        except Exception as e:
                            self.get_logger().debug(f"Response processing error: {e}")
                else:
                    # Small sleep to prevent 100% CPU usage when no data is available
                    time.sleep(0.001)

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except serial.SerialException as e:
                if "device reports readiness to read but returned no data" in str(e):
                    # This happens when multiple processes access the same port
                    self.get_logger().debug("Serial port contention detected")
                    time.sleep(0.1)  # Back off for a moment
                else:
                    self.get_logger().error(f"Serial read error: {e}")
                    break
            except Exception as e:
                self.get_logger().error(f"Unexpected read error: {e}")
                break

    def _process_response(self, response: Dict[str, Any]):
        """Process incoming JSON responses from Wave Rover."""
        try:
            # Track response health
            self._last_response_time = time.time()
            self._response_count += 1

            # Log important responses
            if "T" in response:
                cmd_type = response.get("T")
                self.get_logger().debug(f"Processing response type T={cmd_type}")
                if cmd_type == 1001:  # Continuous feedback (actual Wave Rover format)
                    self._update_continuous_feedback(response)
                elif cmd_type == 1002:  # Full IMU response (reply to T=126)
                    self._update_imu_data(response)
                elif cmd_type == 130:  # Chassis feedback
                    self._update_chassis_state(response)
                elif cmd_type == 126:  # IMU request echo (no data)
                    pass  # Wave Rover echoes T=126 then sends T=1002
            else:
                self.get_logger().debug(f"Response without T field: {response}")

        except Exception as e:
            self.get_logger().error(f"Response processing error: {e}")
            self.get_logger().debug(f"Problematic response: {response}")

    def _update_continuous_feedback(self, response: Dict[str, Any]):
        """Update state from Wave Rover continuous feedback (T=1001)."""
        # Parse the actual Wave Rover feedback format
        # {"T": 1001, "L": 0, "R": 0, "r": roll, "p": pitch, "y": yaw, "temp": temp, "v": voltage}
        try:
            # Create and publish chassis state immediately with real data
            chassis_msg = ChassisState()
            chassis_msg.header = Header()
            chassis_msg.header.stamp = self.get_clock().now().to_msg()
            chassis_msg.header.frame_id = "base_link"

            with self._state_lock:
                chassis_msg.motors_enabled = (
                    self._motors_enabled and not self._emergency_stop_active
                )
                chassis_msg.emergency_stop_active = self._emergency_stop_active

                # Parse motor speeds
                if "L" in response:
                    chassis_msg.left_motor_current = float(response["L"])
                if "R" in response:
                    chassis_msg.right_motor_current = float(response["R"])

                # Parse IMU data from continuous feedback
                if "r" in response:
                    chassis_msg.roll = float(response["r"])
                if "p" in response:
                    chassis_msg.pitch = float(response["p"])
                if "y" in response:
                    chassis_msg.yaw = float(response["y"])
                    chassis_msg.heading_angle = float(response["y"])
                if "v" in response:
                    chassis_msg.battery_voltage = float(response["v"])
                if "temp" in response:
                    chassis_msg.temperature = float(response["temp"])
                    # Estimate power consumption from temperature (rough approximation)
                    base_temp = 25.0  # Assume 25°C baseline
                    temp_diff = max(0, chassis_msg.temperature - base_temp)
                    chassis_msg.power_consumption = temp_diff * 0.1  # Rough estimate

            # Publish chassis state with real data
            try:
                self.chassis_state_pub.publish(chassis_msg)
                self.get_logger().info(
                    f"Published chassis state: roll={chassis_msg.roll:.2f}, \
                        pitch={chassis_msg.pitch:.2f}, yaw={chassis_msg.yaw:.2f},\
                                voltage={chassis_msg.battery_voltage:.2f}V,\
                                      temp={chassis_msg.temperature:.1f}°C"
                )
            except Exception as pub_error:
                self.get_logger().error(f"Error publishing chassis state: {pub_error}")

        except Exception as e:
            self.get_logger().warn(f"Error processing continuous feedback: {e}")
            # Log the problematic response for debugging
            self.get_logger().debug(f"Problematic response: {response}")

        # Also publish as standard Imu message for EKF to ensure high frequency updates
        self._publish_imu_message(response)

    def _update_chassis_state(self, response: Dict[str, Any]):
        """Update chassis state from Wave Rover feedback."""
        # Process chassis feedback response (T=130)
        # Only process if response contains actual data (r, p, y, etc.)
        if "T" in response and response["T"] == 130:
            if any(k in response for k in ["r", "p", "y", "v", "temp"]):
                self.get_logger().debug(f"Chassis feedback with data: {response}")
                self._update_continuous_feedback(response)
            else:
                self.get_logger().debug("Empty chassis feedback response (T=130), skipping")

    def _update_imu_data(self, response: Dict[str, Any]):
        """Update IMU data from Wave Rover feedback."""
        self._publish_imu_message(response)

    def _degrees_to_radians(self, degrees: float) -> float:
        """Convert degrees to radians."""
        return degrees * math.pi / 180.0

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles (in degrees) to quaternion."""
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

    def _publish_imu_message(self, data: Dict[str, Any]):
        """Create and publish Imu message from UART data."""
        try:
            imu_msg = Imu()
            imu_msg.header = Header()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = "imu_link"

            # Orientation
            if all(k in data for k in ["r", "p", "y"]):
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
            else:
                imu_msg.orientation.w = 1.0
                imu_msg.orientation_covariance[0] = -1.0

            # Angular Velocity
            if all(k in data for k in ["gx", "gy", "gz"]):
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
                imu_msg.angular_velocity_covariance[0] = -1.0

            # Linear Acceleration
            if all(k in data for k in ["ax", "ay", "az"]):
                # Wave Rover transmits acceleration in milli-g (mg).
                # e.g. az ~973 mg ≈ 1g. Convert to m/s²: multiply by 9.80665/1000.
                MG_TO_MS2 = 9.80665 / 1000.0
                imu_msg.linear_acceleration.x = float(data["ax"]) * MG_TO_MS2
                imu_msg.linear_acceleration.y = float(data["ay"]) * MG_TO_MS2
                imu_msg.linear_acceleration.z = float(data["az"]) * MG_TO_MS2
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

            self.imu_pub.publish(imu_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing IMU message: {e}")

    def _imu_timer_callback(self):
        """Request IMU data from Wave Rover."""
        if self._serial and self._serial.is_open:
            imu_command = {"T": 126}
            self._send_command(imu_command)

    def _chassis_feedback_callback(self):
        """Request chassis feedback from Wave Rover."""
        if self._serial and self._serial.is_open:
            chassis_command = {"T": 130}
            self._send_command(chassis_command)

    def _twist_to_wheel_speeds(self, twist: Twist) -> Tuple[float, float]:
        """Convert Twist message to left/right wheel speeds using differential drive kinematics.

        Args:
            twist: Twist message with linear.x and angular.z velocities

        Returns:
            Tuple of (left_speed, right_speed) in range [-0.5, 0.5]
        """
        # Limit input velocities
        linear_vel = max(
            -self.motor_config.max_linear_velocity,
            min(self.motor_config.max_linear_velocity, twist.linear.x),
        )
        angular_vel = max(
            -self.motor_config.max_angular_velocity,
            min(self.motor_config.max_angular_velocity, twist.angular.z),
        )

        # Differential drive kinematics
        # v_left = v - (w * wheelbase) / 2
        # v_right = v + (w * wheelbase) / 2
        v_left = linear_vel - (angular_vel * self.motor_config.wheelbase) / 2.0
        v_right = linear_vel + (angular_vel * self.motor_config.wheelbase) / 2.0

        # Convert to Wave Rover speed format (-0.5 to +0.5)
        # Assuming max wheel speed corresponds to max_linear_velocity
        max_wheel_speed = self.motor_config.max_linear_velocity
        left_speed = np.clip(
            v_left / max_wheel_speed * self.motor_config.max_speed,
            -self.motor_config.max_speed,
            self.motor_config.max_speed,
        )
        right_speed = np.clip(
            v_right / max_wheel_speed * self.motor_config.max_speed,
            -self.motor_config.max_speed,
            self.motor_config.max_speed,
        )

        return left_speed, right_speed

    def _send_motor_command(self, left_speed: float, right_speed: float) -> bool:
        """Send motor speed command to Wave Rover."""
        if self._emergency_stop_active:
            left_speed = 0.0
            right_speed = 0.0

        command = {
            "T": 1,  # Speed control command
            "L": float(left_speed),
            "R": float(right_speed),
        }

        success = self._send_command(command)
        if success:
            with self._state_lock:
                self._current_wheel_speeds["left"] = left_speed
                self._current_wheel_speeds["right"] = right_speed

        return success

    def _cmd_vel_callback(self, msg: Twist):
        """Handle incoming cmd_vel messages."""
        with self._state_lock:
            self._last_cmd_vel = msg
            self._last_command_time = time.time()  # Update only on actual commands
            self._commands_received = True  # Mark that we've received commands

            # Track when we receive non-zero motion commands
            if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
                self._last_nonzero_command_time = time.time()

    def _motor_command_callback(self, msg: MotorCommand):
        """Handle direct motor command messages."""
        if msg.enable and not self._emergency_stop_active:
            success = self._send_motor_command(msg.left_speed, msg.right_speed)
            if not success:
                self.get_logger().warn("Failed to send motor command")

        with self._state_lock:
            self._last_command_time = time.time()  # Update only on actual commands
            self._commands_received = True  # Mark that we've received commands

    def _command_timer_callback(self):
        """Periodic command sending based on last received cmd_vel."""
        if self._emergency_stop_active:
            return

        with self._state_lock:
            cmd_vel = self._last_cmd_vel
            time_since_command = time.time() - self._last_command_time

        # Only send commands if we recently received cmd_vel or if motors need to stop
        if time_since_command < self.motor_config.watchdog_timeout:
            # Convert twist to wheel speeds
            left_speed, right_speed = self._twist_to_wheel_speeds(cmd_vel)
            # Send motor command
            self._send_motor_command(left_speed, right_speed)

    def _watchdog_callback(self):
        """Watchdog timer to stop motors if no commands received."""
        current_time = time.time()

        with self._state_lock:
            time_since_command = current_time - self._last_command_time
            time_since_nonzero_command = current_time - self._last_nonzero_command_time
            motors_are_running = any(
                abs(speed) > 0.01 for speed in self._current_wheel_speeds.values()
            )
            commands_received = self._commands_received

        # Conservative watchdog: only trigger if we've been running for a while
        # and we had recent motion commands but now have stopped receiving them
        node_running_time = current_time - self._node_start_time
        had_recent_motion = self._last_nonzero_command_time > 0 and time_since_nonzero_command < 2.0

        if (
            node_running_time > 5.0  # Only after node has been running for 5 seconds
            and commands_received
            and had_recent_motion
            and motors_are_running
            and time_since_command > self.motor_config.watchdog_timeout
        ):
            self.get_logger().warn(f"Watchdog timeout ({time_since_command:.2f}s), stopping motors")
            self._send_motor_command(0.0, 0.0)

    def _status_timer_callback(self):
        """Publish motor status periodically.

        Note: Chassis state is published from _update_continuous_feedback with real
        sensor data. This timer only publishes motor_status for monitoring.
        """
        # Only publish motor_status (not chassis_state which has real data)
        status_msg = ChassisState()
        status_msg.header = Header()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.header.frame_id = "base_link"

        with self._state_lock:
            status_msg.motors_enabled = self._motors_enabled and not self._emergency_stop_active
            status_msg.emergency_stop_active = self._emergency_stop_active

        self.motor_status_pub.publish(status_msg)

        # Log occasionally for debugging
        if hasattr(self, "_status_counter"):
            self._status_counter += 1
        else:
            self._status_counter = 1

        if self._status_counter % 50 == 0:  # Every 5 seconds at 10Hz
            self.get_logger().debug(f"Published chassis state #{self._status_counter}")

        # Warn if no responses received from Wave Rover
        now = time.time()
        time_since_start = now - self._node_start_time
        if time_since_start > 5.0:  # Only check after 5s startup grace period
            if self._last_response_time is None:
                # Never received any response
                if now - self._last_no_response_warning_time > 10.0:
                    self._last_no_response_warning_time = now
                    self.get_logger().warn(
                        "No responses received from Wave Rover! "
                        "Check: Is it powered on? Is the UART cable connected? "
                        f"(port={self.uart_config.port}, "
                        f"uptime={time_since_start:.0f}s, "
                        f"responses={self._response_count})"
                    )
            elif now - self._last_response_time > 10.0:
                # Responses stopped arriving
                if now - self._last_no_response_warning_time > 10.0:
                    self._last_no_response_warning_time = now
                    gap = now - self._last_response_time
                    self.get_logger().warn(
                        f"Wave Rover stopped responding {gap:.0f}s ago! "
                        f"(total responses={self._response_count})"
                    )

    def _emergency_stop_callback(
        self, request: EmergencyStop.Request, response: EmergencyStop.Response
    ) -> EmergencyStop.Response:
        """Handle emergency stop service requests."""
        try:
            if request.enable_stop:
                self._emergency_stop_active = True
                self._send_motor_command(0.0, 0.0)
                self.get_logger().warn(f"Emergency stop activated: {request.reason}")
                response.message = f"Emergency stop activated: {request.reason}"
            else:
                self._emergency_stop_active = False
                self.get_logger().info("Emergency stop deactivated")
                response.message = "Emergency stop deactivated"

            response.success = True

        except Exception as e:
            self.get_logger().error(f"Emergency stop service error: {e}")
            response.success = False
            response.message = f"Error: {e}"

        return response

    def _set_mode_callback(
        self, request: SetMode.Request, response: SetMode.Response
    ) -> SetMode.Response:
        """Handle set mode service requests."""
        # Implementation depends on SetMode message definition
        # For now, just acknowledge
        response.success = True
        response.message = f"Mode set request received: {request}"
        return response

    def destroy_node(self):
        """Cleanup on node shutdown."""
        # Stop reader thread
        if self._reader_thread and self._reader_thread.is_alive():
            self._stop_reader.set()
            self._reader_thread.join(timeout=1.0)

        # Stop motors before shutdown
        if self._serial and self._serial.is_open:
            self.get_logger().info("Stopping motors before shutdown")
            self._send_motor_command(0.0, 0.0)
            time.sleep(0.1)  # Give time for command to send

            # Close serial connection
            try:
                self._serial.close()
            except Exception as e:
                self.get_logger().warn(f"Error closing serial: {e}")

        super().destroy_node()


def main(args=None):
    """Main entry point for the UART motor controller node."""
    rclpy.init(args=args)

    try:
        node = UARTMotorController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in motor controller: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
