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
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rclpy
import serial

# ROS2 messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
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

        # Thread safety
        self._serial_lock = threading.Lock()
        self._state_lock = threading.Lock()

        # Initialize serial connection
        self._serial: Optional[serial.Serial] = None
        self._connect_serial()

        # Set up QoS profiles
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Publishers
        self.motor_status_pub = self.create_publisher(ChassisState, "/motor_status", qos_reliable)

        self.chassis_state_pub = self.create_publisher(ChassisState, "/chassis_state", qos_reliable)

        self.odom_raw_pub = self.create_publisher(Odometry, "/odom_raw", qos_sensor)

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

        # Enable continuous feedback mode
        if self._serial:
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
                self.get_logger().debug(f"Sent command: {command}")
                return True

        except Exception as e:
            self.get_logger().error(f"UART send error: {e}")
            return False

    def _enable_continuous_feedback(self) -> bool:
        """Enable continuous feedback mode on Wave Rover."""
        command = {"T": 131, "cmd": 1}
        success = self._send_command(command)
        if success:
            self.get_logger().info("Continuous feedback mode enabled")
        else:
            self.get_logger().warn("Failed to enable continuous feedback mode")
        return success

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
        """Handle cmd_vel messages."""
        with self._state_lock:
            self._last_cmd_vel = msg
            self._last_command_time = time.time()

    def _motor_command_callback(self, msg: MotorCommand):
        """Handle direct motor command messages."""
        if msg.enable and not self._emergency_stop_active:
            success = self._send_motor_command(msg.left_speed, msg.right_speed)
            if not success:
                self.get_logger().warn("Failed to send motor command")

        with self._state_lock:
            self._last_command_time = time.time()

    def _command_timer_callback(self):
        """Periodic command sending based on last received cmd_vel."""
        if self._emergency_stop_active:
            return

        with self._state_lock:
            cmd_vel = self._last_cmd_vel

        # Convert twist to wheel speeds
        left_speed, right_speed = self._twist_to_wheel_speeds(cmd_vel)

        # Send motor command
        self._send_motor_command(left_speed, right_speed)

    def _watchdog_callback(self):
        """Watchdog timer to stop motors if no commands received."""
        current_time = time.time()

        with self._state_lock:
            time_since_command = current_time - self._last_command_time

        if time_since_command > self.motor_config.watchdog_timeout:
            # Stop motors due to timeout
            if any(speed != 0.0 for speed in self._current_wheel_speeds.values()):
                self.get_logger().warn(
                    f"Watchdog timeout ({time_since_command:.2f}s), stopping motors"
                )
                self._send_motor_command(0.0, 0.0)

    def _status_timer_callback(self):
        """Publish motor status and chassis state."""
        # Create chassis state message
        chassis_msg = ChassisState()
        chassis_msg.header = Header()
        chassis_msg.header.stamp = self.get_clock().now().to_msg()
        chassis_msg.header.frame_id = "base_link"

        with self._state_lock:
            chassis_msg.motors_enabled = self._motors_enabled and not self._emergency_stop_active
            chassis_msg.left_motor_current = 0.0  # TODO: Parse from feedback
            chassis_msg.right_motor_current = 0.0  # TODO: Parse from feedback

        # Publish chassis state
        self.chassis_state_pub.publish(chassis_msg)
        self.motor_status_pub.publish(chassis_msg)

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
        # Stop motors before shutdown
        if self._serial and self._serial.is_open:
            self.get_logger().info("Stopping motors before shutdown")
            self._send_motor_command(0.0, 0.0)
            time.sleep(0.1)  # Give time for command to send

            # Close serial connection
            self._serial.close()

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
