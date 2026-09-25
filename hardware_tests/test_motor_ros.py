#!/usr/bin/env python3
"""
ROS-level motor check for the Wave Rover (run manually, on hardware).

Drives the base at low speed via ``/cmd_vel``, reads back the chassis feedback
and dead-reckoning odometry, and exercises the ``/emergency_stop`` service.
Always stops the motors on exit.

Usage (with the workspace sourced and ``uart_motor_controller`` running):
    source ros2_venv.sh
    python hardware_tests/test_motor_ros.py
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from robot_interfaces.msg import ChassisState
from robot_interfaces.srv import EmergencyStop

FORWARD_SPEED = 0.15  # m/s
TURN_RATE = 0.5  # rad/s
MOVE_SECONDS = 1.5


class MotorCheck(Node):
    """Minimal driver that samples chassis state and odometry while moving."""

    def __init__(self) -> None:
        super().__init__("motor_check")
        self.chassis = None
        self.odom = None
        self.gyro_z_samples = []
        self.collect_gyro = False
        self.create_subscription(
            ChassisState, "/chassis_state", self._on_chassis, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, "/odom_raw", self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Imu, "/imu/data", self._on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.cli = self.create_client(EmergencyStop, "/emergency_stop")

    def _on_chassis(self, msg: ChassisState) -> None:
        self.chassis = msg

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def _on_imu(self, msg: Imu) -> None:
        # angular_velocity_covariance[0] == -1 means "gyro not available"
        # (messages derived from T=1001 carry no gyro); only keep valid samples.
        if self.collect_gyro and msg.angular_velocity_covariance[0] != -1.0:
            self.gyro_z_samples.append(msg.angular_velocity.z)

    def spin_for(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def measure_gyro_z(self, drive_angular: float, seconds: float):
        """Return (mean gyro-Z, sample count) while turning (or idling if 0)."""
        self.gyro_z_samples = []
        self.collect_gyro = True
        if drive_angular == 0.0:
            self.spin_for(seconds)
        else:
            self.drive(0.0, drive_angular, seconds)
        self.collect_gyro = False
        samples = list(self.gyro_z_samples)
        return (sum(samples) / len(samples) if samples else 0.0), len(samples)

    def drive(self, linear: float, angular: float, seconds: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        end = time.time() + seconds
        while time.time() < end:
            self.pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.01)

    def stop(self) -> None:
        for _ in range(10):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.01)
        self.spin_for(1.0)

    def describe(self) -> str:
        c = self.chassis
        o = self.odom
        base = (
            "no chassis state"
            if c is None
            else (
                f"L={c.left_motor_current:+.3f} R={c.right_motor_current:+.3f} "
                f"yaw={c.yaw:7.2f} estop={c.emergency_stop_active} "
                f"motors={c.motors_enabled} v={c.battery_voltage:.2f}"
            )
        )
        if o is not None:
            p = o.pose.pose.position
            base += f" | odom=({p.x:+.3f},{p.y:+.3f})"
        return base

    def call_estop(self, enable: bool):
        if not self.cli.wait_for_service(timeout_sec=3.0):
            return None
        request = EmergencyStop.Request()
        request.enable_stop = bool(enable)
        request.reason = "test_motor_ros"
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result()


def main() -> int:
    rclpy.init()
    node = MotorCheck()
    try:
        node.spin_for(2.0)
        print("baseline       :", node.describe())

        print(f"-> forward {FORWARD_SPEED} m/s for {MOVE_SECONDS}s")
        node.drive(FORWARD_SPEED, 0.0, MOVE_SECONDS)
        print("during forward :", node.describe())
        node.stop()
        print("after stop     :", node.describe())

        print(f"-> turn {TURN_RATE} rad/s for {MOVE_SECONDS}s")
        baseline_gz, base_n = node.measure_gyro_z(0.0, 3.0)
        turn_gz, turn_n = node.measure_gyro_z(TURN_RATE, 3.0)
        print("during turn    :", node.describe())
        print(
            "   gyro-Z baseline=%.3f rad/s turn=%.3f rad/s (bias-corrected %.3f, n=%d/%d)"
            % (baseline_gz, turn_gz, turn_gz - baseline_gz, base_n, turn_n)
        )
        node.stop()
        print("after stop     :", node.describe())

        print("-> emergency stop ON/OFF")
        result = node.call_estop(True)
        print("   service:", None if result is None else (result.success, result.message))
        node.spin_for(1.0)
        print("after estop    :", node.describe())
        result = node.call_estop(False)
        print("   service:", None if result is None else (result.success, result.message))
        node.spin_for(1.0)
        print("after release  :", node.describe())
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
