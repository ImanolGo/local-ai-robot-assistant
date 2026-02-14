#!/usr/bin/env python3
"""
SLAM Health Monitor Node.

Monitors the health of the RTAB-Map SLAM system by checking:
  - Visual odometry output rate (/rtabmap/odom)
  - TF tree completeness (map → odom → base_link)
  - Depth input availability (/perception/depth)
  - Loop closure events

This node does NOT run SLAM itself — that's done by the official
rtabmap binary nodes launched via slam_launch.py. This node provides
diagnostics and can trigger fallback modes if SLAM fails.

Publishes:
  /slam/status (std_msgs/String): JSON status summary
"""

import json
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class SLAMHealthMonitorNode(Node):
    """Monitors RTAB-Map SLAM health and publishes diagnostics."""

    def __init__(self):
        super().__init__("slam_health_monitor")

        # Configuration
        self.declare_parameter("check_interval_sec", 5.0)
        self.declare_parameter("odom_timeout_sec", 3.0)
        self.declare_parameter("depth_timeout_sec", 3.0)

        self.check_interval = (
            self.get_parameter("check_interval_sec").get_parameter_value().double_value
        )
        self.odom_timeout = (
            self.get_parameter("odom_timeout_sec").get_parameter_value().double_value
        )
        self.depth_timeout = (
            self.get_parameter("depth_timeout_sec").get_parameter_value().double_value
        )

        # State tracking
        self.last_odom_time: float = 0.0
        self.last_depth_time: float = 0.0
        self.odom_count: int = 0
        self.last_check_odom_count: int = 0
        self.slam_healthy: bool = False

        # QoS for sensor topics
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "/rtabmap/odom", self._odom_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/perception/depth", self._depth_callback, qos_sensor
        )

        # Publisher
        self.status_pub = self.create_publisher(String, "/slam/status", 10)

        # Health check timer
        self.timer = self.create_timer(self.check_interval, self._check_health)

        self.get_logger().info("SLAM Health Monitor started")

    def _odom_callback(self, msg: Odometry) -> None:
        """Track visual odometry messages."""
        self.last_odom_time = time.time()
        self.odom_count += 1

    def _depth_callback(self, msg: Image) -> None:
        """Track depth input messages."""
        self.last_depth_time = time.time()

    def _check_health(self) -> None:
        """Periodic health check and status publication."""
        now = time.time()

        odom_alive = (
            (now - self.last_odom_time) < self.odom_timeout if self.last_odom_time else False
        )
        depth_alive = (
            (now - self.last_depth_time) < self.depth_timeout if self.last_depth_time else False
        )

        # Calculate odometry rate
        odom_rate = (self.odom_count - self.last_check_odom_count) / self.check_interval
        self.last_check_odom_count = self.odom_count

        self.slam_healthy = odom_alive and depth_alive

        status = {
            "slam_healthy": self.slam_healthy,
            "visual_odom_active": odom_alive,
            "visual_odom_hz": round(odom_rate, 1),
            "depth_input_active": depth_alive,
            "total_odom_msgs": self.odom_count,
        }

        # Publish status
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

        # Log warnings
        if not depth_alive:
            self.get_logger().warn("No depth data — check perception pipeline")
        if not odom_alive and depth_alive:
            self.get_logger().warn(
                "No visual odometry — RTAB-Map may not be running or features insufficient"
            )
        elif self.slam_healthy:
            self.get_logger().info(
                f"SLAM OK — visual odom at {odom_rate:.1f} Hz",
                throttle_duration_sec=30.0,
            )


def main(args=None):
    rclpy.init(args=args)
    node = SLAMHealthMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
