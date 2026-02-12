#!/usr/bin/env python3
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class EKFTuningTest(Node):
    def __init__(self):
        super().__init__("ekf_tuning_test")

        self.ekf_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.ekf_callback, 10
        )

        self.odom_sub = self.create_subscription(Odometry, "/odom_raw", self.odom_callback, 10)

        self.vo_sub = self.create_subscription(Odometry, "/rtabmap/odom", self.vo_callback, 10)

        self.get_logger().info("EKF Tuning Test Node Started")
        self.get_logger().info("Please drive the robot in a square pattern...")

    def ekf_callback(self, msg):
        self.get_logger().info(
            f"EKF Pose: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}"
        )

    def odom_callback(self, msg):
        # self.get_logger().info(f'Raw Odom: x={msg.pose.pose.position.x:.2f}, \
        # y={msg.pose.pose.position.y:.2f}')
        pass

    def vo_callback(self, msg):
        # self.get_logger().info(f'VO Pose: x={msg.pose.pose.position.x:.2f},\
        #  y={msg.pose.pose.position.y:.2f}')
        pass


def main(args=None):
    rclpy.init(args=args)
    node = EKFTuningTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
