#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


class SLAMNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.get_logger().info("SLAM Node (Stub) Started")


def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
