#!/usr/bin/env python3
"""Synthetic workload generator for the full-system integration soak.

Publishes periodic lightweight messages that exercise the command router, the
cognitive core, and the visual verification loop without requiring a person to
speak. No actuation is involved: the soak launch omits the motor controller, so
the ``/cmd_vel`` messages these stimuli provoke are never consumed.

Usage (with the workspace sourced):
    source ros2_venv.sh
    python scripts/testing/integration/soak_workload.py
"""

import uuid

import rclpy
from rclpy.node import Node

from robot_interfaces.msg import CognitiveCommand, MultimodalQuery, TranscriptionResult


class SoakWorkload(Node):
    """Periodically stimulates the cognitive / behavioral pipeline."""

    def __init__(self) -> None:
        super().__init__("soak_workload")

        self.query_pub = self.create_publisher(MultimodalQuery, "/cognitive/multimodal_query", 10)
        self.transcription_pub = self.create_publisher(
            TranscriptionResult, "/audio/transcription", 10
        )
        self.command_pub = self.create_publisher(CognitiveCommand, "/cognitive/command", 10)

        # Staggered so the single-threaded cognitive core is never saturated.
        self.create_timer(20.0, self._vision_query)
        self.create_timer(45.0, self._complex_transcription)
        self.create_timer(70.0, self._simple_transcription)
        self.create_timer(100.0, self._verification_trigger)

        self.get_logger().info("Soak workload generator started")

    def _vision_query(self) -> None:
        """Direct multimodal (vision) query to the cognitive core."""
        query = MultimodalQuery()
        query.header.stamp = self.get_clock().now().to_msg()
        query.query_id = str(uuid.uuid4())
        query.text_query = "What objects are visible in this image?"
        query.include_current_image = True
        query.max_tokens = 64
        query.use_optimizations = True
        self.query_pub.publish(query)
        self.get_logger().info("workload: vision MultimodalQuery sent")

    def _complex_transcription(self) -> None:
        """Transcription that the command router forwards to the cognitive core."""
        msg = TranscriptionResult()
        msg.text = "describe what you see"
        msg.confidence = 0.9
        self.transcription_pub.publish(msg)
        self.get_logger().info("workload: complex transcription sent")

    def _simple_transcription(self) -> None:
        """Simple command handled directly by the command router."""
        msg = TranscriptionResult()
        msg.text = "stop"
        msg.confidence = 0.9
        self.transcription_pub.publish(msg)
        self.get_logger().info("workload: simple 'stop' transcription sent")

    def _verification_trigger(self) -> None:
        """Goal command that starts the visual verification loop."""
        cmd = CognitiveCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.action = "navigate"
        cmd.target_object = "red ball"
        self.command_pub.publish(cmd)
        self.get_logger().info("workload: verification CognitiveCommand sent")


def main(args=None) -> None:
    """Main entry point."""
    rclpy.init(args=args)
    node = SoakWorkload()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
