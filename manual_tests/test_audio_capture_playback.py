#!/usr/bin/env python3
"""
Manual test for audio capture and playback nodes.

Tests the audio infrastructure by:
1. Starting audio capture node
2. Monitoring published audio data
3. Testing audio statistics
"""

import rclpy
from rclpy.node import Node

try:
    from audio_common_msgs.msg import AudioData
except ImportError:
    from robot_interfaces.msg import AudioData


class AudioTestNode(Node):
    """Test node for audio infrastructure."""

    def __init__(self):
        super().__init__("audio_test_node")

        self.audio_sub = self.create_subscription(AudioData, "/audio/raw", self.audio_callback, 10)

        self.audio_count = 0
        self.start_time = self.get_clock().now()

        # Create timer for stats
        self.create_timer(5.0, self.print_stats)

        self.get_logger().info("Audio Test Node initialized")
        self.get_logger().info("Listening to /audio/raw...")

    def audio_callback(self, msg: AudioData):
        """Handle received audio data."""
        self.audio_count += 1

        if self.audio_count == 1:
            # Log first message details
            self.get_logger().info("First audio message received!")

            if hasattr(msg, "header") and msg.header:
                self.get_logger().info(f"  Frame ID: {msg.header.frame_id}")

            if hasattr(msg, "sample_rate"):
                self.get_logger().info(f"  Sample rate: {msg.sample_rate} Hz")

            if hasattr(msg, "channels"):
                self.get_logger().info(f"  Channels: {msg.channels}")

            if hasattr(msg, "encoding"):
                self.get_logger().info(f"  Encoding: {msg.encoding}")

            if hasattr(msg, "chunk_size"):
                self.get_logger().info(f"  Chunk size: {msg.chunk_size} samples")

            data_len = len(msg.data)
            self.get_logger().info(f"  Data size: {data_len} bytes")

    def print_stats(self):
        """Print statistics."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if elapsed > 0:
            rate = self.audio_count / elapsed
            self.get_logger().info(
                f"Received {self.audio_count} audio chunks in {elapsed:.1f}s ({rate:.1f} msg/s)"
            )


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = AudioTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
