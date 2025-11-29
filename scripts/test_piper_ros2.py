#!/usr/bin/env python3
"""
Test script for Piper TTS ROS2 Node.

This script tests the ROS2 integration of Piper TTS by publishing text messages
and verifying that audio is generated correctly.
"""

import sys
import time

import rclpy
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from std_msgs.msg import String


class PiperTTSTestNode(Node):
    """Test node for Piper TTS ROS2 integration."""

    def __init__(self):
        super().__init__("piper_tts_test_node")

        # Create publisher for text messages
        self.text_publisher = self.create_publisher(String, "text_to_synthesize", 10)

        # Create subscriber for audio output
        self.audio_subscriber = self.create_subscription(
            AudioData, "synthesized_audio", self.audio_callback, 10
        )

        self.audio_received = False
        self.audio_data_size = 0

        self.get_logger().info("Piper TTS Test Node initialized")

    def audio_callback(self, msg: AudioData):
        """Handle received audio data."""
        self.audio_received = True
        self.audio_data_size = len(msg.data) if hasattr(msg, "data") else 0
        self.get_logger().info(f"Received audio data: {self.audio_data_size} bytes")

    def publish_test_text(self, text: str):
        """Publish text for synthesis."""
        msg = String()
        msg.data = text
        self.text_publisher.publish(msg)
        self.get_logger().info(f'Published text: "{text}"')

    def run_test(self):
        """Run the integration test."""
        self.get_logger().info("Starting Piper TTS ROS2 integration test")

        # Wait a bit for connections
        time.sleep(2.0)

        # Test phrases
        test_phrases = [
            "Hello ROS2!",
            "Testing Piper TTS integration.",
            "This is a longer test sentence to verify proper audio synthesis.",
        ]

        for i, phrase in enumerate(test_phrases):
            self.get_logger().info(f"Test {i+1}/{len(test_phrases)}")

            self.audio_received = False
            self.publish_test_text(phrase)

            # Wait for audio response
            timeout = 10.0
            start_time = time.time()

            while not self.audio_received and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.audio_received:
                self.get_logger().info(f"✓ PASS: Audio generated ({self.audio_data_size} bytes)")
            else:
                self.get_logger().error("✗ FAIL: No audio received within timeout")

            time.sleep(1.0)  # Brief pause between tests

        self.get_logger().info("Piper TTS ROS2 integration test completed")


def main(args=None):
    """Main function to run the test node."""
    rclpy.init(args=args)

    try:
        test_node = PiperTTSTestNode()

        # Run the test
        test_node.run_test()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
