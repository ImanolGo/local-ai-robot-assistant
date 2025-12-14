#!/usr/bin/env python3
"""
Manual test script for wake word detection node.

This script tests the wake word detector with real audio input.
Run this after starting the audio_capture_node.

Usage:
    Terminal 1: ./launch_node.sh audio_interface_nodes audio_capture_node
    Terminal 2: ./launch_node.sh audio_interface_nodes wake_word_detector_node
    Terminal 3: python manual_tests/test_wake_word_live.py
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from robot_interfaces.msg import AudioEvent


class WakeWordTestNode(Node):
    """Test node to monitor wake word detection."""

    def __init__(self):
        super().__init__("wake_word_test_node")

        self.detection_count = 0
        self.last_confidence = 0.0
        self.start_time = time.time()

        # Subscribe to wake word detection
        self.detection_sub = self.create_subscription(
            Bool, "/audio/wake_word_detected", self.detection_callback, 10
        )

        # Subscribe to confidence scores
        self.confidence_sub = self.create_subscription(
            Float32, "/audio/wake_word_confidence", self.confidence_callback, 10
        )

        # Subscribe to audio events
        self.event_sub = self.create_subscription(
            AudioEvent, "/audio/events", self.event_callback, 10
        )

        # Create status timer
        self.timer = self.create_timer(5.0, self.print_status)

        self.get_logger().info("Wake word test node started. " "Say 'Hey Rover' to test detection.")
        self.get_logger().info(
            "Available wake words: 'hey rover', 'alexa', 'hey mycroft', 'hey rhasspy'"
        )

    def detection_callback(self, msg: Bool):
        """Handle wake word detection."""
        if msg.data:
            self.detection_count += 1
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f"🎤 WAKE WORD DETECTED! "
                f"(#{self.detection_count}, confidence: {self.last_confidence:.3f}, "
                f"time: {elapsed:.1f}s)"
            )

    def confidence_callback(self, msg: Float32):
        """Handle confidence score updates."""
        self.last_confidence = msg.data
        if msg.data > 0.3:  # Log significant confidence values
            self.get_logger().debug(f"Confidence: {msg.data:.3f}")

    def event_callback(self, msg: AudioEvent):
        """Handle audio events."""
        if msg.event_type == "wake_word_detected":
            self.get_logger().info(
                f"📢 Audio Event: {msg.event_type} - {msg.data} "
                f"(confidence: {msg.confidence:.3f})"
            )

    def print_status(self):
        """Print periodic status."""
        elapsed = time.time() - self.start_time
        rate = self.detection_count / (elapsed / 60.0) if elapsed > 0 else 0
        self.get_logger().info(
            f"Status: {self.detection_count} detections in {elapsed:.0f}s "
            f"({rate:.2f} detections/minute)"
        )


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    print("\n" + "=" * 60)
    print("Wake Word Detection Live Test")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Make sure audio_capture_node is running")
    print("2. Make sure wake_word_detector_node is running")
    print("3. Say 'Hey Rover' to test detection")
    print("4. Press Ctrl+C to stop\n")
    print("Default wake words available:")
    print("  - 'hey rover'")
    print("  - 'alexa'")
    print("  - 'hey mycroft'")
    print("  - 'hey rhasspy'")
    print("=" * 60 + "\n")

    try:
        node = WakeWordTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
