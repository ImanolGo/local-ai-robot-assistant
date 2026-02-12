#!/usr/bin/env python3
"""
Command Router Node — Routes voice commands to the appropriate handler.

This node sits between the audio pipeline and the cognitive core / actuation layer.
It classifies incoming transcriptions as either:
  1. **Simple commands** — handled directly (stop, go forward, turn, etc.)
  2. **Complex commands** — forwarded to the cognitive client for VLM reasoning

Architecture Reference: docs/architecture.md §2.6 (Behavioral Architecture)

Subscribes to:
    /audio/transcription (TranscriptionResult): Transcribed voice commands.
    /cognitive/command (CognitiveCommand): Parsed intents from the cognitive core.

Publishes to:
    /cmd_vel (Twist): Direct motor commands for simple actions.
    /cognitive/multimodal_query (MultimodalQuery): Complex queries for VLM.
    /audio/tts_request (String): Verbal acknowledgements.

Author: Local AI Robot Assistant Team
License: See LICENSE file in project root
"""

import re
import uuid
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import CognitiveCommand, MultimodalQuery, TranscriptionResult

# Simple command patterns — matched with regex before invoking the VLM.
# These provide sub-100ms response for safety-critical and trivial commands.
SIMPLE_COMMANDS = {
    # Emergency / safety — highest priority
    r"\b(stop|halt|freeze|emergency)\b": ("stop", "Stopping."),
    # Basic motion
    r"\b(go\s+forward|move\s+forward|advance)\b": ("forward", "Moving forward."),
    r"\b(go\s+back|move\s+back|reverse|back\s+up)\b": ("backward", "Moving backward."),
    r"\b(turn\s+left|go\s+left)\b": ("turn_left", "Turning left."),
    r"\b(turn\s+right|go\s+right)\b": ("turn_right", "Turning right."),
    r"\b(turn\s+around|do\s+a?\s*180)\b": ("turn_around", "Turning around."),
    r"\b(come\s+here|come\s+to\s+me)\b": ("forward", "Coming to you."),
    # Status
    r"\b(what\s+do\s+you\s+see|describe|look\s+around)\b": ("describe", None),
    # Home
    r"\b(go\s+home|return\s+home|return\s+to\s+base)\b": (
        "return_home",
        "Returning home.",
    ),
}


class CommandRouterNode(Node):
    """Routes transcribed voice commands to the correct handler.

    Simple commands are executed directly as Twist messages. Complex commands
    that require visual reasoning are forwarded to the cognitive client node
    for Ollama/Moondream processing.
    """

    def __init__(self):
        super().__init__("command_router_node")

        # --- Parameters ---
        self.declare_parameter("linear_speed", 0.15)  # m/s for simple forward/back
        self.declare_parameter("angular_speed", 0.5)  # rad/s for simple turns
        self.declare_parameter("command_duration", 1.0)  # seconds for timed commands
        self.declare_parameter("confidence_threshold", 0.3)  # minimum ASR confidence

        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value
        self.command_duration = self.get_parameter("command_duration").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value

        # Compile regex patterns once
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), action, response)
            for pattern, (action, response) in SIMPLE_COMMANDS.items()
        ]

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.query_pub = self.create_publisher(MultimodalQuery, "/cognitive/multimodal_query", 10)
        self.tts_pub = self.create_publisher(String, "/audio/tts_request", 10)

        # --- Subscribers ---
        self.create_subscription(
            TranscriptionResult,
            "/audio/transcription",
            self._on_transcription,
            10,
        )
        self.create_subscription(
            CognitiveCommand,
            "/cognitive/command",
            self._on_cognitive_command,
            10,
        )

        # --- Timed command stop timer ---
        self._stop_timer = None

        self.get_logger().info("✅ Command router initialized")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_transcription(self, msg: TranscriptionResult) -> None:
        """Handle incoming voice transcription.

        Args:
            msg: Transcription result from the audio pipeline.
        """
        text = msg.text.strip()
        if not text:
            return

        if msg.confidence < self.confidence_threshold:
            self.get_logger().info(
                f"Ignoring low-confidence transcription: '{text}' "
                f"(conf={msg.confidence:.2f} < {self.confidence_threshold})"
            )
            return

        self.get_logger().info(f"Received command: '{text}' (conf={msg.confidence:.2f})")

        # Try to match a simple command first (architecture §2.6 step 1)
        result = self._match_simple_command(text)
        if result:
            action, response = result
            self.get_logger().info(f"Simple command matched: {action}")
            self._execute_simple_command(action)
            if response:
                tts_msg = String()
                tts_msg.data = response
                self.tts_pub.publish(tts_msg)
            return

        # Complex command — forward to cognitive core (architecture §2.6 step 2)
        self.get_logger().info("Complex command — forwarding to cognitive core.")
        self._forward_to_cognitive(text)

    def _on_cognitive_command(self, msg: CognitiveCommand) -> None:
        """Handle parsed commands from the cognitive core.

        This allows the behavior tree / cognitive core to issue motor commands
        through the same routing infrastructure.

        Args:
            msg: Cognitive command with action and target.
        """
        action = msg.action.lower()
        self.get_logger().info(
            f"Cognitive command received: action={action}, target={msg.target_object}"
        )

        if action == "stop":
            self._execute_simple_command("stop")
        elif action == "navigate":
            # Navigation to a target — would integrate with Nav2 / SLAM in Phase 6+
            self.get_logger().info(f"Navigate to '{msg.target_object}' — forwarding to planner")
            # For now, acknowledge and move forward slowly
            self._execute_simple_command("forward")
        elif action == "speak":
            tts_msg = String()
            tts_msg.data = msg.response_text
            self.tts_pub.publish(tts_msg)

    # ------------------------------------------------------------------
    # Simple command execution
    # ------------------------------------------------------------------

    def _match_simple_command(self, text: str) -> Optional[Tuple[str, Optional[str]]]:
        """Match text against simple command patterns.

        Args:
            text: Transcribed voice text.

        Returns:
            Tuple of (action_name, verbal_response) or None if no match.
        """
        text_lower = text.lower()
        for pattern, action, response in self._compiled_patterns:
            if pattern.search(text_lower):
                return (action, response)
        return None

    def _execute_simple_command(self, action: str) -> None:
        """Execute a simple motor command.

        Args:
            action: Action name from SIMPLE_COMMANDS.
        """
        twist = Twist()

        if action == "stop":
            # Immediate stop — zero velocity
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            # Cancel any pending timed stop
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None
            return

        if action == "forward":
            twist.linear.x = self.linear_speed
        elif action == "backward":
            twist.linear.x = -self.linear_speed
        elif action == "turn_left":
            twist.angular.z = self.angular_speed
        elif action == "turn_right":
            twist.angular.z = -self.angular_speed
        elif action == "turn_around":
            twist.angular.z = self.angular_speed * 2.0  # Faster rotation
        elif action == "return_home":
            # Placeholder — would use Nav2 in future
            twist.linear.x = 0.0
            self.get_logger().info("Return home not yet implemented — stopping.")
        elif action == "describe":
            # Forward a describe request to cognitive core with vision
            self._forward_to_cognitive("Describe what you see in front of you.")
            return

        self.cmd_vel_pub.publish(twist)

        # Auto-stop after duration (safety)
        if self._stop_timer is not None:
            self._stop_timer.cancel()
        self._stop_timer = self.create_timer(
            self.command_duration, self._timed_stop, callback_group=None
        )

    def _timed_stop(self) -> None:
        """Stop motors after the command duration expires."""
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    # ------------------------------------------------------------------
    # Complex command forwarding
    # ------------------------------------------------------------------

    def _forward_to_cognitive(self, text: str) -> None:
        """Forward a complex command to the cognitive client node via MultimodalQuery.

        Args:
            text: The transcribed or constructed text query.
        """
        query = MultimodalQuery()
        query.header.stamp = self.get_clock().now().to_msg()
        query.query_id = str(uuid.uuid4())
        query.text_query = text
        query.include_current_image = True  # Request vision for complex commands
        query.temperature = 0.3
        query.max_tokens = 128

        self.query_pub.publish(query)
        self.get_logger().info(f"Forwarded query to cognitive core: '{text[:60]}...'")

        # Acknowledge to user
        tts_msg = String()
        tts_msg.data = "Let me think about that."
        self.tts_pub.publish(tts_msg)

    def destroy_node(self) -> None:
        """Cleanup — ensure motors stop."""
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        self.get_logger().info("Command router shutting down — motors stopped.")
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = CommandRouterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in command router: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
