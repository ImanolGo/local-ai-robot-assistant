#!/usr/bin/env python3
"""
Visual Verification Node — closes the "did I actually reach/see the goal?" loop.

Implements Plan.md Phase 5 item 3 / architecture.md §12 Phase 5 Task 3. After the
robot performs an action for a goal (e.g. navigate to an object), this node:

  1. stops the robot,
  2. asks the cognitive core to verify the goal against a fresh camera frame,
  3. interprets the yes/no answer,
  4. if unsure or negative, rotates by a fixed angle and retries (ensemble),
  5. gives up after a bounded number of attempts.

It is deliberately backend-agnostic: it talks to the cognitive core purely via
the existing ``MultimodalQuery`` / ``MultimodalResponse`` topics, so it works
with both the ``ollama`` and ``llamacpp`` backends.

Subscribes to:
    /cognitive/multimodal_response (MultimodalResponse): Raw VLM answers.
    /cognitive/command (CognitiveCommand): Goals to verify.

Publishes to:
    /cognitive/multimodal_query (MultimodalQuery): Verification prompts.
    /cmd_vel (Twist): Stop / rotate motions.
    /audio/tts_request (String): Spoken status.
    /verification/status (String): Machine-readable state for monitoring.
"""

import math
import re
import uuid
from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.timer import Timer
from std_msgs.msg import String

from robot_interfaces.msg import CognitiveCommand, MultimodalQuery, MultimodalResponse

# Actions that warrant a follow-up verification pass.
VERIFIABLE_ACTIONS = {"navigate", "search", "approach", "pickup"}

# Dedicated system prompt for verification queries. The command-intent system
# prompt asks for JSON, which fights the single-word Yes/No answer this node
# parses; overriding it per-query keeps intent and verification prompts separate.
VERIFICATION_SYSTEM_PROMPT = (
    "You are a robot's visual verification module. Given a camera image and a "
    "yes/no question about whether a goal has been achieved, answer with a "
    "single word: Yes or No. Do not explain, describe, or output JSON."
)

_AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|affirmative|correct|indeed|achieved|succeeded|"
    r"i can see|i see|visible|there is a|it is there|found it)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(no|nope|negative|not|cannot|can't|don't|do not|absent|"
    r"not visible|not there|not found|no longer)\b",
    re.IGNORECASE,
)


class VerificationState(Enum):
    """State machine states for the verification loop."""

    IDLE = "idle"
    VERIFYING = "verifying"
    ROTATING = "rotating"


def build_verification_prompt(goal: str) -> str:
    """Build a strict boolean verification prompt for the VLM.

    Args:
        goal: Target object / goal description (e.g. ``"red ball"``).

    Returns:
        Prompt instructing the model to answer only Yes or No.
    """
    return (
        f"Is the goal '{goal}' achieved in this image? "
        "Is the target clearly visible? Answer with a single word: Yes or No."
    )


def parse_verification_answer(text: Optional[str]) -> Optional[bool]:
    """Interpret a VLM free-text answer as True / False / None (unsure).

    Args:
        text: Raw response text from the cognitive core.

    Returns:
        True if clearly affirmative, False if clearly negative, None if unsure.
    """
    if not text:
        return None

    text = text.strip()
    # Look for an explicit leading yes/no first.
    leading = re.match(r"^[\s\"']*(yes|no)\b", text, re.IGNORECASE)
    if leading:
        return leading.group(1).lower() == "yes"

    if _AFFIRMATIVE.search(text):
        return True
    if _NEGATIVE.search(text):
        return False
    return None


def rotation_duration(angle_deg: float, angular_speed: float) -> float:
    """Seconds needed to rotate ``angle_deg`` at ``angular_speed`` (rad/s).

    Args:
        angle_deg: Rotation angle in degrees.
        angular_speed: Angular velocity magnitude in rad/s.

    Returns:
        Duration in seconds (0 if speed is non-positive).
    """
    if angular_speed <= 0:
        return 0.0
    return abs(math.radians(angle_deg)) / angular_speed


class VisualVerificationNode(Node):
    """ROS2 node implementing the visual verification loop."""

    def __init__(self):
        super().__init__("visual_verification_node")

        # --- Parameters ---
        self.declare_parameter("max_attempts", 3)
        self.declare_parameter("rotation_angle_deg", 45.0)
        self.declare_parameter("rotation_speed", 0.5)
        self.declare_parameter("response_timeout", 8.0)
        self.declare_parameter("trigger_on_cognitive_command", True)
        self.declare_parameter("verification_max_tokens", 16)

        self.max_attempts = self.get_parameter("max_attempts").value
        self.rotation_angle_deg = self.get_parameter("rotation_angle_deg").value
        self.rotation_speed = self.get_parameter("rotation_speed").value
        self.response_timeout = self.get_parameter("response_timeout").value
        self.trigger_on_cognitive_command = self.get_parameter("trigger_on_cognitive_command").value
        self.verification_max_tokens = self.get_parameter("verification_max_tokens").value

        # --- State ---
        self.state = VerificationState.IDLE
        self.goal = ""
        self.attempt = 0
        self._active_query_id = ""
        self._response_timer: Optional[Timer] = None
        self._rotation_timer: Optional[Timer] = None

        # --- Publishers ---
        self.query_pub = self.create_publisher(MultimodalQuery, "/cognitive/multimodal_query", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tts_pub = self.create_publisher(String, "/audio/tts_request", 10)
        self.status_pub = self.create_publisher(String, "/verification/status", 10)

        # --- Subscribers ---
        self.create_subscription(
            MultimodalResponse,
            "/cognitive/multimodal_response",
            self._on_multimodal_response,
            10,
        )
        self.create_subscription(
            CognitiveCommand,
            "/cognitive/command",
            self._on_cognitive_command,
            10,
        )

        self.get_logger().info("✅ Visual verification node initialized")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_cognitive_command(self, msg: CognitiveCommand) -> None:
        """Start verification when a verifiable goal is issued."""
        if not self.trigger_on_cognitive_command:
            return
        action = msg.action.lower()
        target = msg.target_object.strip()
        if action in VERIFIABLE_ACTIONS and target and self.state == VerificationState.IDLE:
            self.start_verification(target)

    def _on_multimodal_response(self, msg: MultimodalResponse) -> None:
        """Handle a verification answer from the cognitive core."""
        if self.state != VerificationState.VERIFYING:
            return
        if msg.query_id and msg.query_id != self._active_query_id:
            return
        if msg.has_error:
            self.get_logger().warn(f"Verification query errored: {msg.error_message}")
            self._retry_or_give_up()
            return

        self._cancel_response_timer()
        answer = parse_verification_answer(msg.response_text)
        self.get_logger().info(
            f"Verification answer (attempt {self.attempt + 1}): "
            f"{answer} from '{msg.response_text.strip()[:80]}'"
        )

        if answer is True:
            self._succeed()
        else:
            self._retry_or_give_up()

    def _on_response_timeout(self) -> None:
        """No answer arrived in time — treat as unsure and retry."""
        self._cancel_response_timer()
        self.get_logger().warn("Verification response timed out.")
        self._retry_or_give_up()

    # ------------------------------------------------------------------
    # Verification loop
    # ------------------------------------------------------------------

    def start_verification(self, goal: str) -> None:
        """Begin a verification cycle for ``goal``.

        Args:
            goal: Target object / goal description.
        """
        self.goal = goal
        self.attempt = 0
        self.get_logger().info(f"Starting visual verification for goal: '{goal}'")
        self._send_verification_query()

    def _send_verification_query(self) -> None:
        self._stop_robot()
        self.state = VerificationState.VERIFYING
        self.attempt += 1
        self._publish_status()

        query = MultimodalQuery()
        query.header.stamp = self.get_clock().now().to_msg()
        query.query_id = str(uuid.uuid4())
        self._active_query_id = query.query_id
        query.text_query = build_verification_prompt(self.goal)
        query.system_prompt = VERIFICATION_SYSTEM_PROMPT
        query.include_current_image = True
        query.temperature = 0.0
        query.max_tokens = self.verification_max_tokens
        self.query_pub.publish(query)

        self._cancel_response_timer()
        self._response_timer = self.create_timer(
            self.response_timeout, self._on_response_timeout, callback_group=None
        )

    def _retry_or_give_up(self) -> None:
        if self.attempt >= self.max_attempts:
            self.get_logger().warn(
                f"Verification failed after {self.attempt} attempts for '{self.goal}'"
            )
            self._publish_tts(f"I could not confirm the {self.goal} is there.")
            self._stop_robot()
            self.state = VerificationState.IDLE
            self._publish_status()
            return

        angle = self.rotation_angle_deg if self.attempt % 2 == 0 else -self.rotation_angle_deg
        self._rotate(angle)

    def _rotate(self, angle_deg: float) -> None:
        """Rotate in place by ``angle_deg`` then re-send the verification query."""
        self.state = VerificationState.ROTATING
        self._publish_status()

        speed = self.rotation_speed if angle_deg >= 0 else -self.rotation_speed
        twist = Twist()
        twist.angular.z = speed
        self.cmd_vel_pub.publish(twist)

        duration = rotation_duration(angle_deg, self.rotation_speed)
        self._cancel_rotation_timer()
        self._rotation_timer = self.create_timer(
            max(duration, 0.1), self._finish_rotation, callback_group=None
        )

    def _finish_rotation(self) -> None:
        self._cancel_rotation_timer()
        self._stop_robot()
        self._send_verification_query()

    def _succeed(self) -> None:
        self.get_logger().info(f"Goal '{self.goal}' verified achieved.")
        self._publish_tts(f"I found the {self.goal}.")
        self._stop_robot()
        self.state = VerificationState.IDLE
        self._publish_status()

    def _stop_robot(self) -> None:
        self.cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cancel_response_timer(self) -> None:
        if self._response_timer is not None:
            self._response_timer.cancel()
            self._response_timer = None

    def _cancel_rotation_timer(self) -> None:
        if self._rotation_timer is not None:
            self._rotation_timer.cancel()
            self._rotation_timer = None

    def _publish_tts(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)

    def _publish_status(self) -> None:
        msg = String()
        msg.data = f"{self.state.value} | goal='{self.goal}' | attempt={self.attempt}"
        self.status_pub.publish(msg)

    def destroy_node(self) -> None:
        try:
            self._cancel_response_timer()
            self._cancel_rotation_timer()
            self._stop_robot()
        except Exception:  # context may already be shut down
            pass
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    try:
        node = VisualVerificationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in visual verification node: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
