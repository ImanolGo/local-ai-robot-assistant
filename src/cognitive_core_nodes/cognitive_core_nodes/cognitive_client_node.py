#!/usr/bin/env python3
"""
Cognitive Client Node — Ollama/Moondream Bridge for Local AI Robot Assistant.

This node bridges ROS2 with the local Ollama server running Moondream (1.6B VLM).
It receives transcription results and optionally camera snapshots, constructs prompts,
sends HTTP requests to the Ollama API, parses structured JSON intents, and publishes
CognitiveCommand messages to the behavior tree.

Architecture Reference: docs/architecture.md §2.5 (Tier 2 — Strategic Cognitive Core)

Features:
    - HTTP client to local Ollama server (localhost:11434)
    - Vision + text multimodal queries via Moondream
    - Structured JSON intent parsing from VLM responses
    - Connection keep-alive for minimal HTTP overhead
    - Graceful fallback on Ollama timeout / crash
    - Camera snapshot capture on demand

Author: Local AI Robot Assistant Team
License: See LICENSE file in project root
"""

import base64
import json
import re
import time
import uuid
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from robot_interfaces.msg import (
    CognitiveCommand,
    MultimodalQuery,
    MultimodalResponse,
    TranscriptionResult,
)

# Default system prompt for Moondream — kept short and direct per architecture §7.2
SYSTEM_PROMPT = (
    "You are a helpful robot assistant. When given a command and an image, "
    "respond ONLY with a valid JSON object. Use this exact format:\n"
    '{"action": "<action>", "target": "<object>", "explanation": "<brief reason>"}\n'
    "Valid actions: navigate, search, follow, stop, return_home, speak, listen, explore, pickup.\n"
    "If no image is provided, respond with a helpful text answer."
)


class OllamaBridge:
    """HTTP client for the local Ollama server.

    Maintains a persistent session for keep-alive connections to reduce latency.

    Args:
        base_url: Ollama API endpoint.
        model: Model name to use (e.g. 'moondream').
        timeout: Request timeout in seconds.
        logger: ROS2 logger instance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "moondream",
        timeout: float = 10.0,
        logger=None,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.logger = logger

        # Persistent session for TCP keep-alive (architecture §15.2)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def is_available(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        num_ctx: int = 512,
        num_predict: int = 128,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Send a generate request to Ollama.

        Args:
            prompt: Text prompt to send.
            image_base64: Optional base64-encoded image for vision queries.
            num_ctx: Context window size.
            num_predict: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict with 'response' text and 'total_duration' in nanoseconds,
            or 'error' key on failure.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": temperature,
            },
            "keep_alive": -1,  # Keep model loaded indefinitely (architecture §7.1)
        }

        if image_base64:
            payload["images"] = [image_base64]

        try:
            resp = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        except requests.Timeout:
            msg = f"Ollama request timed out after {self.timeout}s"
            if self.logger:
                self.logger.error(msg)
            return {"error": msg}

        except requests.ConnectionError:
            msg = "Cannot connect to Ollama server — is it running?"
            if self.logger:
                self.logger.error(msg)
            return {"error": msg}

        except requests.RequestException as e:
            msg = f"Ollama request failed: {e}"
            if self.logger:
                self.logger.error(msg)
            return {"error": msg}


def parse_json_intent(response_text: str) -> Optional[Dict[str, str]]:
    """Parse a structured JSON intent from the VLM response.

    Handles noisy output from small models — strips markdown code fences,
    extracts the first JSON object, and validates required keys.

    Args:
        response_text: Raw response text from Moondream.

    Returns:
        Parsed dict with 'action', 'target', 'explanation' keys, or None.
    """
    if not response_text:
        return None

    # Strip markdown code fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text)
    cleaned = cleaned.strip()

    # Try to extract first JSON object
    match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        intent = json.loads(match.group())
        # Validate required keys
        if "action" in intent:
            return {
                "action": str(intent.get("action", "speak")),
                "target": str(intent.get("target", "")),
                "explanation": str(intent.get("explanation", "")),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    return None


class CognitiveClientNode(Node):
    """ROS2 node bridging robot state with the local Ollama VLM server.

    Subscribes to:
        /audio/transcription (TranscriptionResult): Transcribed voice commands.
        /camera/undistorted (Image): Camera feed for vision snapshots.
        /cognitive/multimodal_query (MultimodalQuery): Direct query requests.

    Publishes to:
        /cognitive/command (CognitiveCommand): Parsed action intents.
        /cognitive/multimodal_response (MultimodalResponse): Raw VLM responses.
        /audio/tts_request (String): Verbal responses to user.
        /cognitive/status (String): Node health status.
    """

    def __init__(self):
        super().__init__("cognitive_client_node")

        # --- Parameters ---
        self.declare_parameter("ollama_url", "http://localhost:11434")
        self.declare_parameter("model_name", "moondream")
        self.declare_parameter("request_timeout", 10.0)
        self.declare_parameter("num_ctx", 512)
        self.declare_parameter("num_predict", 128)
        self.declare_parameter("temperature", 0.3)
        self.declare_parameter("system_prompt", SYSTEM_PROMPT)
        self.declare_parameter("enable_vision", True)
        self.declare_parameter("health_check_interval", 30.0)

        # Read parameters
        ollama_url = self.get_parameter("ollama_url").value
        model_name = self.get_parameter("model_name").value
        timeout = self.get_parameter("request_timeout").value
        self.num_ctx = self.get_parameter("num_ctx").value
        self.num_predict = self.get_parameter("num_predict").value
        self.temperature = self.get_parameter("temperature").value
        self.system_prompt = self.get_parameter("system_prompt").value
        self.enable_vision = self.get_parameter("enable_vision").value
        health_interval = self.get_parameter("health_check_interval").value

        # --- Ollama bridge ---
        self.ollama = OllamaBridge(
            base_url=ollama_url,
            model=model_name,
            timeout=timeout,
            logger=self.get_logger(),
        )

        # --- CV bridge for image conversion ---
        self.cv_bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_stamp = None

        # --- Performance tracking ---
        self.inference_times: list[float] = []
        self.query_count = 0

        # --- Publishers ---
        self.command_pub = self.create_publisher(CognitiveCommand, "/cognitive/command", 10)
        self.response_pub = self.create_publisher(
            MultimodalResponse, "/cognitive/multimodal_response", 10
        )
        self.tts_pub = self.create_publisher(String, "/audio/tts_request", 10)
        self.status_pub = self.create_publisher(String, "/cognitive/status", 10)

        # --- Subscribers ---
        self.create_subscription(
            TranscriptionResult,
            "/audio/transcription",
            self._on_transcription,
            10,
        )
        self.create_subscription(
            Image,
            "/camera/undistorted",
            self._on_image,
            10,
        )
        self.create_subscription(
            MultimodalQuery,
            "/cognitive/multimodal_query",
            self._on_multimodal_query,
            10,
        )

        # --- Health check timer ---
        self.create_timer(health_interval, self._health_check)

        # --- Startup check ---
        if self.ollama.is_available():
            self.get_logger().info(
                f"✅ Cognitive client initialized — Ollama '{model_name}' at {ollama_url}"
            )
        else:
            self.get_logger().warn(
                f"⚠️  Ollama server not reachable at {ollama_url}. "
                "Node will retry on incoming queries."
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        """Cache the latest camera frame for on-demand vision queries."""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def _on_transcription(self, msg: TranscriptionResult) -> None:
        """Handle an incoming voice transcription.

        Simple commands (stop, go, etc.) could be routed directly by the
        command_router_node. This callback handles complex commands that
        need VLM reasoning.

        Args:
            msg: Transcription result with text and confidence.
        """
        text = msg.text.strip()
        if not text:
            return

        self.get_logger().info(f"Received transcription: '{text}' (conf={msg.confidence:.2f})")

        # Build prompt
        prompt = f'{self.system_prompt}\n\nUser said: "{text}"\n'

        # Optionally include vision
        image_b64 = None
        if self.enable_vision and self.latest_image is not None:
            image_b64 = self._encode_image(self.latest_image)
            prompt += "An image of the robot's current view is attached."

        # Query Ollama
        self._query_ollama(
            prompt=prompt,
            image_base64=image_b64,
            query_id=str(uuid.uuid4()),
            original_text=text,
        )

    def _on_multimodal_query(self, msg: MultimodalQuery) -> None:
        """Handle a direct multimodal query (e.g. from behavior tree verification).

        Args:
            msg: Multimodal query with text, optional image flag, and processing prefs.
        """
        prompt = f"{self.system_prompt}\n\n{msg.text_query}"

        image_b64 = None
        if msg.include_current_image and self.latest_image is not None:
            image_b64 = self._encode_image(self.latest_image)

        temperature = msg.temperature if msg.temperature > 0 else self.temperature
        max_tokens = msg.max_tokens if msg.max_tokens > 0 else self.num_predict

        self._query_ollama(
            prompt=prompt,
            image_base64=image_b64,
            query_id=msg.query_id or str(uuid.uuid4()),
            temperature=temperature,
            num_predict=max_tokens,
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _query_ollama(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        query_id: str = "",
        original_text: str = "",
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> None:
        """Send a query to Ollama and publish results.

        Args:
            prompt: Full prompt including system prompt.
            image_base64: Optional base64 image.
            query_id: Tracking ID for the query.
            original_text: The user's original spoken text.
            temperature: Override sampling temperature.
            num_predict: Override max tokens.
        """
        start = time.time()

        result = self.ollama.generate(
            prompt=prompt,
            image_base64=image_base64,
            num_ctx=self.num_ctx,
            num_predict=num_predict or self.num_predict,
            temperature=temperature or self.temperature,
        )

        elapsed = time.time() - start
        self.inference_times.append(elapsed)
        self.query_count += 1

        # Check for errors
        if "error" in result:
            self.get_logger().error(f"Ollama error: {result['error']}")
            self._publish_error_response(query_id, result["error"])
            # Speak error to user
            tts_msg = String()
            tts_msg.data = "I'm sorry, my reasoning system is temporarily unavailable."
            self.tts_pub.publish(tts_msg)
            return

        response_text = result.get("response", "")
        total_duration_ns = result.get("total_duration", 0)
        model_used = result.get("model", self.ollama.model)

        self.get_logger().info(
            f"Ollama response in {elapsed:.2f}s "
            f"(server: {total_duration_ns / 1e9:.2f}s): {response_text[:100]}..."
        )

        # Publish raw response
        resp_msg = MultimodalResponse()
        resp_msg.header.stamp = self.get_clock().now().to_msg()
        resp_msg.query_id = query_id
        resp_msg.response_text = response_text
        resp_msg.confidence = 0.8  # Moondream doesn't provide confidence
        resp_msg.processing_time = elapsed
        resp_msg.model_used = model_used
        resp_msg.optimization_used = "ollama-gguf-q4"
        resp_msg.has_error = False
        self.response_pub.publish(resp_msg)

        # Try to parse structured intent
        intent = parse_json_intent(response_text)
        if intent:
            self._publish_command(intent, query_id, response_text)
        else:
            # No structured intent — treat as conversational response
            self.get_logger().info("No structured intent parsed — sending verbal response.")
            tts_msg = String()
            tts_msg.data = response_text
            self.tts_pub.publish(tts_msg)

    def _publish_command(self, intent: Dict[str, str], query_id: str, raw_response: str) -> None:
        """Publish a CognitiveCommand from a parsed intent.

        Args:
            intent: Parsed JSON intent dict.
            query_id: Tracking ID.
            raw_response: Original VLM response text.
        """
        cmd = CognitiveCommand()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.action = intent.get("action", "speak")
        cmd.target_object = intent.get("target", "")
        cmd.response_text = intent.get("explanation", raw_response)
        cmd.priority = 0.5
        cmd.timeout = 30.0
        cmd.requires_confirmation = False

        self.command_pub.publish(cmd)
        self.get_logger().info(
            f"Published command: action={cmd.action}, target={cmd.target_object}"
        )

        # Also speak the explanation
        if intent.get("explanation"):
            tts_msg = String()
            tts_msg.data = intent["explanation"]
            self.tts_pub.publish(tts_msg)

    def _publish_error_response(self, query_id: str, error_msg: str) -> None:
        """Publish an error MultimodalResponse.

        Args:
            query_id: Tracking ID.
            error_msg: Error description.
        """
        resp = MultimodalResponse()
        resp.header.stamp = self.get_clock().now().to_msg()
        resp.query_id = query_id
        resp.response_text = ""
        resp.confidence = 0.0
        resp.processing_time = 0.0
        resp.has_error = True
        resp.error_message = error_msg
        self.response_pub.publish(resp)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a numpy image to base64 JPEG string.

        Args:
            image: RGB numpy array.

        Returns:
            Base64-encoded JPEG string.
        """
        # Convert RGB to BGR for OpenCV encoding
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _health_check(self) -> None:
        """Periodic health check for the Ollama server."""
        status = String()
        if self.ollama.is_available():
            avg_time = (
                f"{sum(self.inference_times[-10:]) / min(len(self.inference_times), 10):.2f}s"
                if self.inference_times
                else "N/A"
            )
            status.data = f"OK | queries={self.query_count} | avg_latency={avg_time}"
        else:
            status.data = "ERROR | Ollama server unreachable"
            self.get_logger().warn("Ollama server health check failed!")

        self.status_pub.publish(status)

    def destroy_node(self) -> None:
        """Cleanup on shutdown."""
        self.get_logger().info(f"Shutting down cognitive client. Total queries: {self.query_count}")
        self.ollama.session.close()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = CognitiveClientNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in cognitive client node: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
