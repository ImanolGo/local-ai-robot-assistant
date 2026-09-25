#!/usr/bin/env python3
"""
Cognitive Client Node — Ollama/Moondream Bridge for Local AI Robot Assistant.

This node bridges ROS2 with the local Ollama server running Moondream (1.6B VLM).
It receives transcription results and optionally camera snapshots, constructs prompts,
sends HTTP requests to the Ollama API, parses structured JSON intents, and publishes
CognitiveCommand messages to the command router.

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

from cognitive_core_nodes.llama_cpp_bridge import LlamaCppBridge, find_local_gguf
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


def maybe_fallback_to_ollama(
    bridge,
    backend: str,
    ollama_url: str,
    model_name: str,
    timeout: float,
    logger,
):
    """Fall back from an unavailable ``llamacpp`` bridge to Ollama HTTP.

    The in-process backend is the promoted default, but it needs the GGUF blobs
    and ``llama-cpp-python`` present. If it cannot load, prefer the Ollama
    daemon over a dead cognitive core. The chosen backend is logged and returned
    so callers can reflect it in status output.

    Args:
        bridge: The already-constructed primary bridge.
        backend: The requested backend name (``"llamacpp"`` or ``"ollama"``).
        ollama_url: Ollama base URL for the fallback.
        model_name: Ollama model name for the fallback.
        timeout: Ollama request timeout.
        logger: ROS2 logger instance.

    Returns:
        Tuple ``(bridge, backend)`` — the (possibly replaced) bridge and backend.
    """
    if backend != "llamacpp" or bridge.is_available():
        return bridge, backend

    logger.warn("llamacpp backend unavailable — attempting Ollama HTTP fallback")
    fallback = OllamaBridge(
        base_url=ollama_url,
        model=model_name,
        timeout=timeout,
        logger=logger,
    )
    if fallback.is_available():
        logger.warn("Fell back to the Ollama backend.")
        return fallback, "ollama"

    logger.error("Ollama fallback also unavailable — cognitive core is offline.")
    return bridge, backend


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


def resolve_system_prompt(default_prompt: str, override: Optional[str]) -> str:
    """Pick the per-query system prompt override, else the node default.

    Args:
        default_prompt: The node's configured system prompt.
        override: A query-supplied system prompt (may be blank).

    Returns:
        The trimmed override if non-empty, otherwise ``default_prompt``.
    """
    override = (override or "").strip()
    return override or default_prompt


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
        self.declare_parameter("cognitive_backend", "llamacpp")
        self.declare_parameter("ollama_url", "http://localhost:11434")
        self.declare_parameter("model_name", "moondream")
        self.declare_parameter("request_timeout", 10.0)
        self.declare_parameter("num_ctx", 512)
        self.declare_parameter("num_predict", 128)
        self.declare_parameter("temperature", 0.3)
        self.declare_parameter("system_prompt", SYSTEM_PROMPT)
        self.declare_parameter("enable_vision", True)
        self.declare_parameter("health_check_interval", 30.0)
        self.declare_parameter("llm_model_path", "")
        self.declare_parameter("llm_mmproj_path", "")
        self.declare_parameter("llm_n_gpu_layers", -1)
        self.declare_parameter("llm_n_ctx", 2048)
        self.declare_parameter("llm_flash_attn", True)
        self.declare_parameter("structured_output", True)

        # Read parameters
        backend = self.get_parameter("cognitive_backend").value
        ollama_url = self.get_parameter("ollama_url").value
        model_name = self.get_parameter("model_name").value
        self.model_name = model_name
        timeout = self.get_parameter("request_timeout").value
        self.num_ctx = self.get_parameter("num_ctx").value
        self.num_predict = self.get_parameter("num_predict").value
        self.temperature = self.get_parameter("temperature").value
        self.system_prompt = self.get_parameter("system_prompt").value
        self.enable_vision = self.get_parameter("enable_vision").value
        health_interval = self.get_parameter("health_check_interval").value
        llm_model_path = self.get_parameter("llm_model_path").value
        llm_mmproj_path = self.get_parameter("llm_mmproj_path").value
        llm_n_gpu_layers = self.get_parameter("llm_n_gpu_layers").value
        llm_n_ctx = self.get_parameter("llm_n_ctx").value
        llm_flash_attn = self.get_parameter("llm_flash_attn").value
        self.structured_output = self.get_parameter("structured_output").value

        # --- Cognitive backend bridge (Ollama HTTP vs in-process llama.cpp) ---
        self.backend = backend
        self.bridge = self._create_bridge(
            backend=backend,
            ollama_url=ollama_url,
            model_name=model_name,
            timeout=timeout,
            llm_model_path=llm_model_path,
            llm_mmproj_path=llm_mmproj_path,
            llm_n_gpu_layers=llm_n_gpu_layers,
            llm_n_ctx=llm_n_ctx,
            llm_flash_attn=llm_flash_attn,
        )

        # If the promoted in-process backend cannot load (missing blobs /
        # llama-cpp-python), fall back to the Ollama HTTP daemon so the robot
        # stays operational. The fallback is logged loudly and reflected in
        # self.backend (and therefore in /cognitive/status).
        self.bridge, self.backend = maybe_fallback_to_ollama(
            bridge=self.bridge,
            backend=self.backend,
            ollama_url=ollama_url,
            model_name=model_name,
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
        if self.bridge.is_available():
            self.get_logger().info(
                f"✅ Cognitive client initialized — backend='{self.backend}' "
                f"(model='{model_name}')"
            )
        else:
            self.get_logger().warn(
                f"⚠️  Cognitive backend '{self.backend}' not ready. "
                "Node will retry on incoming queries."
            )

    def _create_bridge(
        self,
        backend: str,
        ollama_url: str,
        model_name: str,
        timeout: float,
        llm_model_path: str,
        llm_mmproj_path: str,
        llm_n_gpu_layers: int,
        llm_n_ctx: int,
        llm_flash_attn: bool = True,
    ):
        """Instantiate the selected cognitive backend bridge.

        Args:
            backend: Either ``"ollama"`` (HTTP daemon) or ``"llamacpp"``
                (in-process llama.cpp).
            ollama_url: Ollama base URL (used for the ``ollama`` backend).
            model_name: Model identifier (Ollama model name).
            timeout: Request timeout for the Ollama backend.
            llm_model_path: GGUF path for the llama.cpp backend. Empty string
                triggers auto-discovery from local files then Ollama blobs.
            llm_mmproj_path: Multimodal projector GGUF path for vision.
            llm_n_gpu_layers: Layers to offload for the llama.cpp backend.
            llm_n_ctx: Context size for the llama.cpp backend (must fit the
                Moondream image tokens, ~729, plus prompt/output).
            llm_flash_attn: Enable flash attention for the llama.cpp context
                (major vision e2e win on Orin).

        Returns:
            An object exposing ``is_available()`` and ``generate()``.
        """
        if backend != "llamacpp":
            return OllamaBridge(
                base_url=ollama_url,
                model=model_name,
                timeout=timeout,
                logger=self.get_logger(),
            )

        # Resolve GGUF paths: explicit params > local models dir > Ollama blobs.
        model_path = llm_model_path
        mmproj_path = llm_mmproj_path
        if not model_path:
            local = find_local_gguf()
            model_path = local.get("model_path", "")
            mmproj_path = mmproj_path or local.get("mmproj_path", "")

        return LlamaCppBridge(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=llm_n_ctx,
            n_gpu_layers=llm_n_gpu_layers,
            flash_attn=llm_flash_attn,
            logger=self.get_logger(),
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

        # Query the selected cognitive backend
        self._query_bridge(
            prompt=prompt,
            image_base64=image_b64,
            query_id=str(uuid.uuid4()),
            original_text=text,
            force_json=self._structured_output_enabled(),
        )

    def _on_multimodal_query(self, msg: MultimodalQuery) -> None:
        """Handle a direct multimodal query (e.g. from visual verification).

        The query may supply its own system prompt (``msg.system_prompt``); when
        blank the node default is used.

        Args:
            msg: Multimodal query with text, optional image flag, and processing prefs.
        """
        prompt = (
            f"{resolve_system_prompt(self.system_prompt, msg.system_prompt)}\n\n{msg.text_query}"
        )

        image_b64 = None
        if msg.include_current_image and self.latest_image is not None:
            image_b64 = self._encode_image(self.latest_image)

        temperature = msg.temperature if msg.temperature > 0 else self.temperature
        max_tokens = msg.max_tokens if msg.max_tokens > 0 else self.num_predict

        self._query_bridge(
            prompt=prompt,
            image_base64=image_b64,
            query_id=msg.query_id or str(uuid.uuid4()),
            temperature=temperature,
            num_predict=max_tokens,
            # `use_optimizations` requests a structured intent (e.g. command
            # router). Verification queries leave it False to get free text.
            force_json=self._structured_output_enabled() and bool(msg.use_optimizations),
        )

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _structured_output_enabled(self) -> bool:
        """Whether the active backend can constrain output to the intent schema."""
        return bool(self.structured_output and self.backend == "llamacpp")

    def _query_bridge(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        query_id: str = "",
        original_text: str = "",
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        force_json: bool = False,
    ) -> None:
        """Send a query to the active cognitive backend and publish results.

        Args:
            prompt: Full prompt including system prompt.
            image_base64: Optional base64 image.
            query_id: Tracking ID for the query.
            original_text: The user's original spoken text.
            temperature: Override sampling temperature.
            num_predict: Override max tokens.
            force_json: Constrain the output to the intent JSON schema when the
                active backend supports it. Only set this for command-intent
                queries — verification prompts expect free text (Yes/No).
        """
        start = time.time()

        result = self.bridge.generate(
            prompt=prompt,
            image_base64=image_base64,
            num_ctx=self.num_ctx,
            num_predict=num_predict or self.num_predict,
            temperature=temperature or self.temperature,
            force_json=force_json,
        )

        elapsed = time.time() - start
        self.inference_times.append(elapsed)
        self.query_count += 1

        # Check for errors
        if "error" in result:
            self.get_logger().error(f"Cognitive backend error: {result['error']}")
            self._publish_error_response(query_id, result["error"])
            # Speak error to user
            tts_msg = String()
            tts_msg.data = "I'm sorry, my reasoning system is temporarily unavailable."
            self.tts_pub.publish(tts_msg)
            return

        response_text = result.get("response", "")
        total_duration_ns = result.get("total_duration", 0)
        model_used = result.get("model", self.model_name)
        optimization = "llamacpp-gguf" if self.backend == "llamacpp" else "ollama-gguf-q4"

        self.get_logger().info(
            f"Cognitive response in {elapsed:.2f}s "
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
        resp_msg.optimization_used = optimization
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
        """Periodic health check for the active cognitive backend."""
        status = String()
        if self.bridge.is_available():
            avg_time = (
                f"{sum(self.inference_times[-10:]) / min(len(self.inference_times), 10):.2f}s"
                if self.inference_times
                else "N/A"
            )
            status.data = (
                f"OK | backend={self.backend} | queries={self.query_count} | "
                f"avg_latency={avg_time}"
            )
        else:
            status.data = f"ERROR | cognitive backend '{self.backend}' unavailable"
            self.get_logger().warn("Cognitive backend health check failed!")

        self.status_pub.publish(status)

    def destroy_node(self) -> None:
        """Cleanup on shutdown."""
        self.get_logger().info(f"Shutting down cognitive client. Total queries: {self.query_count}")
        # OllamaBridge exposes a requests.Session; LlamaCppBridge exposes close().
        session = getattr(self.bridge, "session", None)
        if session is not None:
            session.close()
        close = getattr(self.bridge, "close", None)
        if callable(close):
            close()
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
