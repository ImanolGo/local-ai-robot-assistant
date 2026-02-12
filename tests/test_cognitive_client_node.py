#!/usr/bin/env python3
"""
Unit tests for cognitive_client_node — Ollama/Moondream Bridge.

Tests the OllamaBridge HTTP client, JSON intent parser, and the ROS2 node
logic without requiring an actual Ollama server or ROS2 runtime.

Author: Local AI Robot Assistant Team
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from cognitive_core_nodes.cognitive_client_node import OllamaBridge, parse_json_intent

# Mock ROS2 and robot_interfaces imports before importing the module under test.
# This is necessary because robot_interfaces.msg requires a colcon-built workspace.
_mock_modules = {
    "rclpy": MagicMock(),
    "rclpy.node": MagicMock(),
    "cv_bridge": MagicMock(),
    "sensor_msgs.msg": MagicMock(),
    "std_msgs.msg": MagicMock(),
    "geometry_msgs.msg": MagicMock(),
    "robot_interfaces": MagicMock(),
    "robot_interfaces.msg": MagicMock(),
}
for mod_name, mock in _mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock


class TestParseJsonIntent(unittest.TestCase):
    """Tests for the JSON intent parser."""

    def test_valid_json_intent(self):
        """Test parsing a well-formed JSON intent."""
        response = '{"action": "navigate", "target": "red ball", "explanation": "I see it."}'
        result = parse_json_intent(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "navigate")
        self.assertEqual(result["target"], "red ball")
        self.assertEqual(result["explanation"], "I see it.")

    def test_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        response = '```json\n{"action": "search", "target": "cup"}\n```'
        result = parse_json_intent(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["target"], "cup")

    def test_json_with_surrounding_text(self):
        """Test parsing JSON embedded in surrounding prose."""
        response = 'Sure! Here is the result: {"action": "stop", "target": ""} Let me know.'
        result = parse_json_intent(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "stop")

    def test_missing_action_key(self):
        """Test that missing 'action' key returns None."""
        response = '{"target": "ball", "explanation": "found it"}'
        result = parse_json_intent(response)
        self.assertIsNone(result)

    def test_no_json_at_all(self):
        """Test plain text response with no JSON."""
        response = "I don't see anything interesting."
        result = parse_json_intent(response)
        self.assertIsNone(result)

    def test_empty_response(self):
        """Test empty string."""
        self.assertIsNone(parse_json_intent(""))
        self.assertIsNone(parse_json_intent(None))

    def test_malformed_json(self):
        """Test broken JSON."""
        response = '{"action": "navigate", "target": }'
        result = parse_json_intent(response)
        self.assertIsNone(result)

    def test_action_only_is_valid(self):
        """Test that having just 'action' key is sufficient."""
        response = '{"action": "explore"}'
        result = parse_json_intent(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "explore")
        self.assertEqual(result["target"], "")
        self.assertEqual(result["explanation"], "")


class TestOllamaBridge(unittest.TestCase):
    """Tests for the OllamaBridge HTTP client."""

    def setUp(self):
        """Set up a bridge instance with a mock logger."""
        self.logger = MagicMock()
        self.bridge = OllamaBridge(
            base_url="http://localhost:11434",
            model="moondream",
            timeout=5.0,
            logger=self.logger,
        )

    @patch("cognitive_core_nodes.cognitive_client_node.requests.Session")
    def test_is_available_success(self, mock_session_cls):
        """Test server availability check when Ollama is running."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp
        self.bridge.session = mock_session

        self.assertTrue(self.bridge.is_available())
        mock_session.get.assert_called_once()

    @patch("cognitive_core_nodes.cognitive_client_node.requests.Session")
    def test_is_available_failure(self, mock_session_cls):
        """Test server availability check when Ollama is down."""
        import requests

        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("refused")
        self.bridge.session = mock_session

        self.assertFalse(self.bridge.is_available())

    def test_generate_success(self):
        """Test successful generate request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "moondream",
            "response": '{"action": "navigate", "target": "chair"}',
            "total_duration": 1500000000,
        }
        mock_response.raise_for_status = MagicMock()

        self.bridge.session = MagicMock()
        self.bridge.session.post.return_value = mock_response

        result = self.bridge.generate("Find the chair")
        self.assertIn("response", result)
        self.assertEqual(result["model"], "moondream")

    def test_generate_timeout(self):
        """Test generate request with timeout."""
        import requests

        self.bridge.session = MagicMock()
        self.bridge.session.post.side_effect = requests.Timeout("timed out")

        result = self.bridge.generate("Hello")
        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    def test_generate_connection_error(self):
        """Test generate request when server is unreachable."""
        import requests

        self.bridge.session = MagicMock()
        self.bridge.session.post.side_effect = requests.ConnectionError("refused")

        result = self.bridge.generate("Hello")
        self.assertIn("error", result)
        self.assertIn("Cannot connect", result["error"])

    def test_generate_with_image(self):
        """Test that image is included in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "I see a cat."}
        mock_response.raise_for_status = MagicMock()

        self.bridge.session = MagicMock()
        self.bridge.session.post.return_value = mock_response

        _ = self.bridge.generate("What do you see?", image_base64="AAAA==")

        # Verify the payload included images
        call_kwargs = self.bridge.session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertIn("images", payload)
        self.assertEqual(payload["images"], ["AAAA=="])

    def test_generate_keep_alive_negative_one(self):
        """Test that keep_alive is set to -1 to keep model loaded."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "ok"}
        mock_response.raise_for_status = MagicMock()

        self.bridge.session = MagicMock()
        self.bridge.session.post.return_value = mock_response

        self.bridge.generate("test")

        call_kwargs = self.bridge.session.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        self.assertEqual(payload["keep_alive"], -1)


if __name__ == "__main__":
    unittest.main()
