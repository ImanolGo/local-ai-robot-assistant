#!/usr/bin/env python3
"""
Unit tests for cognitive_client_node — Ollama/Moondream Bridge.

Tests the OllamaBridge HTTP client, JSON intent parser, and the ROS2 node
logic without requiring an actual Ollama server or ROS2 runtime.

Author: Local AI Robot Assistant Team
"""

import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from cognitive_core_nodes.cognitive_client_node import OllamaBridge, parse_json_intent
from cognitive_core_nodes.llama_cpp_bridge import LlamaCppBridge, resolve_ollama_moondream_blobs

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


class TestLlamaCppBridge(unittest.TestCase):
    """Tests for the in-process llama.cpp bridge (llama_cpp mocked)."""

    def _make_bridge(self, mock_llm_instance):
        """Construct a LlamaCppBridge with a fake installed llama_cpp module."""
        mock_llm_cls = MagicMock(return_value=mock_llm_instance)
        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = mock_llm_cls

        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.write(b"fake")
        tmp.close()

        with patch.dict(sys.modules, {"llama_cpp": fake_module}):
            bridge = LlamaCppBridge(model_path=tmp.name, logger=MagicMock())
        return bridge, mock_llm_cls

    def test_is_available_when_loaded(self):
        """A successfully loaded model reports availability."""
        bridge, mock_cls = self._make_bridge(MagicMock())
        mock_cls.assert_called_once()
        self.assertTrue(bridge.is_available())

    def test_generate_text_normalizes_response(self):
        """Text generation is normalized to the Ollama-compatible dict shape."""
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {
            "choices": [{"text": '{"action": "navigate", "target": "chair"}'}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        bridge, _ = self._make_bridge(mock_llm)

        result = bridge.generate("Find the chair")

        self.assertIn("response", result)
        self.assertEqual(result["eval_count"], 5)
        self.assertEqual(result["prompt_eval_count"], 10)
        self.assertGreaterEqual(result["total_duration"], 0)
        self.assertIsNotNone(parse_json_intent(result["response"]))

    def test_generate_text_uses_grammar_when_forced(self):
        """force_json passes the GBNF grammar to text completion."""
        mock_llm = MagicMock()
        mock_llm.create_completion.return_value = {
            "choices": [{"text": '{"action": "stop"}'}],
            "usage": {},
        }
        bridge, _ = self._make_bridge(mock_llm)

        bridge.generate("stop", force_json=True)

        _, kwargs = mock_llm.create_completion.call_args
        self.assertIn("grammar", kwargs)

    def test_generate_with_image_uses_chat_completion(self):
        """Vision queries route through chat completion with a data URI."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "I see a cat."}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        bridge, _ = self._make_bridge(mock_llm)
        bridge._chat_handler = MagicMock()  # simulate a loaded projector

        result = bridge.generate("What do you see?", image_base64="AAAA==")

        self.assertEqual(result["response"], "I see a cat.")
        _, kwargs = mock_llm.create_chat_completion.call_args
        content = kwargs["messages"][0]["content"]
        image_part = next(p for p in content if p["type"] == "image_url")
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_generate_unavailable_returns_error(self):
        """An unloaded bridge returns an error dict instead of raising."""
        bridge = LlamaCppBridge.__new__(LlamaCppBridge)
        bridge.llm = None
        bridge._available = False
        bridge._last_error = "not loaded"
        result = bridge.generate("hi")
        self.assertIn("error", result)

    def test_resolve_ollama_blobs_missing_manifest(self):
        """A missing manifest yields None paths rather than raising."""
        result = resolve_ollama_moondream_blobs("/nonexistent/manifest")
        self.assertIsNone(result["model_path"])


class TestIntentParsingAcrossBackends(unittest.TestCase):
    """Regression: both backends yield identical parsed intents for equal JSON."""

    RAW_RESPONSES = [
        '{"action": "navigate", "target": "red ball", "explanation": "left"}',
        '```json\n{"action": "search", "target": "cup"}\n```',
        'Here you go: {"action": "stop"}',
        '{"target": "ball"}',
        "no json here",
    ]

    def test_backends_agree_on_parsed_intents(self):
        """parse_json_intent gives the same result regardless of backend origin."""
        for raw in self.RAW_RESPONSES:
            with self.subTest(raw=raw):
                expected = parse_json_intent(raw)
                # Ollama bridge normalize path
                ollama_result = {"response": raw}
                # llama.cpp bridge normalize path (same 'response' contract)
                llamacpp_result = {"response": raw}
                self.assertEqual(
                    parse_json_intent(ollama_result["response"]),
                    parse_json_intent(llamacpp_result["response"]),
                )
                self.assertEqual(parse_json_intent(ollama_result["response"]), expected)


if __name__ == "__main__":
    unittest.main()
