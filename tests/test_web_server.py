#!/usr/bin/env python3
"""
Unit tests for the minimal web server health/status endpoints.

Exercises the FastAPI app directly (no ROS2 runtime required).
"""

import unittest

from fastapi.testclient import TestClient

from web_interface_nodes.web_server import SystemState, create_app


class TestWebServerEndpoints(unittest.TestCase):
    """Tests for the /health and /status endpoints."""

    def setUp(self):
        self.state = SystemState()
        self.state.cognitive_status = "OK | backend=ollama | queries=3"
        self.state.verification_status = "idle | goal='red ball' | attempt=0"
        self.client = TestClient(create_app(self.state))

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_status_contains_subsystems(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("cpu_percent", body)
        self.assertIn("memory", body)
        self.assertEqual(body["cognitive_status"], "OK | backend=ollama | queries=3")
        self.assertEqual(body["verification_status"], "idle | goal='red ball' | attempt=0")
        self.assertGreaterEqual(body["uptime_s"], 0)

    def test_status_reflects_updates(self):
        self.state.cognitive_status = "ERROR | backend unavailable"
        body = self.client.get("/status").json()
        self.assertEqual(body["cognitive_status"], "ERROR | backend unavailable")


if __name__ == "__main__":
    unittest.main()
