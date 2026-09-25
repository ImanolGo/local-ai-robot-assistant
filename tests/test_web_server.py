#!/usr/bin/env python3
"""
Unit tests for the monitoring web server endpoints.

Exercises the FastAPI app directly (no ROS2 runtime required).
"""

import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from web_interface_nodes.web_server import SUBSYSTEMS, SystemState, create_app, read_thermal_zones


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
        self.assertIn("subsystems", body)
        self.assertIn("temperatures_c", body)
        self.assertIn("disk", body)

    def test_status_reflects_updates(self):
        self.state.cognitive_status = "ERROR | backend unavailable"
        body = self.client.get("/status").json()
        self.assertEqual(body["cognitive_status"], "ERROR | backend unavailable")

    def test_api_resources(self):
        body = self.client.get("/api/resources").json()
        self.assertIn("cpu_percent", body)
        self.assertIn("used_mb", body["memory"])
        self.assertIsInstance(body["temperatures_c"], dict)
        self.assertIsInstance(body["load_avg"], list)

    def test_api_subsystems_has_all_keys(self):
        body = self.client.get("/api/subsystems").json()
        self.assertEqual(set(body.keys()), set(SUBSYSTEMS))
        for entry in body.values():
            self.assertIn("status", entry)
            self.assertIn("age_s", entry)
            self.assertIn("stale", entry)

    def test_dashboard_served_as_html(self):
        for path in ("/", "/dashboard"):
            with self.subTest(path=path):
                resp = self.client.get(path)
                self.assertEqual(resp.status_code, 200)
                self.assertIn("text/html", resp.headers["content-type"])
                self.assertIn("/status", resp.text)


class TestSystemState(unittest.TestCase):
    """Tests for the subsystem cache and staleness tracking."""

    def test_update_records_value_and_timestamp(self):
        state = SystemState()
        state.update("audio", "wake_word_detected")
        snapshot = state.subsystem_snapshot()
        self.assertEqual(snapshot["audio"]["status"], "wake_word_detected")
        self.assertIsNotNone(snapshot["audio"]["age_s"])
        self.assertFalse(snapshot["audio"]["stale"])

    def test_unseen_subsystem_is_stale(self):
        state = SystemState()
        snapshot = state.subsystem_snapshot()
        self.assertIsNone(snapshot["cognitive"]["age_s"])
        self.assertTrue(snapshot["cognitive"]["stale"])

    def test_stale_after_threshold(self):
        state = SystemState(stale_after_s=10.0)
        state.update("chassis", "battery=12.0V")
        state._seen["chassis"] = time.time() - 30.0
        self.assertTrue(state.subsystem_snapshot()["chassis"]["stale"])

    def test_update_unknown_subsystem_raises(self):
        state = SystemState()
        with self.assertRaises(KeyError):
            state.update("not_a_subsystem", "x")


class TestReadThermalZones(unittest.TestCase):
    """Tests for the /sys thermal-zone reader."""

    def test_reads_type_and_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            zone = os.path.join(tmp, "thermal_zone0")
            os.makedirs(zone)
            with open(os.path.join(zone, "type"), "w", encoding="utf-8") as fh:
                fh.write("cpu-thermal\n")
            with open(os.path.join(zone, "temp"), "w", encoding="utf-8") as fh:
                fh.write("42500\n")

            temps = read_thermal_zones(tmp)

        self.assertEqual(temps, {"cpu-thermal": 42.5})

    def test_missing_directory_returns_empty(self):
        self.assertEqual(read_thermal_zones("/nonexistent/thermal"), {})


if __name__ == "__main__":
    unittest.main()
