#!/usr/bin/env python3
"""
Integration test for the visual verification loop (Plan.md Phase 5 item 3).

Runs the real ``VisualVerificationNode`` in-process and simulates the cognitive
core's reply over ROS2, asserting the loop reaches the verified state.
"""

import time
import unittest

import rclpy
from std_msgs.msg import String

from behavioral_nodes.visual_verification_node import VerificationState, VisualVerificationNode
from robot_interfaces.msg import MultimodalQuery, MultimodalResponse


class TestVisualVerificationLoop(unittest.TestCase):
    """End-to-end test of trigger -> query -> answer -> outcome."""

    @classmethod
    def setUpClass(cls):
        # Other test modules may already own the rclpy context; only initialise
        # (and later shut down) if we are the ones creating it.
        cls._own_context = False
        try:
            rclpy.init()
            cls._own_context = True
        except RuntimeError:
            cls._own_context = False

    @classmethod
    def tearDownClass(cls):
        if cls._own_context:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def setUp(self):
        self.node = VisualVerificationNode()
        self.node.response_timeout = 5.0
        self.last_query = None
        self.tts_messages = []

        self.node.create_subscription(
            MultimodalQuery,
            "/cognitive/multimodal_query",
            self._capture_query,
            10,
        )
        self.rsp_pub = self.node.create_publisher(
            MultimodalResponse, "/cognitive/multimodal_response", 10
        )
        self.node.create_subscription(String, "/audio/tts_request", self._capture_tts, 10)

    def tearDown(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass

    def _capture_query(self, msg):
        self.last_query = msg

    def _capture_tts(self, msg):
        self.tts_messages.append(msg.data)

    def _spin_until(self, predicate, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def _drain(self, seconds=0.3):
        """Spin a little more so queued messages are delivered."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def test_positive_answer_verifies_goal(self):
        self.node.start_verification("red ball")
        self.assertTrue(self._spin_until(lambda: self.last_query is not None))
        self.assertEqual(self.last_query.text_query.count("red ball"), 1)

        resp = MultimodalResponse()
        resp.query_id = self.last_query.query_id
        resp.response_text = "Yes, I can see the red ball."
        resp.has_error = False
        self.rsp_pub.publish(resp)

        self.assertTrue(self._spin_until(lambda: self.node.state == VerificationState.IDLE))
        self._drain()
        self.assertEqual(self.node.attempt, 1)
        self.assertTrue(any("found" in m.lower() for m in self.tts_messages))

    def test_negative_answer_retries_with_rotation(self):
        self.node.start_verification("cup")
        self.assertTrue(self._spin_until(lambda: self.last_query is not None))

        resp = MultimodalResponse()
        resp.query_id = self.last_query.query_id
        resp.response_text = "No, I do not see it."
        resp.has_error = False
        self.rsp_pub.publish(resp)

        # After a negative answer the node should rotate and re-query.
        self.assertTrue(self._spin_until(lambda: self.node.attempt >= 2, timeout=4.0))
        self.assertGreaterEqual(self.node.attempt, 2)


if __name__ == "__main__":
    unittest.main()
