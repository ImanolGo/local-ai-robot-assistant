#!/usr/bin/env python3
"""
Unit tests for command_router_node — Voice command routing logic.

Tests the simple command regex matching and routing decisions
without requiring ROS2 runtime.

Author: Local AI Robot Assistant Team
"""

import re
import sys
import unittest
from unittest.mock import MagicMock

from behavioral_nodes.command_router_node import SIMPLE_COMMANDS

# Mock ROS2 and robot_interfaces imports before importing the module under test.
_mock_modules = {
    "rclpy": MagicMock(),
    "rclpy.node": MagicMock(),
    "geometry_msgs": MagicMock(),
    "geometry_msgs.msg": MagicMock(),
    "std_msgs": MagicMock(),
    "std_msgs.msg": MagicMock(),
    "robot_interfaces": MagicMock(),
    "robot_interfaces.msg": MagicMock(),
}
for mod_name, mock in _mock_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mock


class TestSimpleCommandPatterns(unittest.TestCase):
    """Test that simple command regex patterns match expected phrases."""

    def setUp(self):
        """Compile all patterns."""
        self.compiled = [
            (re.compile(pattern, re.IGNORECASE), action, response)
            for pattern, (action, response) in SIMPLE_COMMANDS.items()
        ]

    def _match(self, text: str):
        """Helper: return (action, response) or None."""
        for pattern, action, response in self.compiled:
            if pattern.search(text.lower()):
                return (action, response)
        return None

    # --- Emergency / stop ---
    def test_stop(self):
        result = self._match("stop")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")

    def test_halt(self):
        result = self._match("halt right now")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")

    def test_freeze(self):
        result = self._match("freeze!")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")

    def test_emergency(self):
        result = self._match("emergency")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")

    # --- Forward ---
    def test_go_forward(self):
        result = self._match("go forward")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "forward")

    def test_move_forward(self):
        result = self._match("please move forward")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "forward")

    # --- Backward ---
    def test_go_back(self):
        result = self._match("go back")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "backward")

    def test_reverse(self):
        result = self._match("reverse now")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "backward")

    # --- Turns ---
    def test_turn_left(self):
        result = self._match("turn left")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "turn_left")

    def test_turn_right(self):
        result = self._match("turn right please")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "turn_right")

    def test_turn_around(self):
        result = self._match("turn around")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "turn_around")

    def test_do_180(self):
        result = self._match("do a 180")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "turn_around")

    # --- Describe (complex-ish, but matched as simple to trigger VLM) ---
    def test_what_do_you_see(self):
        result = self._match("what do you see")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "describe")

    def test_describe(self):
        result = self._match("describe what's in front of you")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "describe")

    # --- Return home ---
    def test_go_home(self):
        result = self._match("go home")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "return_home")

    def test_return_to_base(self):
        result = self._match("return to base")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "return_home")

    # --- Complex commands that should NOT match ---
    def test_find_red_ball_no_match(self):
        """Complex command should not match any simple pattern."""
        result = self._match("find the red ball on the table")
        self.assertIsNone(result)

    def test_pick_up_cup_no_match(self):
        result = self._match("pick up the blue cup")
        self.assertIsNone(result)

    def test_navigate_to_kitchen_no_match(self):
        result = self._match("navigate to the kitchen please")
        self.assertIsNone(result)

    def test_gibberish_no_match(self):
        result = self._match("asdfghjkl")
        self.assertIsNone(result)


class TestCommandPriority(unittest.TestCase):
    """Test that stop/emergency commands match even when embedded in longer text."""

    def setUp(self):
        self.compiled = [
            (re.compile(pattern, re.IGNORECASE), action, response)
            for pattern, (action, response) in SIMPLE_COMMANDS.items()
        ]

    def _match(self, text):
        for pattern, action, response in self.compiled:
            if pattern.search(text.lower()):
                return (action, response)
        return None

    def test_stop_in_sentence(self):
        """'stop' should match even inside a longer sentence."""
        result = self._match("please stop moving right now")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")

    def test_emergency_in_context(self):
        result = self._match("this is an emergency situation")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "stop")


if __name__ == "__main__":
    unittest.main()
