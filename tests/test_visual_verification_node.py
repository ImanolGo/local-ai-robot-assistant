#!/usr/bin/env python3
"""
Unit tests for the visual verification node (Plan.md Phase 5 item 3).

Covers the pure logic — answer interpretation, prompt construction and rotation
timing — without requiring a ROS2 runtime or hardware.
"""

import math
import unittest

from behavioral_nodes.visual_verification_node import (
    VERIFICATION_SYSTEM_PROMPT,
    build_verification_prompt,
    parse_verification_answer,
    rotation_duration,
)


class TestParseVerificationAnswer(unittest.TestCase):
    """Tests for interpreting VLM yes/no answers."""

    def test_leading_yes(self):
        self.assertIs(parse_verification_answer("Yes"), True)
        self.assertIs(parse_verification_answer("yes, I can see it."), True)
        self.assertIs(parse_verification_answer('"Yes."'), True)

    def test_leading_no(self):
        self.assertIs(parse_verification_answer("No"), False)
        self.assertIs(parse_verification_answer("No, it is not there."), False)

    def test_affirmative_phrasing(self):
        self.assertIs(parse_verification_answer("I can see the red ball."), True)
        self.assertIs(parse_verification_answer("The cup is visible."), True)

    def test_negative_phrasing(self):
        self.assertIs(parse_verification_answer("I don't see it."), False)
        self.assertIs(parse_verification_answer("The target is absent."), False)

    def test_unsure(self):
        self.assertIsNone(parse_verification_answer("Maybe later."))
        self.assertIsNone(parse_verification_answer("banana"))

    def test_empty(self):
        self.assertIsNone(parse_verification_answer(""))
        self.assertIsNone(parse_verification_answer(None))


class TestBuildVerificationPrompt(unittest.TestCase):
    """Tests for the verification prompt builder."""

    def test_includes_goal_and_boolean_instruction(self):
        prompt = build_verification_prompt("red ball")
        self.assertIn("red ball", prompt)
        self.assertIn("Yes", prompt)
        self.assertIn("No", prompt)


class TestVerificationSystemPrompt(unittest.TestCase):
    """Tests for the dedicated (non-JSON) verification system prompt."""

    def test_requests_single_word_boolean(self):
        self.assertIn("Yes or No", VERIFICATION_SYSTEM_PROMPT)
        self.assertIn("single word", VERIFICATION_SYSTEM_PROMPT)

    def test_does_not_request_intent_schema(self):
        self.assertNotIn("respond ONLY with a valid JSON object", VERIFICATION_SYSTEM_PROMPT)


class TestRotationDuration(unittest.TestCase):
    """Tests for in-place rotation timing."""

    def test_45_deg_at_half_rad_s(self):
        expected = math.radians(45) / 0.5
        self.assertAlmostEqual(rotation_duration(45.0, 0.5), expected)

    def test_negative_angle_uses_magnitude(self):
        self.assertAlmostEqual(rotation_duration(-45.0, 0.5), math.radians(45) / 0.5)

    def test_zero_speed_returns_zero(self):
        self.assertEqual(rotation_duration(45.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
