#!/usr/bin/env python3
"""
Unit tests for the motor controller's Twist -> wheel-speed mapping, including
the DC-motor deadband compensation.
"""

import unittest

from actuation_nodes.uart_motor_controller import UARTMotorController


class TestTwistToWheelSpeeds(unittest.TestCase):
    """Tests for the differential-drive mapping."""

    WHEELBASE = 0.16
    MAX_LINEAR = 0.3
    MAX_SPEED = 0.5

    def _map(self, linear, angular, min_pwm=0.0):
        return UARTMotorController.twist_to_wheel_speeds(
            linear, angular, self.WHEELBASE, self.MAX_LINEAR, self.MAX_SPEED, min_pwm
        )

    def test_straight_forward_full_scale(self):
        left, right = self._map(self.MAX_LINEAR, 0.0)
        self.assertAlmostEqual(left, self.MAX_SPEED)
        self.assertAlmostEqual(right, self.MAX_SPEED)

    def test_pure_turn_is_differential(self):
        left, right = self._map(0.0, 1.0)
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)
        self.assertAlmostEqual(left, -0.1333, places=3)
        self.assertAlmostEqual(right, 0.1333, places=3)

    def test_input_is_clipped(self):
        left, right = self._map(10.0, 0.0)
        self.assertAlmostEqual(left, self.MAX_SPEED)
        self.assertAlmostEqual(right, self.MAX_SPEED)

    def test_stop_stays_zero_even_with_deadband(self):
        self.assertEqual(self._map(0.0, 0.0, min_pwm=0.3), (0.0, 0.0))

    def test_deadband_raises_small_commands(self):
        # A gentle 0.5 rad/s turn maps to ~0.067, below the motor deadband.
        left, right = self._map(0.0, 0.5, min_pwm=0.3)
        self.assertAlmostEqual(left, -0.3)
        self.assertAlmostEqual(right, 0.3)

    def test_deadband_disabled(self):
        left, right = self._map(0.0, 0.5, min_pwm=0.0)
        self.assertAlmostEqual(left, -0.0667, places=3)
        self.assertAlmostEqual(right, 0.0667, places=3)

    def test_large_command_not_altered_by_deadband(self):
        left, right = self._map(self.MAX_LINEAR, 0.0, min_pwm=0.3)
        self.assertAlmostEqual(left, self.MAX_SPEED)
        self.assertAlmostEqual(right, self.MAX_SPEED)


if __name__ == "__main__":
    unittest.main()
