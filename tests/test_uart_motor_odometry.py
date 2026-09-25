#!/usr/bin/env python3
"""
Unit tests for the motor controller's dead-reckoning odometry integration.

Exercises the pure ``integrate_odometry`` helper without a ROS runtime or serial.
"""

import math
import unittest

from actuation_nodes.uart_motor_controller import UARTMotorController


class TestIntegrateOdometry(unittest.TestCase):
    """Tests for the differential-drive pose integration."""

    def test_straight_forward(self):
        x, y, yaw = UARTMotorController.integrate_odometry(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(yaw, 0.0)

    def test_in_place_rotation(self):
        x, y, yaw = UARTMotorController.integrate_odometry(0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(yaw, 1.0)

    def test_arc_uses_current_heading(self):
        # Start facing +y, drive forward: x should not change, y increases.
        x, y, yaw = UARTMotorController.integrate_odometry(0.0, 0.0, math.pi / 2.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)
        self.assertAlmostEqual(yaw, math.pi / 2.0)

    def test_zero_dt_is_noop(self):
        pose = (1.0, 2.0, 3.0)
        self.assertEqual(UARTMotorController.integrate_odometry(*pose, 1.0, 1.0, 0.0), pose)


if __name__ == "__main__":
    unittest.main()
