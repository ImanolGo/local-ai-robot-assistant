"""Localization and SLAM nodes for the Local AI Robot Assistant.

This package contains nodes for:
- IMU data acquisition via UART from Wave Rover
- Visual SLAM using RTAB-Map
- Sensor fusion with robot_localization EKF
- Pose estimation and mapping
"""

__all__ = ["uart_imu_node", "slam_node"]
__version__ = "0.1.0"
