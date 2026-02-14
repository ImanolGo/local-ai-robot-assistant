"""
Localization Launch File — EKF Sensor Fusion.

Launches:
  1. ekf_node: Extended Kalman Filter fusing IMU + visual odometry
  2. Static TF: base_link → imu_link (identity, IMU is at robot center)

Sensor Inputs:
  /imu/data        — from uart_motor_controller (actuation_nodes)
  /rtabmap/odom    — from rgbd_odometry (slam_launch.py)

Output:
  /odometry/filtered  — Fused pose estimate (nav_msgs/Odometry)
  TF: odom → base_link

Prerequisites:
  - actuation_nodes must be running (provides /imu/data)
  - slam_launch.py should be running for visual odometry fusion
    (EKF will work with IMU-only if SLAM is not yet available)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("localization_nodes")
    config_file = os.path.join(pkg_share, "config", "localization_config.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=config_file,
                description="Path to the config file for the ekf_node",
            ),
            DeclareLaunchArgument(
                "debug",
                default_value="false",
                description="Enable debug output",
            ),
            # EKF node — fuses IMU orientation + visual odometry
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[LaunchConfiguration("config_file")],
                remappings=[("odometry/filtered", "odometry/filtered")],
            ),
            # Static TF: base_link → imu_link
            # IMU is integrated into the Wave Rover chassis (identity transform).
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_imu_link",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "base_link",
                    "--child-frame-id",
                    "imu_link",
                ],
            ),
        ]
    )
