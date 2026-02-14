"""
SLAM Launch File — RTAB-Map Visual SLAM for Jetson Orin Nano.

Launches:
  1. Static TFs: base_link → camera_link → camera_optical_frame
  2. rgbd_odometry: Visual odometry from RGB + monocular depth
  3. rtabmap: Full SLAM with loop closure and mapping

Prerequisites:
  - Perception pipeline running (camera_driver, undistort, depth_estimation)
  - Actuation/IMU running (uart_motor_controller publishes /imu/data)
  - rtabmap_ros installed (see scripts/install_rtabmap.sh)

Topics consumed:
  /camera/undistorted   — RGB image (BEST_EFFORT QoS)
  /perception/depth     — Depth map, 32FC1 (default QoS)
  /camera/camera_info   — Camera intrinsics (BEST_EFFORT QoS)
  /imu/data             — IMU orientation (BEST_EFFORT QoS)

Topics produced:
  /rtabmap/odom         — Visual odometry
  /rtabmap/mapData      — 3D map data
  /rtabmap/grid_map     — 2D occupancy grid
  /map                  — OccupancyGrid for navigation
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory("localization_nodes")
    rtabmap_config = os.path.join(pkg_share, "config", "rtabmap_config.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rtabmap_config",
                default_value=rtabmap_config,
                description="Path to the RTAB-Map config file",
            ),
            DeclareLaunchArgument(
                "delete_db",
                default_value="true",
                description="Delete RTAB-Map database on startup (fresh map)",
            ),
            DeclareLaunchArgument(
                "debug",
                default_value="false",
                description="Enable debug output",
            ),
            # -------------------------------------------------------
            # Static TF: base_link → camera_link
            # Camera is mounted on the front of the Wave Rover chassis.
            # Adjust x/y/z to match the actual mounting position.
            # x=forward, y=left, z=up (REP-103)
            # -------------------------------------------------------
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_camera_link",
                arguments=[
                    "--x",
                    "0.1",  # 10cm forward from base_link
                    "--y",
                    "0",
                    "--z",
                    "0.15",  # 15cm above base_link
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    "base_link",
                    "--child-frame-id",
                    "camera_link",
                ],
            ),
            # -------------------------------------------------------
            # Static TF: camera_link → camera_optical_frame
            # Standard REP-103 rotation:
            #   camera_link:          x=forward, y=left, z=up
            #   camera_optical_frame: z=forward, x=right, y=down
            # Rotation: roll=-π/2, yaw=-π/2
            # -------------------------------------------------------
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_link_to_optical",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "-1.5707963",  # -π/2
                    "--pitch",
                    "0",
                    "--yaw",
                    "-1.5707963",  # -π/2
                    "--frame-id",
                    "camera_link",
                    "--child-frame-id",
                    "camera_optical_frame",
                ],
            ),
            # -------------------------------------------------------
            # Visual Odometry (rgbd_odometry)
            # Uses direct depth subscription (no rgbd_sync needed).
            # Publishes to /rtabmap/odom for EKF fusion.
            # -------------------------------------------------------
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                output="screen",
                parameters=[LaunchConfiguration("rtabmap_config")],
                remappings=[
                    ("rgb/image", "/camera/undistorted"),
                    ("depth/image", "/perception/depth"),
                    ("rgb/camera_info", "/camera/camera_info"),
                    ("odom", "/rtabmap/odom"),
                    ("imu", "/imu/data"),
                ],
            ),
            # -------------------------------------------------------
            # SLAM Node (rtabmap)
            # Builds and maintains the map, detects loop closures.
            # Publishes map → odom TF.
            # -------------------------------------------------------
            Node(
                package="rtabmap_slam",
                executable="rtabmap",
                name="rtabmap",
                output="screen",
                parameters=[LaunchConfiguration("rtabmap_config")],
                remappings=[
                    ("rgb/image", "/camera/undistorted"),
                    ("depth/image", "/perception/depth"),
                    ("rgb/camera_info", "/camera/camera_info"),
                    ("odom", "/rtabmap/odom"),
                    ("imu", "/imu/data"),
                ],
                arguments=["-d"],  # Delete database at startup for fresh map
            ),
        ]
    )
