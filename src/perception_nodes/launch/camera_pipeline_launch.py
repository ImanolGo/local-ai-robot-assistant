#!/usr/bin/env python3
"""
Launch file for the complete camera pipeline.
Starts camera_driver and image_undistort_node together.

Author: Local AI Robot Team
License: Apache-2.0
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for camera pipeline."""

    # Declare launch arguments
    use_gpu_arg = DeclareLaunchArgument(
        "use_gpu",
        default_value="true",
        description="Enable GPU acceleration for undistortion",
    )

    camera_device_arg = DeclareLaunchArgument(
        "camera_device", default_value="0", description="Camera device ID"
    )

    publish_camera_info_arg = DeclareLaunchArgument(
        "publish_camera_info",
        default_value="true",
        description="Publish camera info messages",
    )

    enable_monitoring_arg = DeclareLaunchArgument(
        "enable_monitoring",
        default_value="true",
        description="Enable performance monitoring",
    )

    # Camera driver node
    camera_driver_node = Node(
        package="perception_nodes",
        executable="camera_driver",
        name="camera_driver",
        output="screen",
        parameters=[
            {
                "device_id": LaunchConfiguration("camera_device"),
                "publish_camera_info": LaunchConfiguration("publish_camera_info"),
                "enable_monitoring": LaunchConfiguration("enable_monitoring"),
            }
        ],
        remappings=[
            ("/camera/raw", "/camera/raw"),
            ("/camera/camera_info", "/camera/camera_info"),
        ],
    )

    # Image undistortion node
    image_undistort_node = Node(
        package="perception_nodes",
        executable="image_undistort_node",
        name="image_undistort_node",
        output="screen",
        parameters=[
            {
                "use_gpu": LaunchConfiguration("use_gpu"),
                "enable_monitoring": LaunchConfiguration("enable_monitoring"),
            }
        ],
        remappings=[
            ("/camera/raw", "/camera/raw"),
            ("/camera/undistorted", "/camera/undistorted"),
        ],
    )

    return LaunchDescription(
        [
            use_gpu_arg,
            camera_device_arg,
            publish_camera_info_arg,
            enable_monitoring_arg,
            camera_driver_node,
            image_undistort_node,
        ]
    )
