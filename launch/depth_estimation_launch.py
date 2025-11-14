#!/usr/bin/env python3
"""
Launch file for Depth Estimation Pipeline
Integrates with camera pipeline for real-time depth estimation

Launches:
- Camera driver node
- Image undistortion node
- Depth estimation node
"""

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Launch arguments
    camera_config_arg = DeclareLaunchArgument(
        "camera_config_path",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("local_ai_robot_assistant"),
                "config",
                "camera_config.yaml",
            ]
        ),
        description="Path to camera configuration file",
    )

    calibration_config_arg = DeclareLaunchArgument(
        "calibration_config_path",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("local_ai_robot_assistant"),
                "config",
                "camera_calibration.yaml",
            ]
        ),
        description="Path to camera calibration file",
    )

    depth_engine_arg = DeclareLaunchArgument(
        "depth_engine_path",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("local_ai_robot_assistant"),
                "models",
                "depth_trt",
                "depth_anything_v2_small.trt",
            ]
        ),
        description="Path to TensorRT depth estimation engine",
    )

    depth_config_arg = DeclareLaunchArgument(
        "depth_config_path",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("local_ai_robot_assistant"),
                "models",
                "depth_trt",
                "config.json",
            ]
        ),
        description="Path to depth model configuration",
    )

    publish_colored_arg = DeclareLaunchArgument(
        "publish_colored",
        default_value="true",
        description="Whether to publish colored depth visualizations",
    )

    publish_obstacles_arg = DeclareLaunchArgument(
        "publish_obstacles",
        default_value="true",
        description="Whether to publish obstacle detection",
    )

    obstacle_threshold_arg = DeclareLaunchArgument(
        "obstacle_threshold_m",
        default_value="2.0",
        description="Obstacle detection threshold in meters",
    )

    frame_skip_arg = DeclareLaunchArgument(
        "frame_skip",
        default_value="1",
        description="Process every N-th frame (1=no skipping)",
    )

    # Camera driver node
    camera_driver_node = Node(
        package="perception_nodes",
        executable="camera_driver",
        name="camera_driver",
        parameters=[
            LaunchConfiguration("camera_config_path"),
            {
                "frame_id": "camera_frame",
                "publish_rate": 30.0,
                "auto_exposure": True,
                "auto_white_balance": True,
            },
        ],
        remappings=[
            ("/camera/image_raw", "/camera/image_raw"),
            ("/camera/camera_info", "/camera/camera_info"),
        ],
        output="screen",
    )

    # Image undistortion node
    undistort_node = Node(
        package="perception_nodes",
        executable="image_undistort_node",
        name="image_undistort_node",
        parameters=[
            LaunchConfiguration("calibration_config_path"),
            {
                "output_frame_id": "camera_frame_undistorted",
                "publish_rate": 30.0,
            },
        ],
        remappings=[
            ("/camera/image_raw", "/camera/image_raw"),
            ("/camera/camera_info", "/camera/camera_info"),
            ("/camera/image_undistorted", "/camera/image_undistorted"),
        ],
        output="screen",
    )

    # Depth estimation node
    depth_estimation_node = Node(
        package="perception_nodes",
        executable="depth_estimation_node",
        name="depth_estimation_node",
        parameters=[
            {
                "engine_path": LaunchConfiguration("depth_engine_path"),
                "config_path": LaunchConfiguration("depth_config_path"),
                "publish_colored": LaunchConfiguration("publish_colored"),
                "publish_stats": True,
                "publish_obstacles": LaunchConfiguration("publish_obstacles"),
                "obstacle_threshold_m": LaunchConfiguration("obstacle_threshold_m"),
                "obstacle_roi_height": 0.3,
                "max_depth_m": 10.0,
                "frame_skip": LaunchConfiguration("frame_skip"),
            }
        ],
        remappings=[
            ("/camera/image_undistorted", "/camera/image_undistorted"),
            ("/camera/camera_info", "/camera/camera_info"),
            ("/perception/depth", "/perception/depth"),
            ("/perception/depth_colored", "/perception/depth_colored"),
            ("/perception/depth_stats", "/perception/depth_stats"),
            ("/perception/obstacles", "/perception/obstacles"),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            # Launch arguments
            camera_config_arg,
            calibration_config_arg,
            depth_engine_arg,
            depth_config_arg,
            publish_colored_arg,
            publish_obstacles_arg,
            obstacle_threshold_arg,
            frame_skip_arg,
            # Nodes
            camera_driver_node,
            undistort_node,
            depth_estimation_node,
        ]
    )
