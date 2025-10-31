"""
Perception subsystem launch file.

Starts all perception-related nodes:
- Camera driver
- Image undistortion
- Object detection (YOLO)
- Depth estimation
"""

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Declare launch arguments
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug logging"
    )

    camera_config_arg = DeclareLaunchArgument(
        "camera_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("perception_nodes"), "config", "camera_config.yaml"]
        ),
        description="Path to camera configuration file",
    )

    calibration_config_arg = DeclareLaunchArgument(
        "calibration_config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("perception_nodes"), "config", "camera_calibration.yaml"]
        ),
        description="Path to camera calibration file",
    )

    # Get launch configurations
    debug = LaunchConfiguration("debug")
    camera_config = LaunchConfiguration("camera_config")
    calibration_config = LaunchConfiguration("calibration_config")

    # Camera driver node
    camera_driver_node = Node(
        package="perception_nodes",
        executable="camera_driver",
        name="camera_driver",
        parameters=[camera_config, {"debug": debug}],
        remappings=[
            ("camera/raw", "/camera/raw"),
            ("camera/camera_info", "/camera/camera_info"),
        ],
        output="screen",
    )

    # Image undistortion node
    undistort_node = Node(
        package="perception_nodes",
        executable="image_undistort_node",
        name="image_undistort_node",
        parameters=[calibration_config, {"debug": debug}],
        remappings=[
            ("camera/raw", "/camera/raw"),
            ("camera/undistorted", "/camera/undistorted"),
            ("camera/camera_info", "/camera/camera_info"),
        ],
        output="screen",
    )

    # Object detection node
    object_detector_node = Node(
        package="perception_nodes",
        executable="object_detector",
        name="object_detector",
        parameters=[{"debug": debug}],
        remappings=[
            ("camera/undistorted", "/camera/undistorted"),
            ("perception/objects", "/perception/objects"),
            ("perception/object_overlay", "/perception/object_overlay"),
        ],
        output="screen",
    )

    # Depth estimation node
    depth_estimator_node = Node(
        package="perception_nodes",
        executable="depth_estimator",
        name="depth_estimator",
        parameters=[{"debug": debug}],
        remappings=[
            ("camera/undistorted", "/camera/undistorted"),
            ("perception/depth", "/perception/depth"),
            ("perception/pointcloud", "/perception/pointcloud"),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            debug_arg,
            camera_config_arg,
            calibration_config_arg,
            camera_driver_node,
            undistort_node,
            object_detector_node,
            depth_estimator_node,
        ]
    )
