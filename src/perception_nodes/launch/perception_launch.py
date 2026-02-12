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
        executable="depth_estimation_node",
        name="depth_estimator",
        parameters=[{"debug": debug}],
        remappings=[
            ("camera/undistorted", "/camera/undistorted"),
            ("perception/depth", "/perception/depth"),
            ("perception/depth_viz", "/perception/depth_viz"),
        ],
        output="screen",
    )

    # Point cloud generator node
    pointcloud_generator_node = Node(
        package="perception_nodes",
        executable="pointcloud_generator",
        name="pointcloud_generator",
        parameters=[
            {
                "debug": debug,
                "depth_range_min": 0.1,
                "depth_range_max": 10.0,
                "downsample_factor": 2,
                "enable_rgb": True,
            }
        ],
        remappings=[
            ("perception/depth", "/perception/depth"),
            ("camera/undistorted", "/camera/undistorted"),
            ("camera_info", "/camera/camera_info"),
            ("perception/pointcloud", "/perception/pointcloud"),
        ],
        output="screen",
    )

    # Staggered launch to prevent OOM / CPU spike
    from launch.actions import TimerAction

    object_detector_delayed = TimerAction(period=5.0, actions=[object_detector_node])

    depth_estimator_delayed = TimerAction(period=10.0, actions=[depth_estimator_node])

    pointcloud_generator_delayed = TimerAction(period=12.0, actions=[pointcloud_generator_node])

    return LaunchDescription(
        [
            debug_arg,
            camera_config_arg,
            calibration_config_arg,
            camera_driver_node,
            undistort_node,
            object_detector_delayed,
            depth_estimator_delayed,
            pointcloud_generator_delayed,
        ]
    )
