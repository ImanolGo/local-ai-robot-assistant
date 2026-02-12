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
            # RGB-D Synchronization
            Node(
                package="rtabmap_sync",
                executable="rgbd_sync",
                name="rgbd_sync",
                output="screen",
                parameters=[{"approx_sync": True}],
                remappings=[
                    ("rgb/image", "/camera/undistorted"),
                    ("depth/image", "/perception/depth"),
                    ("rgb/camera_info", "/camera/camera_info"),
                    ("rgbd_image", "rgbd_image"),
                ],
            ),
            # Visual Odometry
            Node(
                package="rtabmap_odom",
                executable="rgbd_odometry",
                name="rgbd_odometry",
                output="screen",
                parameters=[LaunchConfiguration("rtabmap_config")],
                remappings=[
                    ("rgbd_image", "rgbd_image"),
                    ("odom", "/rtabmap/odom"),
                ],
            ),
            # SLAM (Map building)
            Node(
                package="rtabmap_slam",
                executable="rtabmap",
                name="rtabmap",
                output="screen",
                parameters=[LaunchConfiguration("rtabmap_config")],
                remappings=[
                    ("rgbd_image", "rgbd_image"),
                    ("odom", "/rtabmap/odom"),
                    ("imu", "/imu/data"),
                ],
                arguments=["-d"],  # Delete database at startup for fresh map
            ),
        ]
    )
