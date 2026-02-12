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
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[LaunchConfiguration("config_file")],
                remappings=[("odometry/filtered", "odometry/filtered")],
            ),
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
            # uart_imu_node is now integrated into uart_motor_controller
            # to resolve serial port conflicts on /dev/ttyTHS1
        ]
    )
