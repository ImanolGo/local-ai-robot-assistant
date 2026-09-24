"""
Web interface launch file.

Starts the minimal FastAPI health/status server (Plan.md Phase 5 item 4).
The full dashboard is deferred until the rest of the system is stable.

Usage:
    ros2 launch web_interface_nodes web_interface_launch.py
    ros2 launch web_interface_nodes web_interface_launch.py port:=8080
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    host_arg = DeclareLaunchArgument(
        "host", default_value="0.0.0.0", description="HTTP bind address"
    )
    port_arg = DeclareLaunchArgument("port", default_value="8080", description="HTTP port")

    return LaunchDescription(
        [
            host_arg,
            port_arg,
            Node(
                package="web_interface_nodes",
                executable="web_server",
                name="web_server",
                output="screen",
                parameters=[
                    {
                        "host": LaunchConfiguration("host"),
                        "port": LaunchConfiguration("port"),
                    }
                ],
            ),
        ]
    )
