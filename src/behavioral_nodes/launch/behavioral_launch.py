"""
Behavioral architecture launch file.

Starts the command router node which bridges audio transcription
to the cognitive core and direct actuation for simple commands.

Note: behavior_tree_executor is planned for Phase 7+ and not yet implemented.
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug logging"
    )
    debug = LaunchConfiguration("debug")

    return LaunchDescription(
        [
            debug_arg,
            Node(
                package="behavioral_nodes",
                executable="command_router_node",
                name="command_router",
                output="screen",
                parameters=[{"debug": debug}],
            ),
        ]
    )
