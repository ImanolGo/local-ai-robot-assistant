"""
Behavioral architecture launch file.

Starts the command router node (audio transcription -> cognitive core /
actuation) and the visual verification node (goal-completion checking).
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug logging"
    )
    debug = LaunchConfiguration("debug")

    enable_verification_arg = DeclareLaunchArgument(
        "enable_verification",
        default_value="true",
        description="Start the visual verification loop node",
    )
    enable_verification = LaunchConfiguration("enable_verification")

    return LaunchDescription(
        [
            debug_arg,
            enable_verification_arg,
            Node(
                package="behavioral_nodes",
                executable="command_router_node",
                name="command_router",
                output="screen",
                parameters=[{"debug": debug}],
            ),
            Node(
                package="behavioral_nodes",
                executable="visual_verification_node",
                name="visual_verification_node",
                output="screen",
                condition=IfCondition(enable_verification),
            ),
        ]
    )
