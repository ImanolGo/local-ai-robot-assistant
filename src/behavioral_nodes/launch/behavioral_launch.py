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

    verification_response_timeout_arg = DeclareLaunchArgument(
        "verification_response_timeout",
        default_value="8.0",
        description=(
            "Seconds to wait for a verification answer. Increase when the VLM "
            "runs on CPU (GPU memory exhausted by perception)."
        ),
    )
    verification_response_timeout = LaunchConfiguration("verification_response_timeout")

    return LaunchDescription(
        [
            debug_arg,
            enable_verification_arg,
            verification_response_timeout_arg,
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
                parameters=[{"response_timeout": verification_response_timeout}],
            ),
        ]
    )
