"""
Full system launch file for the Local AI Robot Assistant.

This launch file starts all subsystems:
- Perception nodes (camera, object detection, depth estimation)
- Audio interface nodes (wake word, ASR, TTS)
- Localization nodes (IMU, SLAM)
- Behavioral architecture
- Cognitive core
- Actuation nodes (motor control)
- Web interface (optional)
"""

from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    # Declare launch arguments
    web_interface_arg = DeclareLaunchArgument(
        "web_interface",
        default_value="true",
        description="Whether to start the web interface",
    )

    debug_arg = DeclareLaunchArgument(
        "debug", default_value="false", description="Enable debug mode for all nodes"
    )

    # Get launch configurations
    web_interface = LaunchConfiguration("web_interface")
    debug = LaunchConfiguration("debug")

    # Include perception launch
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("perception_nodes"),
                    "launch",
                    "perception_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include audio pipeline launch
    audio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("audio_interface_nodes"),
                    "launch",
                    "audio_pipeline_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include localization launch
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("localization_nodes"),
                    "launch",
                    "localization_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include behavioral architecture launch
    behavioral_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("behavioral_nodes"),
                    "launch",
                    "behavioral_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include cognitive core launch
    cognitive_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("cognitive_core_nodes"),
                    "launch",
                    "cognitive_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include actuation launch
    actuation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("actuation_nodes"),
                    "launch",
                    "actuation_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    # Include web interface launch (conditional)
    web_interface_group = GroupAction(
        condition=IfCondition(web_interface),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("web_interface_nodes"),
                            "launch",
                            "web_interface_launch.py",
                        ]
                    )
                ),
                launch_arguments={"debug": debug}.items(),
            )
        ],
    )

    # Include SLAM launch
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("localization_nodes"),
                    "launch",
                    "slam_launch.py",
                ]
            )
        ),
        launch_arguments={"debug": debug}.items(),
    )

    return LaunchDescription(
        [
            web_interface_arg,
            debug_arg,
            perception_launch,
            audio_launch,
            localization_launch,
            slam_launch,
            behavioral_launch,
            cognitive_launch,
            actuation_launch,
            web_interface_group,
        ]
    )
