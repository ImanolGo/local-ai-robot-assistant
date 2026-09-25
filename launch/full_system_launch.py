"""
Full system launch file for the Local AI Robot Assistant.

This launch file starts all subsystems:
- Perception nodes (camera, object detection, depth estimation)
- Audio interface nodes (wake word, ASR, TTS)
- Behavioral architecture (command router + visual verification)
- Cognitive core
- Actuation nodes (motor control, IMU feedback)
- Web interface (optional)

Note: SLAM/localization is out of scope for the MVP (see docs/architecture.md
v4.0). IMU data is published by the motor controller via /imu/data.
"""

from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
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

    actuation_arg = DeclareLaunchArgument(
        "actuation",
        default_value="true",
        description="Whether to start the motor controller / actuation nodes",
    )

    cognitive_delay_arg = DeclareLaunchArgument(
        "cognitive_start_delay",
        default_value="0.0",
        description=(
            "Seconds to delay the cognitive core startup. Increase on the 8 GB "
            "Orin so heavy VLM weights load after the perception engines."
        ),
    )

    cognitive_backend_arg = DeclareLaunchArgument(
        "cognitive_backend",
        default_value="llamacpp",
        description="Cognitive backend: 'llamacpp' (in-process) or 'ollama' (HTTP)",
    )

    llm_n_gpu_layers_arg = DeclareLaunchArgument(
        "llm_n_gpu_layers",
        default_value="-1",
        description="GPU layers for the llamacpp backend (-1 = all)",
    )

    cognitive_arg = DeclareLaunchArgument(
        "cognitive",
        default_value="true",
        description="Whether to start the cognitive core (VLM) node",
    )

    # Get launch configurations
    web_interface = LaunchConfiguration("web_interface")
    debug = LaunchConfiguration("debug")
    actuation = LaunchConfiguration("actuation")
    cognitive_start_delay = LaunchConfiguration("cognitive_start_delay")
    cognitive_backend = LaunchConfiguration("cognitive_backend")
    llm_n_gpu_layers = LaunchConfiguration("llm_n_gpu_layers")
    cognitive = LaunchConfiguration("cognitive")

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
        launch_arguments={
            "debug": debug,
            "cognitive_backend": cognitive_backend,
            "llm_n_gpu_layers": llm_n_gpu_layers,
        }.items(),
    )

    # Delay the cognitive core so its VLM weights do not collide with the
    # perception engines' TensorRT GPU allocations on the 8 GB Orin. Optionally
    # omit it entirely (cognitive:=false) for actuation/memory-limited soaks.
    cognitive_group = GroupAction(
        condition=IfCondition(cognitive),
        actions=[
            TimerAction(
                period=cognitive_start_delay,
                actions=[cognitive_launch],
            )
        ],
    )

    # Actuation launch definition
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

    # Include actuation launch (conditional — disable for actuation-less soaks)
    actuation_group = GroupAction(
        condition=IfCondition(actuation),
        actions=[actuation_launch],
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

    return LaunchDescription(
        [
            web_interface_arg,
            debug_arg,
            actuation_arg,
            cognitive_delay_arg,
            cognitive_backend_arg,
            llm_n_gpu_layers_arg,
            cognitive_arg,
            perception_launch,
            audio_launch,
            behavioral_launch,
            cognitive_group,
            actuation_group,
            web_interface_group,
        ]
    )
