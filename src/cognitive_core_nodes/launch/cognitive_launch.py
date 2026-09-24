#!/usr/bin/env python3
"""
Launch file for Cognitive Client Node (Ollama/Moondream Bridge).

This launch file starts the cognitive client node that connects to the local
Ollama server running Moondream for visual reasoning and command understanding.

Usage:
    ros2 launch cognitive_core_nodes cognitive_launch.py
    ros2 launch cognitive_core_nodes cognitive_launch.py model_name:=moondream
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate the launch description for cognitive client node."""

    return LaunchDescription(
        [
            # Launch arguments
            DeclareLaunchArgument(
                "cognitive_backend",
                default_value="ollama",
                description="Backend: 'ollama' (HTTP) or 'llamacpp' (in-process GGUF)",
            ),
            DeclareLaunchArgument(
                "model_name",
                default_value="moondream",
                description="Ollama model name to use for reasoning",
            ),
            DeclareLaunchArgument(
                "ollama_url",
                default_value="http://localhost:11434",
                description="Ollama API endpoint URL",
            ),
            DeclareLaunchArgument(
                "llm_model_path",
                default_value="",
                description="GGUF model path for the llamacpp backend (empty = auto-discover)",
            ),
            DeclareLaunchArgument(
                "llm_mmproj_path",
                default_value="",
                description="Multimodal projector GGUF path for the llamacpp backend",
            ),
            DeclareLaunchArgument(
                "llm_n_ctx",
                default_value="2048",
                description="Context size for the llamacpp backend (must fit image tokens)",
            ),
            DeclareLaunchArgument(
                "request_timeout",
                default_value="10.0",
                description="Request timeout in seconds",
            ),
            DeclareLaunchArgument(
                "num_predict",
                default_value="128",
                description="Maximum tokens to generate",
            ),
            DeclareLaunchArgument(
                "temperature",
                default_value="0.3",
                description="Sampling temperature for generation",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="ROS2 log level",
            ),
            # Environment variables for CUDA optimization
            SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0"),
            # Cognitive Client Node (Ollama or llama.cpp bridge)
            Node(
                package="cognitive_core_nodes",
                executable="cognitive_client_node",
                name="cognitive_client_node",
                output="screen",
                parameters=[
                    {
                        "cognitive_backend": LaunchConfiguration("cognitive_backend"),
                        "ollama_url": LaunchConfiguration("ollama_url"),
                        "model_name": LaunchConfiguration("model_name"),
                        "llm_model_path": LaunchConfiguration("llm_model_path"),
                        "llm_mmproj_path": LaunchConfiguration("llm_mmproj_path"),
                        "llm_n_ctx": LaunchConfiguration("llm_n_ctx"),
                        "request_timeout": LaunchConfiguration("request_timeout"),
                        "num_ctx": 512,
                        "num_predict": LaunchConfiguration("num_predict"),
                        "temperature": LaunchConfiguration("temperature"),
                        "enable_vision": True,
                        "health_check_interval": 30.0,
                    }
                ],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    LaunchConfiguration("log_level"),
                ],
            ),
        ]
    )
