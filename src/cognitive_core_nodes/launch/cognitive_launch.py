#!/usr/bin/env python3
"""
Launch file for Cognitive Client Node (in-process llama.cpp / Ollama bridge).

This launch file starts the cognitive client node that runs Moondream for
visual reasoning and command understanding. The promoted default is the
in-process ``llamacpp`` backend (GPU vision via mtmd + flash attention); the
Ollama HTTP daemon remains available via ``cognitive_backend:=ollama``.

Usage:
    ros2 launch cognitive_core_nodes cognitive_launch.py
    ros2 launch cognitive_core_nodes cognitive_launch.py cognitive_backend:=ollama
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    """Generate the launch description for cognitive client node."""

    return LaunchDescription(
        [
            # Launch arguments
            DeclareLaunchArgument(
                "cognitive_backend",
                default_value="llamacpp",
                description="Backend: 'llamacpp' (in-process GGUF, default) or 'ollama' (HTTP)",
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
                "llm_flash_attn",
                default_value="true",
                description="Enable flash attention for the llamacpp backend",
            ),
            DeclareLaunchArgument(
                "llm_n_gpu_layers",
                default_value="-1",
                description=(
                    "GPU layers for the llamacpp backend (-1 = all). Lower it to "
                    "free GPU memory when perception engines are also resident."
                ),
            ),
            DeclareLaunchArgument(
                "cuda_visible_devices",
                default_value="0",
                description=(
                    "CUDA devices visible to the cognitive node; 'none' hides CUDA "
                    "so the VLM runs entirely on CPU when GPU/nvmap is exhausted."
                ),
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
            # Cognitive Client Node (Ollama or llama.cpp bridge). CUDA visibility
            # is scoped to this process only ('none' => empty, CPU inference) so
            # it can differ from the perception nodes' GPU usage.
            Node(
                package="cognitive_core_nodes",
                executable="cognitive_client_node",
                name="cognitive_client_node",
                output="screen",
                additional_env={
                    "CUDA_VISIBLE_DEVICES": PythonExpression(
                        [
                            "'' if '",
                            LaunchConfiguration("cuda_visible_devices"),
                            "' == 'none' else '",
                            LaunchConfiguration("cuda_visible_devices"),
                            "'",
                        ]
                    )
                },
                parameters=[
                    {
                        "cognitive_backend": LaunchConfiguration("cognitive_backend"),
                        "ollama_url": LaunchConfiguration("ollama_url"),
                        "model_name": LaunchConfiguration("model_name"),
                        "llm_model_path": LaunchConfiguration("llm_model_path"),
                        "llm_mmproj_path": LaunchConfiguration("llm_mmproj_path"),
                        "llm_n_ctx": LaunchConfiguration("llm_n_ctx"),
                        "llm_flash_attn": LaunchConfiguration("llm_flash_attn"),
                        "llm_n_gpu_layers": LaunchConfiguration("llm_n_gpu_layers"),
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
