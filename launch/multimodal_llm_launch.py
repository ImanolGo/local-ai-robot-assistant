#!/usr/bin/env python3
"""
Launch file for Gemma 3n E2B Multimodal LLM Node

This launch file starts the multimodal language model node with proper configuration
for the Jetson Orin Nano platform.

Usage:
    ros2 launch cognitive_core_nodes multimodal_llm_launch.py
    ros2 launch cognitive_core_nodes multimodal_llm_launch.py use_optimum_nvidia:=false
"""

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate the launch description for multimodal LLM node."""

    return LaunchDescription(
        [
            # Launch arguments
            DeclareLaunchArgument(
                "use_optimum_nvidia",
                default_value="true",
                description="Use Optimum-NVIDIA optimization if available",
            ),
            DeclareLaunchArgument(
                "use_fp8",
                default_value="false",
                description="Use FP8 optimization (experimental)",
            ),
            DeclareLaunchArgument(
                "max_new_tokens",
                default_value="256",
                description="Maximum number of tokens to generate",
            ),
            DeclareLaunchArgument(
                "temperature",
                default_value="0.7",
                description="Sampling temperature for generation",
            ),
            DeclareLaunchArgument("log_level", default_value="info", description="ROS2 log level"),
            # Environment variables for CUDA optimization
            SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", "0"),
            SetEnvironmentVariable("TOKENIZERS_PARALLELISM", "false"),
            # Multimodal LLM Node
            Node(
                package="cognitive_core_nodes",
                executable="multimodal_llm_node.py",
                name="multimodal_llm_node",
                output="screen",
                parameters=[
                    {
                        "use_optimum_nvidia": LaunchConfiguration("use_optimum_nvidia"),
                        "use_fp8": LaunchConfiguration("use_fp8"),
                        "max_new_tokens": LaunchConfiguration("max_new_tokens"),
                        "max_prompt_length": 1024,
                        "max_output_length": 2048,
                        "max_batch_size": 1,
                        "temperature": LaunchConfiguration("temperature"),
                        "top_k": 40,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1,
                    }
                ],
                arguments=[
                    "--ros-args",
                    "--log-level",
                    LaunchConfiguration("log_level"),
                ],
                remappings=[
                    ("/camera/undistorted", "/perception/camera/undistorted"),
                    ("/cognitive/multimodal_query", "/cognitive/query"),
                    ("/cognitive/multimodal_response", "/cognitive/response"),
                ],
            ),
        ]
    )
