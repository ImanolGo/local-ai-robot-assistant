"""
Audio pipeline launch file.

Starts the self-contained audio processing pipeline:
- Audio capture node (includes wake word, VAD, and ASR internally)
- Audio playback node (includes Piper TTS internally)

Note: stt_node, tts_node, and wake_word_detector_node were deprecated in the
Phase 5 refactoring. All functionality is now integrated into the two nodes below.
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
            # Self-contained audio capture pipeline
            # (wake word + VAD + ASR all run internally)
            Node(
                package="audio_interface_nodes",
                executable="audio_capture_node",
                name="audio_capture_node",
                output="screen",
                parameters=[{"debug": debug}],
            ),
            # Audio playback with integrated Piper TTS
            Node(
                package="audio_interface_nodes",
                executable="audio_playback_node",
                name="audio_playback_node",
                output="screen",
                parameters=[{"debug": debug}],
            ),
        ]
    )
