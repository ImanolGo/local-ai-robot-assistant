#!/bin/bash
# Convenience wrapper to run audio test with proper ROS2 environment

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Source ROS2 workspace
source "$WORKSPACE_DIR/install/setup.bash"

# Run the test
python3 "$SCRIPT_DIR/test_audio_capture_playback.py" "$@"
