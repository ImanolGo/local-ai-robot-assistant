#!/bin/bash
# Launch script for audio_capture_node with proper venv activation
# This ensures all Python dependencies from .venv are available

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$WORKSPACE_ROOT"

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace
source install/setup.bash

# Activate venv
source .venv/bin/activate

# Run the node with venv Python
exec python3 -m audio_interface_nodes.audio_capture_node
