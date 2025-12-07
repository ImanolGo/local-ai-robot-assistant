#!/bin/bash
# Wrapper to run audio interface nodes with proper Python environment
# Usage: ./run_audio_node.sh <node_name>

if [ -z "$1" ]; then
    echo "Usage: $0 <node_name>"
    echo ""
    echo "Available nodes:"
    echo "  - audio_capture_node"
    echo "  - wake_word_detector_node"
    echo "  - speech_recognition_node"
    echo "  - tts_node"
    exit 1
fi

NODE_NAME="$1"
shift  # Remove first argument, keep the rest as node arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source ROS2 and workspace first
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/install/setup.bash"

# Add venv site-packages to PYTHONPATH so ROS2 Python can find packages
export PYTHONPATH="$REPO_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Run the node directly
exec "$REPO_ROOT/install/audio_interface_nodes/lib/audio_interface_nodes/$NODE_NAME" "$@"
