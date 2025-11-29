#!/bin/bash
# Launch ROS2 Python nodes with venv support
# Usage: ./launch_node.sh <package_name> <node_name>
#
# Example: ./launch_node.sh audio_interface_nodes audio_capture_node

if [ $# -ne 2 ]; then
    echo "Usage: $0 <package_name> <node_name>"
    echo "Example: $0 audio_interface_nodes audio_capture_node"
    exit 1
fi

PACKAGE_NAME=$1
NODE_NAME=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source environments
source "$SCRIPT_DIR/ros2_venv.sh"

# Find the node Python module - check both install and source directories
NODE_MODULE=$(find "$SCRIPT_DIR/install/$PACKAGE_NAME/lib/python3.10/site-packages" -name "${NODE_NAME}.py" 2>/dev/null | head -1)

# If not found in install (e.g., symlink-install), check source directory
if [ -z "$NODE_MODULE" ]; then
    NODE_MODULE=$(find "$SCRIPT_DIR/src/$PACKAGE_NAME/$PACKAGE_NAME" -name "${NODE_NAME}.py" 2>/dev/null | head -1)
fi

if [ -z "$NODE_MODULE" ]; then
    echo "❌ Node not found: $NODE_NAME in package $PACKAGE_NAME"
    echo "   Searched in:"
    echo "   - $SCRIPT_DIR/install/$PACKAGE_NAME/lib/python3.10/site-packages"
    echo "   - $SCRIPT_DIR/src/$PACKAGE_NAME/$PACKAGE_NAME"
    exit 1
fi

echo "🚀 Launching: $NODE_NAME from $PACKAGE_NAME"
python "$NODE_MODULE"
