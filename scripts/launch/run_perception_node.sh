#!/bin/bash
# Wrapper to run perception nodes directly
# Usage: ./run_perception_node.sh <node_name>

if [ -z "$1" ]; then
    echo "Usage: $0 <node_name>"
    echo ""
    echo "Available nodes:"
    echo "  - camera_driver"
    echo "  - depth_estimation_node"
    echo "  - depth_estimator"
    echo "  - image_undistort_node"
    echo "  - object_detector"
    echo "  - pointcloud_generator"
    exit 1
fi

NODE_NAME="$1"
shift  # Remove first argument, keep the rest as node arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source ROS2 and workspace first
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/install/setup.bash"

# Add venv site-packages to PYTHONPATH so system Python can find pycuda, tensorrt, etc.
export PYTHONPATH="$REPO_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Run the node directly
exec "$REPO_ROOT/install/perception_nodes/bin/$NODE_NAME" "$@"
