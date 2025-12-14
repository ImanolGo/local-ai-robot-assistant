#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Source ROS2 and venv
source "$PROJECT_ROOT/ros2_venv.sh"

echo "Starting Perception Pipeline..."
# Launch the perception pipeline using the launch file we verified
# Note: we launch the python file directly since it might not be installed in share yet if not built/installed fully
ros2 launch "$PROJECT_ROOT/launch/perception_launch.py"
