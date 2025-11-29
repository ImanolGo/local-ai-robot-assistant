#!/bin/bash
# Source both venv and ROS2 environment
# Usage: source ros2_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate venv first
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    return 1
fi

# Source ROS2 workspace
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
    echo "✅ ROS2 workspace sourced"
else
    echo "❌ ROS2 workspace not built. Run: colcon build"
    return 1
fi

# Add venv site-packages to PYTHONPATH so ROS2 nodes can find venv packages
VENV_SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"
export PYTHONPATH="$VENV_SITE_PACKAGES:$PYTHONPATH"

echo "✅ Environment ready"
echo "   Python: $(which python)"
echo "   PYTHONPATH includes venv packages"
