#!/bin/bash
# Test Reorganization Validation Script
# Validates that reorganized tests still work correctly

set -e

echo "🧪 Validating test reorganization..."

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Build workspace
echo "🔨 Building workspace..."
colcon build --packages-select perception_nodes actuation_nodes localization_nodes

# Source workspace
source install/setup.bash

echo "🧪 Running unit tests (in-package)..."
colcon test --packages-select perception_nodes actuation_nodes localization_nodes

echo "🧪 Running integration tests (top-level)..."
pytest integration_tests/ -v -m "not hardware"

echo "🧪 Running hardware tests (if available)..."
if [ "$1" = "--include-hardware" ]; then
    pytest integration_tests/ -v -m "hardware"
    pytest hardware_tests/ -v
else
    echo "  (Skipped - use --include-hardware to run hardware tests)"
fi

echo "✅ All tests validated successfully!"
