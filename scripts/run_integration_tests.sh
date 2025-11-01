#!/usr/bin/env bash
# Script to build and source the ROS2 workspace for testing

set -e

echo "Building ROS2 workspace..."
colcon build --symlink-install

echo "Sourcing workspace..."
source install/setup.bash

echo "Running integration tests..."
python3 integration_tests/test_uart_integration.py --port /dev/ttyTHS1
