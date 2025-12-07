#!/bin/bash
# Complete wake word detection test script
# This script starts all necessary nodes and monitors wake word detection

set -e

echo "=============================================="
echo "Wake Word Detection Full System Test"
echo "=============================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up from testing/audio/ to scripts/ to workspace root
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

cd "$WORKSPACE_DIR"

# Source ROS2 setup
echo "Sourcing ROS2 environment..."
source install/setup.bash

# Add venv site-packages to PYTHONPATH
export PYTHONPATH="$WORKSPACE_DIR/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Set optimal audio levels
echo -e "\n1. Setting optimal audio levels..."
./scripts/utils/set_audio_levels.sh

# Create temporary directory for logs
LOG_DIR="/tmp/wake_word_test_$$"
mkdir -p "$LOG_DIR"

echo -e "\n2. Starting audio capture node..."
./scripts/launch/run_audio_node.sh audio_capture_node > "$LOG_DIR/audio_capture.log" 2>&1 &
AUDIO_PID=$!
sleep 2

# 3. Wake word detection is now integrated into audio_capture_node
# No need to start separate detector node

# Check if audio node is running
if ! kill -0 $AUDIO_PID 2>/dev/null; then
    echo "ERROR: Audio capture node failed to start!"
    echo "Check log: $LOG_DIR/audio_capture.log"
    cat "$LOG_DIR/audio_capture.log"
    exit 1
fi

echo -e "\n✓ All nodes started successfully!"
echo -e "\n4. Checking active nodes and topics..."
echo "Active nodes:"
ros2 node list

echo -e "\nActive topics:"
ros2 topic list | grep audio

echo -e "\n5. Starting wake word detection monitor..."
echo "Say 'Hey Rover', 'Alexa', 'Hey Mycroft', or 'Hey Rhasspy'"
echo "Press Ctrl+C to stop"
echo "=============================================="
echo ""

# Cleanup function
cleanup() {
    echo -e "\n\nStopping nodes..."
    kill $AUDIO_PID 2>/dev/null || true
    sleep 1
    echo "Logs saved in: $LOG_DIR"
    echo "  - Audio capture: $LOG_DIR/audio_capture.log"
}

trap cleanup EXIT INT TERM

# Monitor wake word detections
python3 manual_tests/test_wake_word_live.py

# Wait for processes to finish (they won't, but this keeps script alive)
wait
