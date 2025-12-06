#!/bin/bash
# Test script for the refactored self-contained audio pipeline
# This script runs the audio_capture_node and monitors its output topics

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Audio Pipeline Test Script${NC}"
echo -e "${BLUE}Self-Contained Pipeline with VAD + Whisper${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: Not running on Jetson. Some features may not work.${NC}"
fi

# Source ROS2 environment
echo -e "${GREEN}[1/5] Sourcing ROS2 environment...${NC}"
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo -e "${RED}Error: ROS2 Humble not found${NC}"
    exit 1
fi

# Source workspace
echo -e "${GREEN}[2/5] Sourcing workspace...${NC}"
if [ -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    source "$WORKSPACE_ROOT/install/setup.bash"
else
    echo -e "${RED}Error: Workspace not built. Run 'colcon build' first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}[3/5] Activating virtual environment...${NC}"
if [ -f "$WORKSPACE_ROOT/.venv/bin/activate" ]; then
    source "$WORKSPACE_ROOT/.venv/bin/activate"
else
    echo -e "${RED}Error: Virtual environment not found at $WORKSPACE_ROOT/.venv${NC}"
    exit 1
fi

# Check if audio device is available
echo -e "${GREEN}[4/5] Checking audio device...${NC}"
if arecord -l | grep -q "USB PnP Sound Device"; then
    echo -e "${GREEN}✓ USB microphone detected${NC}"
else
    echo -e "${YELLOW}Warning: USB PnP Sound Device not found${NC}"
    echo "Available devices:"
    arecord -l
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Audio Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}The pipeline will:${NC}"
echo -e "  1. Listen for wake word (\"Hey Rover\")"
echo -e "  2. Activate VAD when wake word detected"
echo -e "  3. Capture speech until silence detected"
echo -e "  4. Transcribe speech using Whisper"
echo -e "  5. Publish transcription to /audio/transcription"
echo ""
echo -e "${YELLOW}Topics to monitor:${NC}"
echo -e "  ${GREEN}/audio/events${NC}        - Wake word, speech start/end events"
echo -e "  ${GREEN}/audio/transcription${NC} - Transcribed text with confidence"
echo ""
echo -e "${YELLOW}To monitor topics in separate terminals:${NC}"
echo -e "  ${BLUE}Terminal 2:${NC} source .venv/bin/activate && ros2 topic echo /audio/events"
echo -e "  ${BLUE}Terminal 3:${NC} source .venv/bin/activate && ros2 topic echo /audio/transcription"
echo ""
echo -e "${YELLOW}Test procedure:${NC}"
echo -e "  1. Wait for node to initialize (~5 seconds)"
echo -e "  2. Say: ${GREEN}\"Hey Rover\"${NC}"
echo -e "  3. Wait for wake word detection"
echo -e "  4. Say: ${GREEN}\"What time is it?\"${NC} (or any command)"
echo -e "  5. Wait for transcription result"
echo ""
echo -e "${RED}Press Ctrl+C to stop${NC}"
echo ""

# Give user time to read
sleep 3

echo -e "${GREEN}[5/5] Starting audio_capture_node...${NC}"
echo ""

# Run the node
ros2 run audio_interface_nodes audio_capture_node
