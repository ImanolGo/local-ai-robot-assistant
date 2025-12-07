#!/bin/bash
# Check which processes are using audio devices

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Checking Audio Device Usage${NC}"
echo ""

# Check for processes using audio devices
echo -e "${YELLOW}Processes using audio devices:${NC}"
if command -v fuser &> /dev/null; then
    # Check speaker (card 0)
    echo -e "${GREEN}Speaker (card 0):${NC}"
    fuser -v /dev/snd/pcmC0D0p 2>&1 | grep -v "Cannot" || echo "  Not in use"

    # Check microphone (card 1)
    echo -e "${GREEN}Microphone (card 1):${NC}"
    fuser -v /dev/snd/pcmC1D0c 2>&1 | grep -v "Cannot" || echo "  Not in use"
else
    echo -e "${RED}fuser command not found. Install with: sudo apt install psmisc${NC}"
fi

echo ""
echo -e "${YELLOW}ROS2 audio nodes running:${NC}"
if pgrep -f "audio_playback_node" > /dev/null; then
    echo -e "${RED}✗ audio_playback_node is running (PID: $(pgrep -f audio_playback_node))${NC}"
    echo "  Stop with: pkill -f audio_playback_node"
else
    echo -e "${GREEN}✓ audio_playback_node not running${NC}"
fi

if pgrep -f "audio_capture_node" > /dev/null; then
    echo -e "${RED}✗ audio_capture_node is running (PID: $(pgrep -f audio_capture_node))${NC}"
    echo "  Stop with: pkill -f audio_capture_node"
else
    echo -e "${GREEN}✓ audio_capture_node not running${NC}"
fi

echo ""
echo -e "${YELLOW}ALSA device status:${NC}"
cat /proc/asound/card*/pcm*/sub*/status 2>/dev/null | grep -E "state:|owner_pid" || echo "No detailed status available"

echo ""
echo -e "${BLUE}Summary:${NC}"
if pgrep -f "audio.*node" > /dev/null; then
    echo -e "${YELLOW}Audio nodes are running. Stop them before running hardware tests.${NC}"
    echo ""
    echo -e "To stop all audio nodes:"
    echo -e "  ${GREEN}pkill -f 'audio.*node'${NC}"
else
    echo -e "${GREEN}No audio nodes running. Safe to run hardware tests.${NC}"
fi
