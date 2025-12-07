#!/bin/bash
# Combined audio pipeline monitor - runs both capture and playback nodes
# This gives you full audio functionality: wake word detection + TTS + notifications

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}Full Audio Pipeline Monitor${NC}"
echo -e "${YELLOW}This will open 4 terminals:${NC}"
echo -e "  1. Audio capture node (wake word + transcription)"
echo -e "  2. Audio playback node (TTS + notifications)"
echo -e "  3. Audio events monitor"
echo -e "  4. Transcription monitor"
echo ""

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo -e "${GREEN}Using tmux for split terminal view${NC}"

    # Create a new tmux session
    SESSION="audio_full"

    # Kill existing session if it exists
    tmux kill-session -t $SESSION 2>/dev/null || true

    # Create new session with first window
    tmux new-session -d -s $SESSION -n "Audio Pipeline"

    # Create 2x2 grid layout
    # Split window horizontally
    tmux split-window -h -t $SESSION

    # Split left pane vertically
    tmux split-window -v -t $SESSION:0.0

    # Split right pane vertically
    tmux split-window -v -t $SESSION:0.1

    # Set up pane 0 (top left) - Audio capture node
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting audio_capture_node...${NC}'" C-m
    tmux send-keys -t $SESSION:0.0 "./scripts/run_audio_capture_node.sh" C-m

    # Set up pane 1 (top right) - Audio playback node
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${GREEN}Starting audio_playback_node...${NC}'" C-m
    tmux send-keys -t $SESSION:0.1 "./scripts/run_audio_playback_node.sh" C-m

    # Set up pane 2 (bottom left) - Events monitor
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring /audio/events...${NC}'" C-m
    tmux send-keys -t $SESSION:0.2 "echo 'Waiting for nodes to start...'" C-m
    tmux send-keys -t $SESSION:0.2 "sleep 5 && ros2 topic echo /audio/events" C-m

    # Set up pane 3 (bottom right) - Transcription monitor
    tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.3 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring /audio/transcription...${NC}'" C-m
    tmux send-keys -t $SESSION:0.3 "echo 'Waiting for nodes to start...'" C-m
    tmux send-keys -t $SESSION:0.3 "sleep 5 && ros2 topic echo /audio/transcription" C-m

    # Attach to the session
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    echo -e "${YELLOW}Layout:${NC}"
    echo -e "  ┌─────────────────┬─────────────────┐"
    echo -e "  │  Capture Node   │  Playback Node  │"
    echo -e "  ├─────────────────┼─────────────────┤"
    echo -e "  │  Events         │  Transcription  │"
    echo -e "  └─────────────────┴─────────────────┘"
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Ctrl+B then D     - Detach from session"
    echo -e "  Ctrl+B then Arrow - Navigate between panes"
    echo -e "  Ctrl+C            - Stop process in current pane"
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo -e "  Say 'Hey Rover' to trigger wake word detection"
    echo -e "  You'll hear an ascending notification sound"
    echo -e "  Then speak your command"
    echo -e "  You'll hear a descending sound when speech ends"
    echo -e "  Transcription will appear in bottom-right pane"
    echo ""
    sleep 2
    tmux attach-session -t $SESSION

else
    echo -e "${YELLOW}tmux not found. Please install tmux: sudo apt install tmux${NC}"
    exit 1
fi
