#!/bin/bash
# Helper script to monitor audio pipeline topics in a split terminal setup
# This script opens multiple terminals to monitor different aspects of the pipeline

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}Audio Pipeline Monitor${NC}"
echo -e "${YELLOW}This will open 3 terminals:${NC}"
echo -e "  1. Audio capture node"
echo -e "  2. Events monitor (/audio/events)"
echo -e "  3. Transcription monitor (/audio/transcription)"
echo ""

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo -e "${GREEN}Using tmux for split terminal view${NC}"

    # Create a new tmux session
    SESSION="audio_pipeline"

    # Kill existing session if it exists
    tmux kill-session -t $SESSION 2>/dev/null || true

    # Create new session with first window
    tmux new-session -d -s $SESSION -n "Audio Pipeline"

    # Split window horizontally
    tmux split-window -h -t $SESSION

    # Split the right pane vertically
    tmux split-window -v -t $SESSION

    # Set up pane 0 (left) - Audio capture node
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "echo 'Starting audio_capture_node with venv...'" C-m
    tmux send-keys -t $SESSION:0.0 "./scripts/launch/run_audio_capture_node.sh" C-m

    # Set up pane 1 (top right) - Events monitor
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.1 "echo 'Monitoring /audio/events...'" C-m
    tmux send-keys -t $SESSION:0.1 "sleep 5 && ros2 topic echo /audio/events" C-m

    # Set up pane 2 (bottom right) - Transcription monitor
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.2 "echo 'Monitoring /audio/transcription...'" C-m
    tmux send-keys -t $SESSION:0.2 "sleep 5 && ros2 topic echo /audio/transcription" C-m

    # Attach to the session
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    echo -e "${YELLOW}Use Ctrl+B then D to detach${NC}"
    echo -e "${YELLOW}Use Ctrl+C in each pane to stop${NC}"
    sleep 1
    tmux attach-session -t $SESSION

else
    echo -e "${YELLOW}tmux not found. Opening terminals manually...${NC}"
    echo ""
    echo -e "${BLUE}Please open 3 separate terminals and run:${NC}"
    echo ""
    echo -e "${GREEN}Terminal 1 (Audio Node):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 run audio_interface_nodes audio_capture_node"
    echo ""
    echo -e "${GREEN}Terminal 2 (Events):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 topic echo /audio/events"
    echo ""
    echo -e "${GREEN}Terminal 3 (Transcription):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 topic echo /audio/transcription"
    echo ""
fi
