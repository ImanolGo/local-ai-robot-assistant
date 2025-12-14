#!/bin/bash
# Combined perception pipeline monitor
# This gives you full visibility into the vision stack: Camera -> YOLO -> Events

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}Full Perception Pipeline Monitor${NC}"
echo -e "${YELLOW}This will open 4 terminals:${NC}"
echo -e "  1. Perception Pipeline (Camera, YOLO, Depth, etc.)"
echo -e "  2. Perception Events (ENTERED/LEFT/MOVED)"
echo -e "  3. Camera Input Stats (FPS)"
echo -e "  4. Object Detection Stats (Hz)"
echo ""

# Stop any existing ROS2 daemon to prevent XMLRPC errors
echo -e "${YELLOW}Stopping ROS2 daemon to ensure clean state...${NC}"
ros2 daemon stop > /dev/null 2>&1 || true
sleep 1

if command -v tmux &> /dev/null; then
    echo -e "${GREEN}Using tmux for split terminal view${NC}"

    # Create a new tmux session
    SESSION="perception_full"

    # Kill existing session if it exists
    tmux kill-session -t $SESSION 2>/dev/null || true

    # Create new session with first window
    tmux new-session -d -s $SESSION -n "Perception Pipeline"

    # Create 2x2 grid layout
    # Split window horizontally
    tmux split-window -h -t $SESSION

    # Split left pane vertically
    tmux split-window -v -t $SESSION:0.0

    # Split right pane vertically
    tmux split-window -v -t $SESSION:0.1

    # Set up pane 0 (top left) - Main Pipeline
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting perception pipeline...${NC}'" C-m
    tmux send-keys -t $SESSION:0.0 "./scripts/launch/run_perception_pipeline.sh" C-m

    # Set up pane 1 (top right) - Events Monitor
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}Monitoring /perception/events...${NC}'" C-m
    tmux send-keys -t $SESSION:0.1 "echo 'Waiting for nodes to start...'" C-m
    tmux send-keys -t $SESSION:0.1 "sleep 20 && ros2 topic echo /perception/events" C-m

    # Set up pane 2 (bottom left) - Camera Input Stats
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring Camera FPS (/camera/undistorted)...${NC}'" C-m
    tmux send-keys -t $SESSION:0.2 "echo 'Waiting for nodes to start...'" C-m
    tmux send-keys -t $SESSION:0.2 "sleep 10 && ros2 topic hz /camera/undistorted" C-m

    # Set up pane 3 (bottom right) - Object Detector Stats
    tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.3 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring Detector Output (/perception/objects)...${NC}'" C-m
    tmux send-keys -t $SESSION:0.3 "echo 'Waiting for nodes to start...'" C-m
    tmux send-keys -t $SESSION:0.3 "sleep 20 && ros2 topic hz /perception/objects" C-m

    # Attach to the session
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    echo -e "${YELLOW}Layout:${NC}"
    echo -e "  ┌─────────────────┬─────────────────┐"
    echo -e "  │  Perception App │  Events Feed    │"
    echo -e "  ├─────────────────┼─────────────────┤"
    echo -e "  │  Camera Hz      │  Objects Hz     │"
    echo -e "  └─────────────────┴─────────────────┘"
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Ctrl+B then D     - Detach from session"
    echo -e "  Ctrl+B then Arrow - Navigate between panes"
    echo -e "  Ctrl+C            - Stop process in current pane"
    echo ""
    sleep 2
    tmux attach-session -t $SESSION

else
    echo -e "${YELLOW}tmux not found. Please install tmux: sudo apt install tmux${NC}"
    exit 1
fi
