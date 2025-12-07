#!/bin/bash
# Helper script to monitor audio playback node in a split terminal setup
# This script opens multiple terminals to monitor different aspects of playback

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}Audio Playback Node Monitor${NC}"
echo -e "${YELLOW}This will open 4 terminals:${NC}"
echo -e "  1. Audio playback node (with integrated TTS)"
echo -e "  2. TTS request publisher (for testing)"
echo -e "  3. Events monitor (/audio/events)"
echo -e "  4. Node logs (rosout)"
echo ""

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo -e "${GREEN}Using tmux for split terminal view${NC}"

    # Create a new tmux session
    SESSION="audio_playback"

    # Kill existing session if it exists
    tmux kill-session -t $SESSION 2>/dev/null || true

    # Create new session with first window
    tmux new-session -d -s $SESSION -n "Audio Playback"

    # Create 2x2 grid layout
    # Split window horizontally
    tmux split-window -h -t $SESSION

    # Split left pane vertically
    tmux split-window -v -t $SESSION:0.0

    # Split right pane vertically (after previous splits: 0=top-left, 1=right, 2=bottom-left)
    tmux split-window -v -t $SESSION:0.1

    # Set up pane 0 (top left) - Audio playback node
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting audio_playback_node with venv...${NC}'" C-m
    tmux send-keys -t $SESSION:0.0 "./scripts/run_audio_playback_node.sh" C-m

    # Set up pane 1 (bottom left) - TTS request publisher
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}TTS Request Publisher${NC}'" C-m
    tmux send-keys -t $SESSION:0.1 "echo 'Waiting for node to start...'" C-m
    tmux send-keys -t $SESSION:0.1 "sleep 3" C-m
    tmux send-keys -t $SESSION:0.1 "echo ''" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${GREEN}Test Commands:${NC}'" C-m
    tmux send-keys -t $SESSION:0.1 "echo '  Short TTS:'" C-m
    tmux send-keys -t $SESSION:0.1 "echo '    ros2 topic pub --once /audio/tts_request std_msgs/String \"data: Hello world\"'" C-m
    tmux send-keys -t $SESSION:0.1 "echo ''" C-m
    tmux send-keys -t $SESSION:0.1 "echo '  Wake word notification:'" C-m
    tmux send-keys -t $SESSION:0.1 "echo '    ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \"event_type: wake_word_detected\"'" C-m
    tmux send-keys -t $SESSION:0.1 "echo ''" C-m
    tmux send-keys -t $SESSION:0.1 "echo '  Speech end notification:'" C-m
    tmux send-keys -t $SESSION:0.1 "echo '    ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \"event_type: speech_ended\"'" C-m
    tmux send-keys -t $SESSION:0.1 "echo ''" C-m

    # Set up pane 2 (top right) - Events monitor
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring /audio/events...${NC}'" C-m
    tmux send-keys -t $SESSION:0.2 "sleep 5 && ros2 topic echo /audio/events" C-m

    # Set up pane 3 (bottom right) - Node logs
    tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.3 "source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate" C-m
    tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring node logs...${NC}'" C-m
    tmux send-keys -t $SESSION:0.3 "sleep 5 && ros2 topic echo /rosout --field msg" C-m

    # Attach to the session
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    echo -e "${YELLOW}Layout:${NC}"
    echo -e "  ┌─────────────────┬─────────────────┐"
    echo -e "  │  Playback Node  │  Events Monitor │"
    echo -e "  ├─────────────────┼─────────────────┤"
    echo -e "  │  TTS Commands   │   Node Logs     │"
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
    echo -e "${YELLOW}tmux not found. Opening terminals manually...${NC}"
    echo ""
    echo -e "${BLUE}Please open 4 separate terminals and run:${NC}"
    echo ""
    echo -e "${GREEN}Terminal 1 (Playback Node):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 run audio_interface_nodes audio_playback_node"
    echo ""
    echo -e "${GREEN}Terminal 2 (TTS Test Commands):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo ""
    echo "  # Test TTS:"
    echo "  ros2 topic pub --once /audio/tts_request std_msgs/String \"data: 'Hello world'\""
    echo ""
    echo "  # Test wake word notification:"
    echo "  ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \"event_type: 'wake_word_detected'\""
    echo ""
    echo "  # Test speech end notification:"
    echo "  ros2 topic pub --once /audio/events robot_interfaces/AudioEvent \"event_type: 'speech_ended'\""
    echo ""
    echo -e "${GREEN}Terminal 3 (Events Monitor):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 topic echo /audio/events"
    echo ""
    echo -e "${GREEN}Terminal 4 (Node Logs):${NC}"
    echo "  cd $WORKSPACE_ROOT"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source install/setup.bash"
    echo "  source .venv/bin/activate"
    echo "  ros2 topic echo /rosout | grep audio_playback"
    echo ""
fi
