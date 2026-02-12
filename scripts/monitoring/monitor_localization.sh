#!/bin/bash

# Monitoring script for Localization
# This script uses tmux to show multiple panes with relevant data.

SESSION="localization_monitor"
WORKSPACE_ROOT="/home/imanolgo/repos/local-ai-robot-assistant"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if tmux is installed
if command -v tmux >/dev/null 2>&1; then
    # Kill existing session
    tmux kill-session -t $SESSION 2>/dev/null

    # Create new session
    tmux new-session -d -s $SESSION

    # Pane 0 (Top Left): Launch Output
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting Localization...${NC}'" C-m
    tmux send-keys -t $SESSION:0.0 "ros2 launch localization_nodes localization_launch.py" C-m

    # Pane 1 (Top Right): Monitor Output Topic
    tmux split-window -h -t $SESSION:0.0
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}Monitoring /odometry/filtered...${NC}'" C-m
    # Increased sleep to let nodes initialize
    tmux send-keys -t $SESSION:0.1 "sleep 10 && ros2 topic echo /odometry/filtered" C-m

    # Pane 2 (Bottom Left): Input Stats (Herz)
    tmux split-window -v -t $SESSION:0.0
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring Inputs...${NC}'" C-m
    # Added longer sleep and explicit QoS for reliability
    tmux send-keys -t $SESSION:0.2 "sleep 12 && ros2 topic hz /imu/data --qos-reliability best_effort --qos-durability volatile" C-m

    # Pane 3 (Bottom Right): TF Monitor
    tmux split-window -v -t $SESSION:0.1
    tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.3 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring TF (odom -> base_link)...${NC}'" C-m
    # Explicitly check for transform
    tmux send-keys -t $SESSION:0.3 "sleep 15 && ros2 run tf2_ros tf2_echo odom base_link" C-m

    # Attach
    echo -e "${GREEN}Attaching to tmux session...${NC}"
    sleep 1
    tmux attach-session -t $SESSION

else
    echo -e "${YELLOW}tmux not found. Please install tmux.${NC}"
    exit 1
fi
