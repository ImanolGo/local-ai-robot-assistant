#!/bin/bash

# Monitoring script for SLAM and Localization
# This script uses tmux to show multiple panes with relevant data.

SESSION="slam_monitor"
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

    # Pane 0 (Top Left): Launch everything
    tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.0 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting All Subsystems...${NC}'" C-m
    tmux send-keys -t $SESSION:0.0 "ros2 launch launch/full_system_launch.py" C-m

    # Split horizontally
    tmux split-window -h -t $SESSION:0.0

    # Pane 1 (Top Right): Monitor SLAM Topics (RTAB-Map info)
    tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.1 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}Monitoring SLAM State...${NC}'" C-m
    tmux send-keys -t $SESSION:0.1 "sleep 25 && ros2 topic echo /rtabmap/info" C-m

    # Split vertically
    tmux split-window -v -t $SESSION:0.0
    tmux split-window -v -t $SESSION:0.1

    # Pane 2 (Bottom Left): Input Stats (Rates)
    tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.2 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring Input Rates...${NC}'" C-m
    # Using separate hz commands to avoid argument errors
    tmux send-keys -t $SESSION:0.2 "sleep 20 && watch -n 2 'ros2 topic hz /camera/undistorted --window 10 && ros2 topic hz /imu/data --qos-reliability best_effort --window 10'" C-m

    # Pane 3 (Bottom Right): TF Monitor (map -> odom)
    tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
    tmux send-keys -t $SESSION:0.3 "source ros2_venv.sh" C-m
    tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring SLAM TF (map -> odom)...${NC}'" C-m
    tmux send-keys -t $SESSION:0.3 "sleep 30 && ros2 run tf2_ros tf2_echo map odom" C-m

    # Attach
    echo -e "${GREEN}Attaching to tmux SLAM monitor...${NC}"
    sleep 1
    tmux attach-session -t $SESSION

else
    echo -e "${YELLOW}tmux not found. Please install tmux.${NC}"
    exit 1
fi
