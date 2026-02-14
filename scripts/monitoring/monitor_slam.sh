#!/bin/bash

# Monitoring script for SLAM and Localization (RTAB-Map)
#
# This script uses tmux to show a 2x2 grid monitoring:
#   Top-Left:     Launch perception + actuation + localization + SLAM
#   Top-Right:    RTAB-Map visual odometry output
#   Bottom-Left:  Input topic rates (depth, camera, IMU)
#   Bottom-Right: TF tree monitoring (map → odom → base_link)
#
# Prerequisites:
#   - rtabmap_ros installed (scripts/install_rtabmap.sh)
#   - Camera connected and working
#   - Wave Rover connected via UART

SESSION="slam_monitor"
WORKSPACE_ROOT="/home/imanolgo/repos/local-ai-robot-assistant"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------
echo -e "${GREEN}=== SLAM Monitor Pre-flight Checks ===${NC}"

# Check tmux
if ! command -v tmux >/dev/null 2>&1; then
    echo -e "${RED}tmux not found. Please install: sudo apt install tmux${NC}"
    exit 1
fi

# Check Wave Rover serial connection
if [ -e /dev/ttyTHS1 ]; then
    echo -e "${GREEN}✓ Wave Rover serial port found (/dev/ttyTHS1)${NC}"
else
    echo -e "${YELLOW}⚠ Wave Rover serial port not found (/dev/ttyTHS1)${NC}"
    echo "  IMU data will not be available. Continue? (y/n)"
    read -r response
    [[ "$response" != "y" ]] && exit 1
fi

# Check for stale ROS2 processes
echo "Cleaning up stale ROS2 processes..."
pkill -f "ros2" 2>/dev/null || true
pkill -f "rtabmap" 2>/dev/null || true
sleep 2

# ---------------------------------------------------------------
# Launch tmux session
# ---------------------------------------------------------------

# Kill existing session
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION

# ---------------------------------------------------------------
# Pane 0 (Top Left): Launch the full system
# ---------------------------------------------------------------
tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.0 "source ros2_venv.sh" C-m
tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting Perception + Actuation + Localization + SLAM...${NC}'" C-m
tmux send-keys -t $SESSION:0.0 "ros2 launch launch/full_system_launch.py web_interface:=false" C-m

# Split horizontally
tmux split-window -h -t $SESSION:0.0

# ---------------------------------------------------------------
# Pane 1 (Top Right): Monitor Visual Odometry + SLAM status
# ---------------------------------------------------------------
tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.1 "source ros2_venv.sh" C-m
tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}Waiting for SLAM to start...${NC}'" C-m
tmux send-keys -t $SESSION:0.1 "sleep 25 && echo '--- Visual Odometry ---' && ros2 topic hz /rtabmap/odom --window 10" C-m

# Split vertically
tmux split-window -v -t $SESSION:0.0
tmux split-window -v -t $SESSION:0.1

# ---------------------------------------------------------------
# Pane 2 (Bottom Left): Input topic rates
# ---------------------------------------------------------------
tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.2 "source ros2_venv.sh" C-m
tmux send-keys -t $SESSION:0.2 "echo -e '${YELLOW}Monitoring input topic rates...${NC}'" C-m
tmux send-keys -t $SESSION:0.2 "sleep 20 && watch -n 5 '\
echo \"=== SLAM Input Rates ===\"; \
echo \"--- Depth ---\"; ros2 topic hz /perception/depth --window 5 2>&1 | head -2; \
echo \"--- Camera ---\"; ros2 topic hz /camera/undistorted --window 5 2>&1 | head -2; \
echo \"--- IMU ---\"; ros2 topic hz /imu/data --qos-reliability best_effort --window 5 2>&1 | head -2; \
echo \"--- SLAM Health ---\"; ros2 topic echo /slam/status --once 2>&1 | head -5'" C-m

# ---------------------------------------------------------------
# Pane 3 (Bottom Right): TF tree monitoring
# ---------------------------------------------------------------
tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.3 "source ros2_venv.sh" C-m
tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Monitoring TF tree (map → odom → base_link)...${NC}'" C-m
tmux send-keys -t $SESSION:0.3 "sleep 30 && ros2 run tf2_ros tf2_echo map base_link" C-m

# ---------------------------------------------------------------
# Attach to session
# ---------------------------------------------------------------
echo -e "${GREEN}Attaching to tmux SLAM monitor...${NC}"
echo "  Ctrl+B then D to detach"
echo "  Ctrl+B then arrow keys to switch panes"
sleep 1
tmux attach-session -t $SESSION
