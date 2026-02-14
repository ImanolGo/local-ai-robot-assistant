#!/bin/bash

# Monitoring script for Localization
# This script uses tmux to show multiple panes with relevant data.
#
# The EKF node requires /imu/data published by uart_motor_controller.
# This script launches both actuation and localization nodes together.
#
# Layout:
#   ┌──────────────────┬──────────────────┐
#   │  Actuation Node  │  EKF Launch      │
#   ├──────────────────┼──────────────────┤
#   │  IMU Hz Monitor  │  Odom Echo       │
#   └──────────────────┴──────────────────┘

SESSION="localization_monitor"
WORKSPACE_ROOT="/home/imanolgo/repos/local-ai-robot-assistant"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if tmux is installed
if ! command -v tmux >/dev/null 2>&1; then
    echo -e "${RED}tmux not found. Please install tmux: sudo apt install tmux${NC}"
    exit 1
fi

# --- Cleanup stale processes from previous runs ---
echo -e "${YELLOW}Cleaning up stale ROS2 processes...${NC}"
tmux kill-session -t $SESSION 2>/dev/null || true
sleep 1

# Kill any lingering ROS2 nodes from previous monitoring sessions
pkill -f "uart_motor_controller" 2>/dev/null || true
pkill -f "ekf_node" 2>/dev/null || true
pkill -f "static_transform_publisher.*base_link.*imu_link" 2>/dev/null || true
echo -e "${YELLOW}Waiting for processes to terminate...${NC}"
sleep 3

# Force-kill if SIGTERM wasn't enough
pkill -9 -f "uart_motor_controller" 2>/dev/null || true
pkill -9 -f "ekf_node" 2>/dev/null || true
sleep 1

# Verify serial port is available
if [ ! -e /dev/ttyTHS1 ]; then
    echo -e "${RED}/dev/ttyTHS1 not found. Is the Wave Rover connected?${NC}"
    exit 1
fi

# Wait for serial port to be released (retry loop)
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
    if ! lsof /dev/ttyTHS1 >/dev/null 2>&1; then
        echo -e "${GREEN}/dev/ttyTHS1 is available.${NC}"
        break
    fi
    echo -e "${YELLOW}/dev/ttyTHS1 still in use (attempt $i/$MAX_RETRIES). Forcing cleanup...${NC}"
    fuser -k /dev/ttyTHS1 2>/dev/null || true
    sleep 2
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo -e "${RED}/dev/ttyTHS1 could not be freed after $MAX_RETRIES attempts. Aborting.${NC}"
        exit 1
    fi
done

# --- Pre-flight: verify Wave Rover is responding ---
echo -e "${YELLOW}Testing Wave Rover serial communication...${NC}"
UART_RESPONSE=$(python3 -c "
import serial, json, time, sys
try:
    ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
    ser.setRTS(False)
    ser.setDTR(False)
    ser.flushInput()
    ser.flushOutput()
    time.sleep(0.3)
    ser.write((json.dumps({'T': 126}) + '\\n').encode())
    ser.flush()
    start = time.time()
    while time.time() - start < 3:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if '{' in line:
                print('OK')
                ser.close()
                sys.exit(0)
        time.sleep(0.01)
    print('NO_RESPONSE')
    ser.close()
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

if [[ "$UART_RESPONSE" == "OK" ]]; then
    echo -e "${GREEN}Wave Rover is responding on /dev/ttyTHS1 ✓${NC}"
elif [[ "$UART_RESPONSE" == "NO_RESPONSE" ]]; then
    echo -e "${RED}⚠ Wave Rover is NOT responding on /dev/ttyTHS1!${NC}"
    echo -e "${RED}  - Is the Wave Rover powered on?${NC}"
    echo -e "${RED}  - Check the UART cable connection.${NC}"
    echo -e "${YELLOW}Continuing anyway (nodes will retry)...${NC}"
    sleep 2
else
    echo -e "${RED}⚠ Serial test error: $UART_RESPONSE${NC}"
    echo -e "${YELLOW}Continuing anyway...${NC}"
    sleep 1
fi

# --- Create tmux session with 2x2 grid ---
tmux new-session -d -s $SESSION -n "Localization"

# Create 2x2 layout reliably:
# Step 1: Split horizontally -> left (pane 0) and right (pane 1)
tmux split-window -h -t $SESSION:0.0
# Step 2: Split left pane vertically -> top-left and bottom-left
# After this: pane 0=top-left, pane 1=bottom-left, pane 2=right
tmux split-window -v -t $SESSION:0.0
# Step 3: Split right pane (now index 2) vertically -> top-right and bottom-right
# After this: pane 0=top-left, pane 1=bottom-left, pane 2=top-right, pane 3=bottom-right
tmux split-window -v -t $SESSION:0.2
sleep 0.3

# Verify we have 4 panes
PANE_COUNT=$(tmux list-panes -t $SESSION 2>/dev/null | wc -l)
if [ "$PANE_COUNT" -ne 4 ]; then
    echo -e "${RED}Error: Expected 4 panes but got ${PANE_COUNT}. Terminal may be too small.${NC}"
    echo -e "${YELLOW}Try making your terminal larger and re-running.${NC}"
    tmux kill-session -t $SESSION 2>/dev/null || true
    exit 1
fi

# Pane 0 (Top Left): Actuation Launch (motor controller = IMU data source)
# Chain source + launch so a source failure prevents a broken launch
tmux send-keys -t $SESSION:0.0 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.0 "echo -e '${GREEN}Starting Actuation (motor controller + IMU)...${NC}'" C-m
tmux send-keys -t $SESSION:0.0 "source ros2_venv.sh && ros2 launch actuation_nodes actuation_launch.py" C-m

# Pane 2 (Top Right): Localization Launch (EKF + static TF)
tmux send-keys -t $SESSION:0.2 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.2 "echo -e '${GREEN}Waiting 5s for motor controller, then starting EKF...${NC}'" C-m
tmux send-keys -t $SESSION:0.2 "source ros2_venv.sh && sleep 5 && ros2 launch localization_nodes localization_launch.py" C-m

# Pane 1 (Bottom Left): Monitor /imu/data rate
# NOTE: ros2 topic hz in Humble has NO QoS options and defaults to RELIABLE,
# which cannot subscribe to BEST_EFFORT publishers like /imu/data.
# Use 'ros2 topic echo' with sensor_data QoS and count messages instead.
tmux send-keys -t $SESSION:0.1 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.1 "echo -e '${YELLOW}Waiting 10s then monitoring /imu/data...${NC}'" C-m
tmux send-keys -t $SESSION:0.1 "source ros2_venv.sh && sleep 10 && ros2 topic echo /imu/data --qos-profile sensor_data --field header.stamp" C-m

# Pane 3 (Bottom Right): Odometry output
tmux send-keys -t $SESSION:0.3 "cd $WORKSPACE_ROOT" C-m
tmux send-keys -t $SESSION:0.3 "echo -e '${YELLOW}Waiting 15s then monitoring /odometry/filtered...${NC}'" C-m
tmux send-keys -t $SESSION:0.3 "source ros2_venv.sh && sleep 15 && ros2 topic echo /odometry/filtered --qos-profile sensor_data" C-m

# Attach
echo -e "${GREEN}Attaching to tmux session...${NC}"
echo -e "${YELLOW}Layout:${NC}"
echo -e "  ┌──────────────────┬──────────────────┐"
echo -e "  │  Actuation Node  │  EKF Launch      │"
echo -e "  ├──────────────────┼──────────────────┤"
echo -e "  │  IMU Hz Monitor  │  Odom Echo       │"
echo -e "  └──────────────────┴──────────────────┘"
echo ""
echo -e "${YELLOW}Controls:${NC}"
echo -e "  Ctrl+B then D     - Detach from session"
echo -e "  Ctrl+B then Arrow - Navigate between panes"
echo -e "  Ctrl+C            - Stop process in current pane"
echo -e "  tmux kill-session -t $SESSION  - Stop all nodes"
echo ""
sleep 2
tmux attach-session -t $SESSION
