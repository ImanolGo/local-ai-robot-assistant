#!/usr/bin/env bash
# Full-system integration soak harness (STATUS.md Next Milestone #1).
#
# Runs the system without actuation, a synthetic workload, and tegrastats for a
# fixed duration, then collects logs. Ctrl-C / timeout shuts everything down.
#
# Usage:
#   scripts/testing/integration/run_soak.sh [duration_s] [outdir] [cognitive_delay_s]
#
# Defaults: 3600 s, /tmp/opencode/soak_<timestamp>, 30 s cognitive delay

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HARNESS_DIR/../../.." && pwd)"
DURATION="${1:-3600}"
OUTDIR="${2:-/tmp/opencode/soak_$(date +%Y%m%d_%H%M%S)}"
COGNITIVE_DELAY="${3:-30.0}"
COGNITIVE_BACKEND="${COGNITIVE_BACKEND:-llamacpp}"
LLM_N_GPU_LAYERS="${LLM_N_GPU_LAYERS:--1}"
COGNITIVE_ENABLED="${COGNITIVE_ENABLED:-true}"
mkdir -p "$OUTDIR"

# NOTE: do not use `set -u`; ROS's setup.bash references unbound variables.
cd "$REPO_DIR" || exit 1
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$REPO_DIR/ros2_venv.sh" >/dev/null 2>&1

echo "$DURATION" >"$OUTDIR/duration_s.txt"
echo "$COGNITIVE_DELAY" >"$OUTDIR/cognitive_delay_s.txt"
echo "$COGNITIVE_BACKEND" >"$OUTDIR/cognitive_backend.txt"
echo "$LLM_N_GPU_LAYERS" >"$OUTDIR/llm_n_gpu_layers.txt"
echo "$COGNITIVE_ENABLED" >"$OUTDIR/cognitive_enabled.txt"
date -Is >"$OUTDIR/start_time.txt"

tegrastats --interval 1000 --logfile "$OUTDIR/tegrastats.log" >/dev/null 2>&1 &
TEGRA_PID=$!
python "$HARNESS_DIR/soak_workload.py" >"$OUTDIR/workload.log" 2>&1 &
WORK_PID=$!

# Graceful SIGINT to ros2 launch after DURATION; SIGKILL after a 30 s grace.
timeout --signal=INT --kill-after=30 "$DURATION" \
    ros2 launch launch/full_system_launch.py \
    actuation:=false web_interface:=true "cognitive_start_delay:=$COGNITIVE_DELAY" \
    "cognitive_backend:=$COGNITIVE_BACKEND" "llm_n_gpu_layers:=$LLM_N_GPU_LAYERS" \
    "cognitive:=$COGNITIVE_ENABLED" \
    >"$OUTDIR/launch.log" 2>&1
LAUNCH_RC=$?

kill "$WORK_PID" 2>/dev/null
kill "$TEGRA_PID" 2>/dev/null
wait "$WORK_PID" 2>/dev/null
wait "$TEGRA_PID" 2>/dev/null

free -m >"$OUTDIR/final_free.txt"
date -Is >"$OUTDIR/end_time.txt"
echo "$LAUNCH_RC" >"$OUTDIR/launch_rc.txt"
echo "SOAK_DONE $(date -Is) rc=$LAUNCH_RC" >>"$OUTDIR/status.txt"
