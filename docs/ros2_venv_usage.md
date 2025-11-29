# ROS2 with Virtual Environment - Usage Guide

## Problem
ROS2 `colcon build` creates executables with `#!/usr/bin/python3` shebang (system Python), but packages are installed in `.venv`. This causes `ModuleNotFoundError` when running nodes.

## Solution
Use the provided launcher scripts that properly combine venv and ROS2 environments.

## Quick Start

### 1. Launch a ROS2 Node
```bash
./launch_node.sh <package_name> <node_name>
```

Example:
```bash
./launch_node.sh audio_interface_nodes audio_capture_node
```

### 2. Source Environment for Manual Commands
```bash
source ros2_venv.sh
ros2 topic list
ros2 topic hz /audio/raw
```

### 3. Run Tests
```bash
source ros2_venv.sh
python manual_tests/test_audio_capture_playback.py
```

## Available Scripts

- **`ros2_venv.sh`** - Sources both venv and ROS2, adds venv packages to PYTHONPATH
- **`launch_node.sh`** - Launches ROS2 Python nodes with venv support
- **`manual_tests/run_audio_capture.sh`** - Convenience wrapper for audio capture node
- **`manual_tests/run_audio_test.sh`** - Convenience wrapper for audio test

## Example Workflow

Terminal 1 - Start audio capture:
```bash
./launch_node.sh audio_interface_nodes audio_capture_node
```

Terminal 2 - Monitor topic:
```bash
source ros2_venv.sh
ros2 topic hz /audio/raw
# Should show ~33 Hz
```

Terminal 3 - Run test:
```bash
source ros2_venv.sh
python manual_tests/test_audio_capture_playback.py
```

## Why Not `ros2 run`?

The `ros2 run` command uses the shebang in the installed executable (`#!/usr/bin/python3`), which points to system Python. Even with venv activated and PYTHONPATH set, the shebang takes precedence.

**Options:**
1. ✅ **Use `launch_node.sh`** - Runs the Python module directly with venv Python
2. ❌ Install packages to system Python - Breaks isolation
3. ❌ Modify shebangs after build - Gets overwritten on rebuild

## Alternative: Launch Files

For complex multi-node scenarios, use ROS2 launch files which can be configured to use venv Python:

```bash
source ros2_venv.sh
ros2 launch <package> <launch_file>
```
