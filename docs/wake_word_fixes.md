# Wake Word Detection System Fixes

## Issues Found and Resolved

### Issue 1: Missing Python Dependencies (openwakeword)

**Problem:**
When running ROS2 nodes via `ros2 run` or direct execution, the nodes couldn't find `openwakeword` and other dependencies installed in `.venv` via `uv`.

**Root Cause:**
- Dependencies were installed in `.venv/lib/python3.10/site-packages`
- ROS2 nodes were using system Python (`/usr/bin/python3`) which doesn't have access to venv packages
- The standalone test script (`test_audio_models.py`) worked because it was likely run with a different Python environment

**Solution:**
Added `PYTHONPATH` export to include venv packages before running ROS2 nodes:
```bash
export PYTHONPATH="$WORKSPACE_DIR/.venv/lib/python3.10/site-packages:$PYTHONPATH"
```

This was applied to:
- `scripts/testing/audio/test_wake_word_full.sh` (updated)
- `scripts/launch/run_perception_node.sh` (already had this pattern)

### Issue 2: Node Initialization Error Handling Bug

**Problem:**
When `wake_word_detector_node` failed to initialize (e.g., missing dependencies), it crashed with:
```
UnboundLocalError: local variable 'node' referenced before assignment
```

**Root Cause:**
The `main()` function tried to call `node.destroy_node()` in the finally block even when node creation failed.

**Solution:**
Modified `wake_word_detector_node.py` to initialize `node = None` and check if it exists before destroying:
```python
node = None
try:
    node = WakeWordDetectorNode()
    rclpy.spin(node)
except Exception as e:
    print(f"Error: {e}")
finally:
    if rclpy.ok():
        if node is not None:  # Check before destroying
            node.destroy_node()
        rclpy.shutdown()
```

### Issue 3: ros2 run Command Not Finding Executables

**Problem:**
`ros2 run audio_interface_nodes <node>` failed with "No executable found" even though executables existed in `install/audio_interface_nodes/bin/`.

**Root Cause:**
Unclear - possibly related to colcon build configuration or ROS2 package indexing issues.

**Solution:**
Changed `test_wake_word_full.sh` to use direct executable paths instead of `ros2 run`:
```bash
# Before:
ros2 run audio_interface_nodes audio_capture_node

# After:
"$WORKSPACE_DIR/install/audio_interface_nodes/bin/audio_capture_node"
```

## System Architecture

### Audio Processing Pipeline

```
Microphone (USB)
    ↓ [hardware: 44100Hz]
audio_capture_node
    ↓ [resamples to 16kHz]
    ↓ [publishes to /audio/raw]
wake_word_detector_node
    ↓ [processes with openWakeWord]
    ↓ [detects "Hey Rover"]
    ↓ [publishes to /audio/wake_word_detected]
Application
```

### Key Configuration

**Audio Config** (`config/audio_config.yaml`):
- Target sample rate: 16000 Hz
- Chunk duration: 80ms (1280 samples at 16kHz)
- Wake word threshold: 0.6
- Cooldown: 2.0 seconds

**Available Wake Words:**
- `hey_rover` (primary)
- `alexa`
- `hey_mycroft`
- `hey_rhasspy`
- `timer`
- `weather`

## Testing

### Full System Test
```bash
./scripts/testing/audio/test_wake_word_full.sh
```

This script:
1. Sources ROS2 environment
2. Configures PYTHONPATH for venv dependencies
3. Sets optimal audio levels
4. Starts audio_capture_node
5. Starts wake_word_detector_node
6. Monitors wake word detection

### Standalone Model Test
```bash
python scripts/testing/audio/test_audio_models.py --test-wake-word
```

Tests the wake word model directly without ROS2.

## Important Notes

### Building with colcon

**Issue:** Building fails when PYTHONPATH includes venv because venv's setuptools doesn't support `--editable` option.

**Solution:** Unset PYTHONPATH before building:
```bash
unset PYTHONPATH
colcon build --packages-select audio_interface_nodes --symlink-install
```

### Running Nodes

**Always** set PYTHONPATH before running nodes:
```bash
export PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages:$PYTHONPATH"
```

Or use the provided helper scripts which handle this automatically.

## Files Modified

1. `scripts/testing/audio/test_wake_word_full.sh` - Added PYTHONPATH config, use direct executables
2. `src/audio_interface_nodes/audio_interface_nodes/wake_word_detector_node.py` - Fixed error handling bug
3. `docs/wake_word_fixes.md` - This documentation

## Verification

After the fixes, the system should:
- ✅ Both nodes start successfully
- ✅ audio_capture_node publishes to `/audio/raw`
- ✅ wake_word_detector_node processes audio chunks
- ✅ Wake word detection works when saying "Hey Rover"

Check logs at `/tmp/wake_word_test_<pid>/` for detailed output.
