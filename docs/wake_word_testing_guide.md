# Wake Word Pipeline Testing Guide

This guide explains how to test the Wake Word Detection pipeline for the Local AI Robot Assistant.

## Prerequisites

- ROS2 installed and sourced.
- `audio_interface_nodes` package built.
- Microphone connected and configured.
- `openwakeword` installed (`pip install openwakeword`).

## Automated Testing

We have provided a comprehensive test script that automates the process of starting the necessary nodes and monitoring the output.

### Running the Full Test

1.  Navigate to the repository root:
    ```bash
    cd /home/imanolgo/repos/local-ai-robot-assistant
    ```

2.  Run the test script:
    ```bash
    ./scripts/test_wake_word_full.sh
    ```

    **What this script does:**
    - Sources the ROS2 environment.
    - Sets optimal audio levels (using `scripts/set_audio_levels.sh`).
    - Starts the `audio_capture_node`.
    - Starts the `wake_word_detector_node`.
    - Checks if nodes started successfully.
    - Runs the `manual_tests/test_wake_word_live.py` monitor.

3.  **Interact with the Robot:**
    - Say "Hey Rover" clearly.
    - You should see "🎤 WAKE WORD DETECTED!" in the output.
    - The script logs to `/tmp/wake_word_test_<PID>/`.

4.  **Stop the Test:**
    - Press `Ctrl+C`. The script will automatically kill the background nodes.

## Manual Testing

If you prefer to run each component manually (e.g., for debugging):

1.  **Terminal 1: Audio Capture**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/.venv/lib/python3.10/site-packages
    ros2 run audio_interface_nodes audio_capture_node
    ```

2.  **Terminal 2: Wake Word Detector**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/.venv/lib/python3.10/site-packages
    ros2 run audio_interface_nodes wake_word_detector_node
    ```

3.  **Terminal 3: Monitor**
    ```bash
    python3 manual_tests/test_wake_word_live.py
    ```
    *Or simply echo the topic:*
    ```bash
    ros2 topic echo /audio/wake_word_detected
    ```

## Troubleshooting

-   **Nodes fail to start:** Check the logs in the temporary directory printed by the script.
-   **No detection:**
    -   Check microphone levels (`alsamixer`).
    -   Verify `audio_capture_node` is publishing to `/audio/raw` (`ros2 topic hz /audio/raw`).
    -   Check `wake_word_detector_node` logs for errors.
-   **False positives:** Increase `confidence_threshold` in `config/audio_config.yaml` or via ROS2 parameters.

## Configuration

Wake word parameters can be adjusted in `config/audio_config.yaml` or passed as ROS2 parameters:
-   `wake_word`: The model to use (default: "hey_rover").
-   `confidence_threshold`: Detection sensitivity (0.0 - 1.0).
-   `cooldown_seconds`: Time to wait after a detection before detecting again.
