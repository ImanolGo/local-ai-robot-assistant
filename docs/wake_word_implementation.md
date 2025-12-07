# Wake Word Detection Implementation

## Overview

Phase 5.2 implements wake word detection using openWakeWord for the "Hey Rover" wake phrase. The system continuously monitors audio input and triggers when the wake word is detected with sufficient confidence.

## Components

### wake_word_detector_node.py

Main ROS2 node that provides continuous wake word detection:

- **Input**: `/audio/raw` (robot_interfaces/AudioData at 16kHz)
- **Outputs**:
  - `/audio/wake_word_detected` (std_msgs/Bool) - Detection events
  - `/audio/wake_word_confidence` (std_msgs/Float32) - Confidence scores
  - `/audio/events` (robot_interfaces/AudioEvent) - Detailed event information

### Key Features

- **Always-On Operation**: Runs continuously with minimal resource usage
- **Confidence Thresholding**: Configurable threshold (default: 0.6)
- **Cooldown Period**: Prevents multiple triggers (default: 2 seconds)
- **Thread-Safe**: Audio processing in dedicated thread
- **Automatic Model Loading**: Downloads models on first run

## Configuration

Wake word settings in `config/audio_config.yaml`:

```yaml
pipeline:
  wake_word:
    sample_rate: 16000
    channels: 1
    chunk_duration_ms: 80  # 80ms chunks for openWakeWord
    chunk_size: 1280  # 80ms at 16kHz
    confidence_threshold: 0.6  # Minimum confidence (0.0-1.0)
    cooldown_seconds: 2.0  # Prevent multiple triggers
    wake_word_name: "hey_rover"
    model_path: ""  # Empty = use default models
    enable_verbose_logging: false
    target_cpu_usage: 5.0  # Target <5% CPU
```

## Running the Node

### Method 1: Using Helper Script (Recommended)

```bash
# Terminal 1: Start audio capture
./launch_node.sh audio_interface_nodes audio_capture_node

# Terminal 2: Start wake word detector
./launch_node.sh audio_interface_nodes wake_word_detector_node

# Terminal 3: Monitor detections
source ros2_venv.sh
ros2 topic echo /audio/wake_word_detected
```

### Method 2: Manual Launch

```bash
# Source environments
source .venv/bin/activate
source install/setup.bash

# Run node directly
python src/audio_interface_nodes/audio_interface_nodes/wake_word_detector_node.py
```

## Testing

### Unit Tests

```bash
source ros2_venv.sh
pytest tests/test_wake_word.py -v
```

### Live Testing

```bash
# Start all required nodes first
./launch_node.sh audio_interface_nodes audio_capture_node
./launch_node.sh audio_interface_nodes wake_word_detector_node

# Run test script
source ros2_venv.sh
python manual_tests/test_wake_word_live.py
```

## Available Wake Words

openWakeWord comes with several pre-trained models:

- **hey_rover** (primary)
- alexa
- hey_mycroft
- hey_rhasspy
- timer
- weather

You can change the wake word by modifying the `wake_word_name` parameter.

## Performance Targets

- **Detection Latency**: <100ms from word completion
- **CPU Usage**: <5% continuously
- **False Positive Rate**: <1 per hour in quiet environment
- **Detection Accuracy**: >95% in normal conditions

## Current Status

✅ **Implemented**:
- openWakeWord integration
- Real-time audio processing
- Confidence thresholding
- Cooldown period
- ROS2 message publishing
- Unit tests
- Configuration system

⏳ **Pending**:
- Custom "Hey Rover" model training/fine-tuning
- Performance benchmarking on Jetson hardware
- False positive rate testing
- Multi-environment robustness testing
- CPU usage optimization

## Troubleshooting

### Model Download Issues

If models fail to download automatically:

```bash
source .venv/bin/activate
python -c "from openwakeword.model import Model; Model()"
```

### High CPU Usage

- Reduce `chunk_size` (but may affect accuracy)
- Disable verbose logging
- Check for other CPU-intensive processes

### Low Detection Rate

- Increase microphone volume
- Move closer to microphone
- Reduce `confidence_threshold` (but may increase false positives)
- Check audio_capture_node is running and publishing

### No Detections

Verify the pipeline:

```bash
# Check audio is being published
source ros2_venv.sh
ros2 topic hz /audio/raw

# Check wake word node is running
ros2 node list | grep wake_word

# Monitor confidence scores
ros2 topic echo /audio/wake_word_confidence
```

## Next Steps

Phase 5.3 will implement Voice Activity Detection (VAD) to intelligently capture complete utterances after wake word detection.

## References

- [openWakeWord Documentation](https://github.com/dscripka/openWakeWord)
- [Implementation Plan - Phase 5.2](../docs/implementation_plan.md#52-wake-word-detection-hey-rover)
- [Audio Infrastructure Documentation](../docs/audio_infrastructure_implementation.md)
