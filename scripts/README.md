# Scripts Directory

This directory contains all development, testing, deployment, and utility scripts for the Local AI Robot Assistant project. Scripts are organized by function for better maintainability.

## Directory Structure

```
scripts/
├── setup/                  # Installation & configuration scripts
├── launch/                 # ROS2 node launcher scripts
├── monitoring/             # System monitoring scripts
├── testing/                # Test scripts organized by domain
│   ├── audio/             # Audio pipeline tests
│   ├── vision/            # Vision pipeline tests
│   └── llm/               # LLM integration tests
├── benchmarking/          # Performance benchmarking tools
├── optimization/          # Model and system optimization
├── utils/                 # General utilities
└── deprecated/            # Archived/obsolete scripts
```

## Setup Scripts (`setup/`)

Scripts for initial installation and environment configuration.

| Script | Purpose | Usage |
|--------|---------|-------|
| `create_repo_structure.sh` | Creates initial directory structure | `./scripts/setup/create_repo_structure.sh` |
| `download_models.sh` | Downloads all AI models (wrapper) | `./scripts/setup/download_models.sh` |
| `download_models.py` | Downloads AI models with validation | `python scripts/setup/download_models.py` |
| `setup_depth.sh` | Validates depth model environment | `./scripts/setup/setup_depth.sh` |
| `setup_pytorch_jetson.sh` | Installs PyTorch for Jetson | `./scripts/setup/setup_pytorch_jetson.sh` |
| `setup_ollama.sh` | Installs and configures Ollama service | `./scripts/setup/setup_ollama.sh` |

**Key Commands:**
```bash
# Download all models (run after setup.sh)
./scripts/setup/download_models.sh

# Verify model downloads
python scripts/setup/download_models.py --verify
```

## Launch Scripts (`launch/`)

Scripts that start ROS2 nodes with proper environment configuration.

| Script | Purpose | Node Launched |
|--------|---------|--------------|
| `run_audio_capture_node.sh` | Launches audio capture with venv | `audio_capture_node` |
| `run_audio_node.sh` | Generic audio node launcher | Any audio node |
| `run_audio_playback_node.sh` | Launches audio playback with venv | `audio_playback_node` |
| `run_integration_tests.sh` | Builds workspace and runs integration tests | N/A |
| `run_perception_node.sh` | Launches perception nodes with venv | Perception nodes |

**Key Commands:**
```bash
# Start audio capture (wake word + transcription)
./scripts/launch/run_audio_capture_node.sh

# Start audio playback (TTS + notifications)
./scripts/launch/run_audio_playback_node.sh

# Run all integration tests
./scripts/launch/run_integration_tests.sh
```

## Monitoring Scripts (`monitoring/`)

Scripts that open tmux sessions for multi-pane system monitoring.

| Script | Purpose | Monitors |
|--------|---------|----------|
| `monitor_audio_full.sh` | Full audio pipeline (capture + playback) | 4 panes: capture node, playback node, events, transcription |
| `monitor_audio_pipeline.sh` | Audio capture only | 3 panes: capture node, events, transcription |
| `monitor_audio_playback.sh` | Audio playback only | 4 panes: playback node, TTS commands, events, logs |
| `check_audio_usage.sh` | Checks which processes use audio devices | Audio device usage |

**Key Commands:**
```bash
# Monitor full audio system
./scripts/monitoring/monitor_audio_full.sh

# Monitor audio capture pipeline
./scripts/monitoring/monitor_audio_pipeline.sh

# Check audio device conflicts
./scripts/monitoring/check_audio_usage.sh
```

**Tmux Controls:**
- `Ctrl+B then D` - Detach from session
- `Ctrl+B then Arrow` - Navigate between panes
- `Ctrl+C` - Stop process in current pane

## Testing Scripts (`testing/`)

### Audio Tests (`testing/audio/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_audio_models.py` | Comprehensive audio model testing | `python scripts/testing/audio/test_audio_models.py` |
| `test_audio_pipeline.sh` | Tests refactored audio_capture_node | `./scripts/testing/audio/test_audio_pipeline.sh` |
| `test_audio_playback_node.py` | Tests TTS & notifications | `python scripts/testing/audio/test_audio_playback_node.py` |
| `test_piper_tts.py` | Tests Piper TTS (quality, latency) | `python scripts/testing/audio/test_piper_tts.py --benchmark` |
| `test_piper_ros2.py` | Tests Piper TTS ROS2 integration | `python scripts/testing/audio/test_piper_ros2.py` |
| `test_wake_word_full.sh` | Complete wake word system test | `./scripts/testing/audio/test_wake_word_full.sh` |
| `test_wakeword_from_microphone.py` | Tests wake word from live mic | `python scripts/testing/audio/test_wakeword_from_microphone.py` |
| `test_arecord.py` | Tests raw arecord audio capture | `python scripts/testing/audio/test_arecord.py` |
| `verify_mic.py` | Verifies microphone capture with resampling | `python scripts/testing/audio/verify_mic.py` |

**Common Audio Tests:**
```bash
# Full wake word system test
./scripts/testing/audio/test_wake_word_full.sh

# Benchmark Piper TTS performance
python scripts/testing/audio/test_piper_tts.py --benchmark --quality

# Test streaming TTS
python scripts/testing/audio/test_piper_tts.py --stream --text "Hello world"
```

### Vision Tests (`testing/vision/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_yolo.py` | Comprehensive YOLO benchmarking | `python scripts/testing/vision/test_yolo.py` |
| `benchmark_depth.py` | Depth Anything V2 TensorRT benchmark | `python scripts/testing/vision/benchmark_depth.py` |
| `test_depth_pipeline.py` | Tests depth estimation pipeline | `python scripts/testing/vision/test_depth_pipeline.py` |
| `benchmark_camera_pipeline.py` | Benchmarks camera pipeline performance | `python scripts/testing/vision/benchmark_camera_pipeline.py` |

**Common Vision Tests:**
```bash
# Benchmark camera pipeline
python scripts/testing/vision/benchmark_camera_pipeline.py

# Test depth estimation
python scripts/testing/vision/test_depth_pipeline.py

# YOLO performance test
python scripts/testing/vision/test_yolo.py
```

### LLM Tests (`testing/llm/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_ollama_moondream.py` | Tests Ollama Moondream VLM | `python scripts/testing/llm/test_ollama_moondream.py` |

### Generic Test Runner (`testing/`)

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_test.py` | Generic test runner with Python path setup | `python scripts/testing/run_test.py <test_module>` |

## Benchmarking Scripts (`benchmarking/`)

Scripts for performance measurement and reporting.

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_performance_report.py` | Aggregates all model benchmarks | `python scripts/benchmarking/generate_performance_report.py` |
| `jetson_audio_optimized.py` | Jetson-optimized audio processing | `python scripts/benchmarking/jetson_audio_optimized.py` |

**Key Commands:**
```bash
# Generate full performance report
python scripts/benchmarking/generate_performance_report.py

# Test optimized audio processing
python scripts/benchmarking/jetson_audio_optimized.py
```

## Optimization Scripts (`optimization/`)

Scripts for optimizing models and system performance.

| Script | Purpose | Usage |
|--------|---------|-------|
| `optimize_audio_models.py` | Optimizes audio models for Jetson | `python scripts/optimization/optimize_audio_models.py` |

**Key Commands:**
```bash
# Optimize wake word and Whisper models
python scripts/optimization/optimize_audio_models.py
```

## Utility Scripts (`utils/`)

General-purpose utilities and helper scripts.

| Script | Purpose | Usage |
|--------|---------|-------|
| `config_demo.py` | Demo of ROS2 configuration utilities | `python scripts/utils/config_demo.py` |
| `generate_dewarp_config.py` | Generates nvdewarper config from calibration | `python scripts/utils/generate_dewarp_config.py` |
| `set_audio_levels.sh` | Sets optimal audio levels | `./scripts/utils/set_audio_levels.sh` |
| `debug_audio_quality.sh` | Records and analyzes audio quality | `./scripts/utils/debug_audio_quality.sh` |

**Key Commands:**
```bash
# Set optimal audio levels
./scripts/utils/set_audio_levels.sh

# Debug audio quality issues
./scripts/utils/debug_audio_quality.sh

# Generate camera dewarp config
python scripts/utils/generate_dewarp_config.py
```

## Common Workflows

### Initial Setup
```bash
# 1. Run main setup script
./setup.sh

# 2. Download AI models
./scripts/setup/download_models.sh

# 3. Set up Ollama (optional for VLM)
./scripts/setup/setup_ollama.sh
```

### Audio Development
```bash
# 1. Test wake word detection
./scripts/testing/audio/test_wake_word_full.sh

# 2. Monitor full audio pipeline
./scripts/monitoring/monitor_audio_full.sh

# 3. Optimize audio models
python scripts/optimization/optimize_audio_models.py
```

### Vision Development
```bash
# 1. Benchmark camera pipeline
python scripts/testing/vision/benchmark_camera_pipeline.py

# 2. Test depth estimation
python scripts/testing/vision/test_depth_pipeline.py

# 3. Generate performance report
python scripts/benchmarking/generate_performance_report.py
```

### Troubleshooting
```bash
# Check audio device usage
./scripts/monitoring/check_audio_usage.sh

# Debug audio quality
./scripts/utils/debug_audio_quality.sh

# Verify microphone
python scripts/testing/audio/verify_mic.py
```

## Environment Requirements

Most scripts require:
- **ROS2 Humble** sourced: `source /opt/ros/humble/setup.bash`
- **Workspace built**: `colcon build`
- **Python venv activated** (for scripts using AI models): `source .venv/bin/activate`

Launcher and monitoring scripts handle environment setup automatically.

## Best Practices

### Running Scripts

1. **Always run from workspace root:**
   ```bash
   cd /home/imanolgo/repos/local-ai-robot-assistant
   ./scripts/<category>/<script>
   ```

2. **Use launcher scripts for ROS2 nodes:**
   ```bash
   # Good
   ./scripts/launch/run_audio_capture_node.sh

   # Avoid (manual env setup)
   ros2 run audio_interface_nodes audio_capture_node
   ```

3. **Use monitoring scripts for debugging:**
   ```bash
   # Good - see all relevant info in one view
   ./scripts/monitoring/monitor_audio_full.sh

   # Avoid - managing multiple terminals manually
   ```

### Adding New Scripts

1. Place in appropriate category directory
2. Make executable: `chmod +x script.sh`
3. Add shebang: `#!/bin/bash` or `#!/usr/bin/env python3`
4. Update this README with script description
5. Document required environment setup
6. Update documentation if script is referenced

### Script Naming Conventions

- **test_*.py/sh** - Testing scripts
- **benchmark_*.py** - Performance benchmarking
- **run_*.sh** - Node launcher scripts
- **monitor_*.sh** - Monitoring scripts
- **setup_*.sh** - Installation/setup scripts
- **optimize_*.py** - Optimization scripts

## Dependencies

### System Tools
- `tmux` - Required for monitoring scripts
- `arecord`/`aplay` - Audio capture/playback
- `colcon` - ROS2 build system

### Python Packages (in .venv)
- `openwakeword` - Wake word detection
- `faster-whisper` - Speech transcription
- `piper-tts` - Text-to-speech
- `sounddevice` - Audio I/O
- `opencv-python` - Vision processing
- `tensorrt` - GPU inference
- `ultralytics` - YOLO models

## Troubleshooting

### Script Not Found
Ensure you're running from workspace root:
```bash
cd /home/imanolgo/repos/local-ai-robot-assistant
./scripts/<category>/<script>
```

### Import Errors
Activate virtual environment:
```bash
source .venv/bin/activate
```

Or use launcher scripts which handle this automatically.

### Audio Device Conflicts
Check for processes using audio:
```bash
./scripts/monitoring/check_audio_usage.sh
```

### ROS2 Node Failures
Check that workspace is built and sourced:
```bash
colcon build
source install/setup.bash
```

## Related Documentation

- [Main README](../README.md) - Project overview
- [Implementation Plan](../docs/implementation_plan.md) - Development roadmap
- [Architecture](../docs/architecture.md) - System design
- [Wake Word Testing Guide](../docs/wake_word_testing_guide.md) - Audio pipeline testing
- [Camera Pipeline](../src/perception_nodes/README.md) - Vision pipeline
- [Model Performance](../docs/model_performance.md) - Benchmark results

## Migration Notes

**2025-12-07**: Scripts reorganized into functional categories:
- Moved testing scripts to `testing/{audio,vision,llm}/`
- Moved launcher scripts to `launch/`
- Moved monitoring scripts to `monitoring/`
- Moved benchmarking scripts to `benchmarking/`
- Moved optimization scripts to `optimization/`
- Moved utilities to `utils/`
- Updated all documentation references

Old paths are deprecated. Update any custom scripts or workflows to use new paths.
