# Architecture Context for Copilot

## System Layers (Bottom to Top)

1. **Hardware Layer**: Jetson Orin Nano, Wave Rover, IMX219 camera, USB audio
2. **Driver Layer**: UART, Camera, Audio device drivers
3. **ROS2 Middleware**: Inter-process communication
4. **Perception Layer**: YOLO, Depth, SLAM
5. **Cognitive Layer**: NanoLLM for reasoning
6. **Behavioral Layer**: Behavior Trees for decision making
7. **Application Layer**: Voice commands, navigation, monitoring

## Data Flow Patterns

### Voice Command Flow
```
USB Mic → Wake Word → ASR → LLM → Intent → Behavior Tree → Action
```

### Perception Flow
```
Camera → Undistort → YOLO/Depth → SLAM → World Model → Behavior Tree
```

### Control Flow
```
Behavior Tree → Twist Command → Diff Drive → UART JSON → Wave Rover
```

## Key Integration Points

- **IMU Data**: UART ({"T":126}) → /imu/data topic → EKF fusion
- **Motor Control**: /cmd_vel → Diff Drive Math → UART ({"T":1}) → Motors
- **Object Detection**: Camera → YOLO → /perception/objects → SLAM semantic tags
- **Voice**: Wake Word Trigger → ASR Start → Text → LLM → Response → TTS → Speaker

## Critical Constraints

- Maximum 8GB RAM (7.5GB usable after OS)
- Single GPU (share between YOLO, Depth, Whisper, LLM)
- No wheel encoders (rely on visual odometry)
- Fisheye camera (must undistort before processing)
- UART latency ~10-50ms
- Audio latency budget <2 seconds end-to-end

## State Management

- **Blackboard**: Centralized state for behavior trees
  - Robot pose (x, y, θ)
  - Semantic map (objects with 3D coordinates)
  - Current mission
  - Audio state (listening/processing/speaking)

- **World Model**: Text representation for LLM
  ```
  Robot at (1.2, 0.5, 0°)
  Visible objects:
  - red_ball at (2.0, 1.0, 0.0)
  - blue_cup at (1.5, -0.5, 0.5)
  Current mission: navigate_to_object
  ```

## Resource Budgets

| Component | RAM | GPU | CPU |
|-----------|-----|-----|-----|
| System/ROS2 | 1.5GB | - | 10% |
| RTAB-Map | 1.5GB | 15% | 20% |
| YOLO + Depth | 1.0GB | 40% | 10% |
| Whisper | 0.5GB | 20% | 5% |
| LLM (when active) | 2.5GB | 25% | 30% |
| Buffers/Other | 0.7GB | - | 15% |
| **Total** | **~8.0GB** | **100%** | **90%** |

Note: LLM loaded on-demand for complex reasoning
