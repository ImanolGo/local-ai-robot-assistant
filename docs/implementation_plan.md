# Implementation Plan & Development Checklist
## Local AI Robot Assistant Project - Complete Edition

**Project Timeline**: 16-18 weeks
**Team Size**: 1-3 developers
**Methodology**: Agile/Iterative with hardware-in-the-loop testing

---

## Phase 0: Project Setup & Environment (Week 1)

### 0.1 Repository Setup
- [x] Create GitHub repository with proper .gitignore
- [x] Initialize with README.md, LICENSE
- [x] Set up branch protection rules (main/develop)
- [x] Create GitHub Issues templates (bug, feature, hardware)
- [x] Set up GitHub Projects board with kanban workflow
- [x] Create initial repository structure

### 0.2 Development Environment
- [x] Flash Jetson Orin Nano with JetPack SDK (5.x or 6.x)
- [x] Install Ubuntu and configure headless mode
- [x] Install ROS2 Humble and verify installation
- [x] Set up SSH access and configure networking
- [x] Install development tools (git, vim/nano, tmux)
- [x] Configure NVMe SSD and create large swap file (16GB)
- [x] Install Docker and set up containerization
- [x] Set up VS Code remote development or similar
- [x] See `docs/guides/jetson_orin_setup.md` for detailed setup instructions

### 0.3 Hardware Inventory & Testing
- [x] Verify all hardware components received
- [x] Label and document all cables and connections
- [x] Create hardware connection diagram
- [x] Set up proper workspace with ESD protection
- [x] Organize storage for components

### 0.4 Documentation Setup
- [x] Create `/docs` directory structure
- [x] Initialize Sphinx or MkDocs for documentation
- [x] Set up automated doc generation pipeline
- [x] Create hardware setup guide
- [x] Create development workflow guide

### 0.5 Model Conversion Tools Setup
- [x] Install TensorRT and trtexec
- [x] Install ONNX runtime with CUDA support
- [x] Create `tools/` directory for conversion scripts
- [x] Test TensorRT installation with sample model
- [x] Create model conversion pipeline template
- [x] Document conversion best practices

**Deliverables**:
- GitHub repo initialized
- Jetson fully configured with 16GB swap
- Hardware inventory documented
- Model conversion tools ready

---

## Phase 1: Hardware Validation (Week 2)

### 1.1 Wave Rover UART Communication
- [x] Connect Wave Rover to Jetson via UART
- [x] Identify correct serial port (`/dev/ttyTHS0` or `/dev/ttyUSB0`)
- [x] Write test script to send/receive JSON commands
- [x] Test motor control commands (forward, backward, turn)
- [x] Test IMU data retrieval (`{"T":126}`)
- [x] Test continuous feedback mode (`{"T":131,"cmd":1}`)
- [x] Test OLED display commands
- [x] Document baud rate (115200) and communication protocol
- [x] Create `hardware_tests/test_waveroever_uart.py`

**Test Script Requirements**:
# ✅ COMPLETED: Full thermal/power validation documented
```python
# hardware_tests/test_waveroever_uart.py
- Test connection establishment
- Test JSON parsing of responses
- Test all motor control commands
- Test IMU data format
- Test communication error handling
- Benchmark communication latency (target: <10ms)
- Test at 50 Hz polling rate for IMU
```

### 1.2 Camera Validation
- [x] Connect IMX219 camera to MIPI CSI-2 port
- [x] Verify camera detection (`ls /dev/video*`)
- [x] Test camera capture with `nvgstcapture-1.0`
- [x] Capture test images and verify resolution using DeepStream
- [x] Test different resolutions and frame rates with hardware acceleration
- [x] Document optimal camera settings for DeepStream pipeline
- [x] Create `hardware_tests/test_camera_capture.py` (DeepStream-based)

**Test Script Requirements**:
```python
# hardware_tests/test_camera_capture.py
- Test DeepStream pipeline initialization
- Test hardware-accelerated frame capture at various resolutions
- Test frame rate measurement with NVMM memory
- Save sample images for validation
- Test continuous capture for 5 minutes
- Benchmark GPU memory usage and performance
- Test CSI camera with nvarguscamerasrc element
- Target: 30 FPS at 640x480, 20 FPS at 1920x1080
```

### 1.3 Camera Calibration (Critical)
- [x] Print checkerboard calibration pattern (9x6, 25mm squares)
- [x] Capture 20-30 calibration images at various angles
- [x] Run OpenCV calibration script
- [x] Generate camera intrinsics matrix
- [x] Generate fisheye distortion coefficients
- [x] Validate calibration with test images
- [x] Save calibration to `config/camera_calibration.yaml`
- [x] Create `hardware_tests/calibrate_camera.py`
- [x] Create `hardware_tests/test_undistortion.py`

**Test Script Requirements**:
```python
# hardware_tests/calibrate_camera.py
- Automated checkerboard detection
- Fisheye calibration optimization
- Reprojection error calculation (target: <0.5 pixels)
- YAML export of parameters

# hardware_tests/test_undistortion.py
- Load calibration parameters
- Apply undistortion to test images
- Visual comparison tool
- Measure improvement in line straightness
```

### 1.4 USB Audio Validation
- [x] Connect USB microphone
- [x] Verify microphone detection (`arecord -l`)
- [x] Test audio recording with `arecord`
- [x] Measure microphone noise floor
- [x] Test various sample rates (16kHz, 44.1kHz)
- [x] Connect USB speakers
- [x] Verify speaker detection (`aplay -l`)
- [x] Test audio playback with `aplay`
- [x] Test speaker volume range
- [x] Document optimal audio device settings
- [x] Create `hardware_tests/test_audio_devices.py`

**Test Script Requirements**:
```python
# hardware_tests/test_audio_devices.py
- Detect and list audio devices
- Test microphone recording (5 seconds)
- Test speaker playback
- Test simultaneous record/playback
- Measure audio latency (target: <200ms round-trip)
- Test noise levels and SNR
- Test USB device reconnection
```

### 1.5 Power & Thermal Testing
- [x] Test Jetson power consumption under idle
- [x] Test power consumption under full load
- [x] Monitor temperature during extended operation
- [x] Test thermal throttling behavior
- [x] Verify cooling solution adequacy
- [x] Create `hardware_tests/test_thermal_power.py`
- [x] Document thermal/power validation results: `thermal_power_validation_report.md`

**Test Script Requirements**:
```python
# hardware_tests/test_thermal_power.py
- Monitor CPU/GPU temperatures (target: <80°C sustained)
- Test thermal throttling threshold
- Measure power consumption (idle vs. load)
- Test for 30 minutes under full load
- Log thermal zones and fan speed
```

**Deliverables**:
- All hardware validated and tested
- Test scripts for each component with passing tests
- Hardware test results documented in `docs/hardware_validation_report.md`
- Camera calibration file generated and validated

---

## Phase 2: Core Infrastructure (Weeks 3-4)

### 2.1 ROS2 Workspace Setup
- [x] Create ROS2 workspace structure (`ros2_ws/src/`)
- [x] Create all package directories
- [x] Set up package.xml files for each package
- [x] Set up CMakeLists.txt or setup.py files
- [x] Create custom message definitions
- [x] Build workspace and verify (`colcon build`)
- [x] Set up colcon build configuration
- [x] Create launch file directory structure

### 2.2 UART Communication Nodes
- [x] Implement `uart_motor_controller.py`
  - [x] Serial port initialization with error handling
  - [x] JSON command builder for Wave Rover protocol
  - [x] JSON response parser with validation
  - [x] ROS2 node structure
  - [x] Subscribe to `/cmd_vel` topic (geometry_msgs/Twist)
  - [x] Publish to `/motor_status` topic
  - [x] Implement differential drive kinematics
  - [x] Add watchdog timer (500ms timeout)
  - [x] Add emergency stop service (`/emergency_stop`)
  - [x] Command rate limiting (20 Hz)
  - [x] Publish dead reckoning odometry to `/odom_raw` (high covariance)
- [x] Implement `uart_imu_node.py`
  - [x] Periodic IMU query at 50 Hz (increased from 20 Hz)
  - [x] JSON response parser with validation
  - [x] Publish to `/imu/data` topic (sensor_msgs/Imu)
  - [x] Data validation and outlier rejection
  - [x] Error handling and automatic reconnection
- [x] Create unit tests for both nodes
- [x] Create integration test for UART package
- [x] Document UART protocol in package README

**Tests Required**:
```python
# tests/test_uart_motor_controller.py
- Test JSON command generation
- Test velocity to wheel speed conversion
- Test watchdog timer functionality
- Test emergency stop service
- Mock serial communication
- Test command rate limiting

# tests/test_uart_imu_node.py
- Test IMU data parsing at 50 Hz
- Test ROS message conversion
- Test error handling and reconnection
- Validate data ranges and units
```

### 2.3 Camera Pipeline (DeepStream-Accelerated)
- [x] Implement `camera_driver.py`
  - [x] DeepStream pipeline setup with nvarguscamerasrc
  - [x] ROS2 node structure
  - [x] Publish raw images to `/camera/raw` (using NVMM buffers)
  - [x] Implement hardware-accelerated frame rate control
  - [x] Add camera info publisher with calibration data
  - [x] Utilize GPU memory for zero-copy operations
  - [x] Target: 30 FPS at 640x480
- [x] Implement `image_undistort_node.py`
  - [x] Load calibration from YAML
  - [x] Subscribe to `/camera/raw`
  - [x] Apply GPU-accelerated undistortion with DeepStream
  - [x] Publish to `/camera/undistorted`
  - [x] Add performance monitoring for GPU usage
  - [x] Latency target: <20ms per frame
- [x] Create unit tests
- [x] Create integration test
- [x] Benchmark processing latency and GPU memory usage

**Tests Required**:
```python
# tests/test_camera_driver.py
- Test DeepStream pipeline initialization
- Test hardware-accelerated frame publishing
- Test frame rate consistency with NVMM memory
- Test GPU memory usage optimization
- Test reconnection after camera disconnect

# tests/test_image_undistort.py
- Test calibration loading
- Test GPU-accelerated undistortion algorithm
- Test output image quality with DeepStream
- Benchmark performance vs CPU implementation (target: 3x faster)
```

### 2.4 Configuration Management
- [x] Create `config/uart_config.yaml`
  - Port, baud rate, timeouts, command rates
- [x] Create `config/camera_config.yaml`
  - Resolution, frame rate, calibration file path
- [x] Create `config/audio_config.yaml`
  - Device indices, sample rates, buffer sizes
- [x] Create parameter loading utilities
- [x] Test configuration validation
- [x] Create `config/memory_management.yaml` (NEW)
  - RAM thresholds for model loading/unloading

**Deliverables**:
- Working UART communication at 20 Hz (motors) and 50 Hz (IMU)
- Working camera pipeline with undistortion
- Unit tests for all nodes (>80% coverage)
- Configuration files documented

---

## Phase 3: Model Conversion & Optimization (Week 5)

### 3.1 Model Acquisition
- [ ] Download YOLOv8n PyTorch model from Ultralytics
- [ ] Download FastDepth PyTorch model from MIT
- [ ] Download Whisper Tiny model from OpenAI
- [ ] Download Piper TTS model and voice files
- [ ] Download openWakeWord models
- [ ] Download LLaMA-2 7B or Gemma 2-7B or Phi-2
- [ ] Organize models in `/models` directory
- [ ] Document model sources and licenses in `docs/model_credits.md`

### 3.2 Vision Model Conversion (TensorRT)
- [ ] Implement `tools/convert_yolo.py`
  - [ ] Export YOLOv8n to ONNX format
  - [ ] Convert ONNX to TensorRT engine (FP16)
  - [ ] Validate output accuracy (mAP drop <2%)
  - [ ] Benchmark inference time (target: <50ms on Jetson)
  - [ ] Save engine to `models/yolo_trt/yolov8n_fp16.engine`
- [ ] Implement `tools/convert_depth.py`
  - [ ] Export FastDepth to ONNX format
  - [ ] Convert ONNX to TensorRT engine (FP16)
  - [ ] Validate depth map quality
  - [ ] Benchmark inference time (target: <70ms)
  - [ ] Save engine to `models/depth_trt/fastdepth_fp16.engine`
- [ ] Document conversion process in `docs/guides/model_conversion.md`

### 3.3 Audio Model Setup
- [ ] Set up openWakeWord
  - [ ] Download pre-trained models
  - [ ] Test wake word detection accuracy
  - [ ] Optimize for <5% CPU usage
- [ ] Set up faster-whisper (PRIMARY OPTION)
  - [ ] Install faster-whisper library (CTranslate2)
  - [ ] Download Whisper Tiny model (faster-whisper format)
  - [ ] Test inference speed (target: real-time factor <0.3x)
  - [ ] Benchmark RAM usage (<300 MB)
- [ ] ALTERNATIVE: Implement `tools/convert_whisper.py`
  - [ ] Export Whisper Tiny to ONNX
  - [ ] Convert to TensorRT (FP16)
  - [ ] Validate Word Error Rate (WER)
  - [ ] Compare performance with faster-whisper
- [ ] Set up Piper TTS
  - [ ] Download Piper binary and voice files
  - [ ] Test synthesis quality
  - [ ] Benchmark latency (target: <500ms for 20 words)

### 3.4 LLM Setup (NanoLLM)
- [ ] Install NVIDIA NanoLLM
- [ ] Download and quantize LLM:
  - [ ] Option 1: LLaMA-2 7B (INT4 via AWQ)
  - [ ] Option 2: Gemma 2-7B (INT4)
  - [ ] Option 3: Phi-2 (INT8, smaller footprint)
- [ ] Test inference speed (target: <3s for typical response)
- [ ] Test RAM usage (target: <2.5 GB)
- [ ] Implement lazy loading mechanism
- [ ] Save quantized model to `models/nanollm_quantized/`

### 3.5 Model Profiling
- [ ] Implement `tools/profile_model.py`
  - [ ] Measure inference time for each model
  - [ ] Measure RAM/VRAM usage
  - [ ] Measure GPU utilization
  - [ ] Generate performance report
- [ ] Run profiling for all models
- [ ] Document results in `docs/model_performance.md`

**Deliverables**:
- All models converted to optimal formats (TensorRT for vision)
- Model performance benchmarks documented
- Conversion scripts tested and documented
- Model registry with metadata

---

## Phase 4: Perception Models Integration (Week 6)

### 4.1 YOLO Object Detection Node
- [ ] Implement `object_detector.py`
  - [ ] Load TensorRT engine from Phase 3.2
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Run inference using TensorRT runtime
  - [ ] Publish detections to `/perception/objects`
  - [ ] Add visualization overlay
  - [ ] Implement confidence thresholding (default: 0.5)
  - [ ] Add NMS (Non-Maximum Suppression)
- [ ] Benchmark inference time (target: 20+ FPS)
- [ ] Test on various objects
- [ ] Create unit tests
- [ ] Document supported object classes

**Tests Required**:
```python
# tests/test_object_detector.py
- Test TensorRT engine loading
- Test inference on sample images
- Test detection accuracy on validation set
- Benchmark FPS (target: 20+)
- Test bounding box format and coordinates

# manual_tests/test_yolo_realtime.py
- Live camera feed test
- Visual validation of detections
- FPS monitoring over 5 minutes
```

### 4.2 Depth Estimation Node
- [ ] Implement `depth_estimator.py`
  - [ ] Load TensorRT engine from Phase 3.2
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Run inference using TensorRT runtime
  - [ ] Publish depth maps to `/perception/depth`
  - [ ] Add depth colormap visualization
  - [ ] Implement depth range normalization
- [ ] Benchmark inference time (target: 15+ FPS)
- [ ] Test depth accuracy with known distances
- [ ] Create unit tests

**Tests Required**:
```python
# tests/test_depth_estimator.py
- Test TensorRT engine loading
- Test inference on sample images
- Test depth range validation (0.5m - 10m)
- Benchmark FPS (target: 15+)

# manual_tests/test_depth_accuracy.py
- Compare estimated depth with measured depth
- Test at various distances (0.5m, 1m, 2m, 5m)
- Visualize depth maps with colormap
- Calculate mean absolute error (target: <15%)
```

### 4.3 Point Cloud Generation
- [ ] Implement `pointcloud_generator.py`
  - [ ] Subscribe to `/perception/depth` and `/camera/undistorted`
  - [ ] Load camera intrinsics from calibration file
  - [ ] Back-project depth map to 3D points
  - [ ] Apply coordinate transformations
  - [ ] Publish to `/perception/pointcloud` (sensor_msgs/PointCloud2)
  - [ ] Add RGB color mapping
- [ ] Test point cloud accuracy with known geometry
- [ ] Visualize in RViz2
- [ ] Optimize for real-time performance (target: 10+ Hz)

### 4.4 Perception Integration Test
- [ ] Create `launch/perception_launch.py`
- [ ] Test complete perception pipeline:
  - Camera → Undistortion → YOLO + Depth → Point Cloud
- [ ] Measure end-to-end latency (target: <200ms)
- [ ] Profile GPU/CPU usage
- [ ] Test for memory leaks (24-hour test)

**Deliverables**:
- Working object detection node (20+ FPS)
- Working depth estimation node (15+ FPS)
- Working point cloud generation
- Perception pipeline integrated and tested
- Performance benchmarks documented

---

## Phase 5: Audio Pipeline (Week 7)

### 5.1 Audio Capture & Playback
- [ ] Implement `audio_capture_node.py`
  - [ ] PyAudio initialization
  - [ ] Configure USB microphone from config
  - [ ] Continuous audio streaming at 16 kHz
  - [ ] Publish to `/audio/raw` topic (audio_msgs/AudioData)
  - [ ] Add circular buffer management (5-second buffer)
  - [ ] Implement USB device health monitoring
- [ ] Implement `audio_playback_node.py`
  - [ ] PyAudio initialization
  - [ ] Configure USB speakers from config
  - [ ] Subscribe to `/audio/tts_output`
  - [ ] Queue-based playback system
  - [ ] Handle playback interruptions (emergency stop)
  - [ ] Monitor playback errors
- [ ] Test audio latency (target: <200ms round-trip)
- [ ] Test simultaneous capture/playback
- [ ] Test USB device reconnection

**Tests Required**:
```python
# tests/test_audio_capture.py
- Test device initialization
- Test audio streaming at 16 kHz
- Test buffer management
- Test device disconnection handling

# tests/test_audio_playback.py
- Test device initialization
- Test audio playback
- Test queue management
- Test interruption handling
```

### 5.2 Wake Word Detection
- [ ] Install openWakeWord library
- [ ] Implement `wake_word_detector_node.py`
  - [ ] Load wake word model (ONNX)
  - [ ] Subscribe to `/audio/raw`
  - [ ] Run continuous detection (separate thread)
  - [ ] Publish to `/audio/wake_word_detected` (std_msgs/Bool)
  - [ ] Add detection confidence threshold (default: 0.5)
  - [ ] Implement cooldown period (2 seconds)
- [ ] Train or configure custom wake word
- [ ] Test false positive rate (target: <1 per hour)
- [ ] Test detection latency (target: <100ms)
- [ ] Optimize for low CPU usage (target: <5%)

**Tests Required**:
```python
# tests/test_wake_word.py
- Test model loading
- Test detection on sample audio
- Benchmark CPU usage (target: <5%)
- Measure detection latency

# manual_tests/test_wake_word_accuracy.py
- Test with multiple speakers (5+ people)
- Test in various noise levels (quiet, moderate, noisy)
- Measure false positive/negative rates
- Test in different room acoustics
```

### 5.3 Speech-to-Text (Whisper)
- [ ] Set up faster-whisper (PRIMARY)
  - [ ] Install faster-whisper library
  - [ ] Test model loading time
- [ ] Implement VAD (Voice Activity Detection)
  - [ ] Use webrtcvad or silero-vad
  - [ ] Configure silence threshold
- [ ] Implement `stt_node.py`
  - [ ] Load Whisper model (faster-whisper or TensorRT)
  - [ ] Subscribe to `/audio/wake_word_detected`
  - [ ] Capture audio segment with VAD (max 10 seconds)
  - [ ] Run transcription
  - [ ] Publish to `/audio/transcribed_text` (std_msgs/String)
  - [ ] Add timeout handling (15 seconds max)
  - [ ] Implement noise suppression (optional)
- [ ] Test transcription accuracy (target: WER <10% for clean speech)
- [ ] Test with various accents/speaking styles
- [ ] Benchmark inference time (target: <2s for 5s audio)
- [ ] Optimize for low latency

**Tests Required**:
```python
# tests/test_stt.py
- Test model loading (faster-whisper or TensorRT)
- Test transcription on sample audio (10+ samples)
- Test VAD functionality
- Benchmark latency (target: real-time factor <0.4x)

# manual_tests/test_stt_accuracy.py
- Test with various commands (50+ utterances)
- Test in noisy environments (SNR: 20dB, 10dB, 5dB)
- Calculate Word Error Rate (WER)
- Test with different speakers (5+ people)
```

### 5.4 Text-to-Speech (Piper)
- [ ] Install Piper TTS
- [ ] Download and test voice model
- [ ] Implement `tts_node.py`
  - [ ] Load Piper model
  - [ ] Subscribe to `/audio/tts_request` (std_msgs/String)
  - [ ] Synthesize speech (ONNX inference)
  - [ ] Publish to `/audio/tts_output` (audio_msgs/AudioData)
  - [ ] Add speech rate control (parameter)
  - [ ] Add volume normalization
- [ ] Test voice quality (subjective evaluation)
- [ ] Test synthesis latency (target: <500ms for 20 words)
- [ ] Test various sentence lengths and structures
- [ ] Optimize for low latency

**Tests Required**:
```python
# tests/test_tts.py
- Test model loading
- Test synthesis on sample text (10+ sentences)
- Benchmark latency (target: <500ms for typical sentence)
- Test voice quality metrics (MOS if available)

# manual_tests/test_tts_naturalness.py
- Test various sentence types (questions, commands, statements)
- Test punctuation handling (commas, periods, exclamations)
- User feedback on naturalness (5+ participants)
- Test long text synthesis (100+ words)
```

### 5.5 Audio Pipeline Integration
- [ ] Create `launch/audio_pipeline_launch.py`
- [ ] Test complete audio flow:
  - Mic → Wake Word → ASR → Transcribed Text
  - TTS Request → TTS → Audio → Speaker
- [ ] Measure end-to-end latency (target: <4 seconds wake-to-response)
- [ ] Test error recovery (USB disconnect, model failure)
- [ ] Optimize resource usage
- [ ] Test 1-hour continuous operation

**Deliverables**:
- Working audio pipeline (capture to playback)
- Wake word detection (<5% CPU)
- Speech-to-text (WER <10%)
- Text-to-speech (natural voice)
- Audio tests passing
- Latency benchmarks documented

---

## Phase 6: SLAM & Localization (Week 8)

### 6.1 Robot Localization Setup (EKF Fusion)
- [ ] Install `robot_localization` package
- [ ] Create `config/localization_config.yaml`
- [ ] Configure EKF parameters:
  - [ ] IMU sensor configuration (50 Hz input)
  - [ ] Visual odometry configuration (10-30 Hz)
  - [ ] Dead reckoning configuration (20 Hz, high covariance)
  - [ ] Process noise and measurement noise
- [ ] Set up sensor inputs:
  - `/imu/data` (orientation, angular velocity)
  - `/rtabmap/odom` (visual odometry)
  - `/odom_raw` (dead reckoning backup)
- [ ] Test odometry fusion with simulated data
- [ ] Tune EKF parameters for smooth output
- [ ] Create launch file (`launch/localization_launch.py`)

**Tests Required**:
```python
# tests/test_localization.py
- Test EKF initialization
- Test sensor fusion accuracy (vs ground truth)
- Test with simulated IMU + odometry data
- Benchmark performance (target: <1ms update time)
- Test covariance estimation

# manual_tests/test_ekf_tuning.py
- Drive robot in square pattern
- Compare estimated pose with measured pose
- Tune process and measurement noise
- Test IMU-only mode (visual odometry failure)
```

### 6.2 RTAB-Map SLAM Setup
- [ ] Install `rtabmap_ros` package
- [ ] Create `config/rtabmap_config.yaml`
- [ ] Configure RTAB-Map parameters:
  - [ ] RGB-D mode with monocular depth
  - [ ] Loop closure detection settings
  - [ ] Memory management (max nodes, optimization frequency)
  - [ ] Odometry type (visual odometry)
- [ ] Configure input topics:
  - `/camera/undistorted` (RGB image)
  - `/perception/depth` (depth map)
  - `/camera_info` (calibration)
  - `/imu/data` (IMU for gravity reference)
- [ ] Set up SLAM output topics:
  - `/rtabmap/odom` (visual odometry)
  - `/rtabmap/mapData` (3D map)
  - `/rtabmap/grid_map` (2D occupancy grid)
- [ ] Test SLAM initialization
- [ ] Test loop closure detection
- [ ] Visualize in RViz2

### 6.3 Semantic SLAM Integration
- [ ] Implement semantic landmark injection
  - [ ] Subscribe to `/perception/objects` (YOLO detections)
  - [ ] Convert detections to landmark constraints
  - [ ] Publish semantic landmarks to RTAB-Map
- [ ] Test object-based loop closure
- [ ] Visualize semantic map in RViz2

### 6.4 SLAM Testing & Tuning
- [ ] Create test environments:
  - [ ] Small room (3m x 3m)
  - [ ] Corridor (10m x 2m)
  - [ ] Open space (5m x 5m)
- [ ] Test SLAM accuracy:
  - [ ] Drive robot in closed loop
  - [ ] Measure loop closure error (target: <5%)
  - [ ] Test in various lighting conditions
- [ ] Test failure modes:
  - [ ] Low texture environments (blank walls)
  - [ ] Darkness (complete visual odometry failure)
  - [ ] Fast rotation (motion blur)
- [ ] Implement fallback modes:
  - [ ] IMU-only localization
  - [ ] Dead reckoning mode
- [ ] Tune RTAB-Map parameters for performance

### 6.5 Localization Integration Test
- [ ] Create `launch/slam_launch.py`
- [ ] Test complete localization pipeline:
  - Camera → Depth → RTAB-Map → Visual Odom
  - IMU → EKF Fusion
  - Output: `/odom` (fused pose)
- [ ] Measure localization accuracy (target: <10cm drift per 10m)
- [ ] Test long-duration operation (30 minutes)
- [ ] Profile memory usage

**Deliverables**:
- Working EKF sensor fusion
- Working RTAB-Map SLAM
- Semantic landmarks integrated
- Localization tested in multiple environments
- Fallback modes implemented and tested
- Launch files for localization and SLAM

---

## Phase 7: Cognitive Core (LLM Integration) (Weeks 9-10)

### 7.1 NanoLLM Setup
- [ ] Install NVIDIA NanoLLM
- [ ] Download and quantize LLM (Phase 3.4 continuation):
  - [ ] Verify model size (<2.5 GB)
  - [ ] Test inference speed (<3s)
  - [ ] Test RAM usage
- [ ] Create `cognitive_core_node` ROS2 package
- [ ] Implement model loading utilities

### 7.2 World State Serialization
- [ ] Implement `world_state_serializer.py`
  - [ ] Subscribe to `/odom` (robot pose)
  - [ ] Subscribe to `/rtabmap/mapData` (semantic map)
  - [ ] Subscribe to `/perception/objects` (current objects)
  - [ ] Subscribe to `/imu/data` (orientation)
  - [ ] Create JSON representation of world state
  - [ ] Limit context size (max 2000 tokens)
- [ ] Design world state schema:
  ```json
  {
    "robot_pose": {"x": 1.2, "y": 0.5, "orientation": 90},
    "nearby_objects": [
      {"type": "cup", "distance": 0.8, "bearing": 45},
      {"type": "person", "distance": 2.1, "bearing": 120}
    ],
    "current_mission": "navigate to red ball",
    "status": "moving"
  }
  ```
- [ ] Test serialization performance (<10ms)

### 7.3 LLM Interface Node
- [ ] Implement `nanollm_interface.py`
  - [ ] ROS2 node structure
  - [ ] Subscribe to `/audio/transcribed_text`
  - [ ] Subscribe to world state from serializer
  - [ ] Lazy load LLM on first complex command
  - [ ] Build prompt with system message + world state + user query
  - [ ] Run inference with NanoLLM
  - [ ] Parse JSON output (structured intent)
  - [ ] Publish intent to `/cognitive/intent` (custom msg)
  - [ ] Publish natural language response to `/audio/tts_request`
  - [ ] Implement model unloading on idle (5 minutes)
- [ ] Design prompt template:
  ```
  You are an AI robot assistant. Parse user commands and output JSON.
  Current state: {world_state}
  User command: {transcribed_text}
  Output format: {"action": "...", "target": "...", "parameters": {...}}
  ```
- [ ] Test prompt engineering for accuracy
- [ ] Implement conversation history (last 3 exchanges)

### 7.4 Intent Message Definition
- [ ] Create custom ROS2 message: `cognitive_msgs/Intent`
  ```
  string action  # navigate, pickup, search, stop, etc.
  string target  # object name or location
  string[] parameters  # additional parameters
  float32 confidence  # LLM confidence (0-1)
  ```
- [ ] Test message serialization

### 7.5 LLM Testing
- [ ] Create test dataset of commands (50+ examples):
  - Simple: "go forward", "stop", "turn left"
  - Complex: "find the red ball and bring it here"
  - Ambiguous: "get that thing over there"
- [ ] Test LLM command understanding:
  - [ ] Accuracy (target: >90% for clear commands)
  - [ ] JSON output format compliance
  - [ ] Context awareness (uses world state)
- [ ] Test conversational abilities:
  - [ ] Multi-turn dialogue
  - [ ] Clarification questions
  - [ ] Status updates
- [ ] Benchmark performance:
  - [ ] Inference time (target: <3s)
  - [ ] RAM usage (target: <2.5 GB when loaded)
  - [ ] Loading time (target: <5s)

### 7.6 Memory Management Integration
- [ ] Implement `model_manager.py`
  - [ ] Monitor system RAM usage
  - [ ] Trigger LLM loading/unloading
  - [ ] Coordinate with perception models
  - [ ] Implement model swapping strategy:
    * Unload YOLO/Depth when loading LLM if RAM >85%
    * Reload perception after LLM response
- [ ] Test memory management:
  - [ ] Simulate high memory pressure
  - [ ] Verify graceful model swapping
  - [ ] Measure swapping time (target: <10s)
- [ ] Create `config/memory_management.yaml`:
  ```yaml
  thresholds:
    warning: 0.80  # 80% RAM usage
    critical: 0.85  # 85% RAM usage
    emergency: 0.90  # 90% RAM usage

  strategies:
    warning: log_warning
    critical: unload_llm
    emergency: unload_all_except_motors
  ```

**Deliverables**:
- Working NanoLLM integration
- World state serialization
- Intent message definition
- LLM command understanding (>90% accuracy)
- Memory management system
- Lazy loading/unloading tested

---

## Phase 8: Behavioral Architecture (Weeks 11-12)

### 8.1 BehaviorTree.CPP Setup
- [ ] Install BehaviorTree.CPP library
- [ ] Create `behavioral_nodes` ROS2 package
- [ ] Set up BehaviorTree.CPP ROS2 integration
- [ ] Create blackboard data structure
- [ ] Test basic behavior tree execution

### 8.2 Command Router
- [ ] Implement `command_router.py`
  - [ ] Subscribe to `/audio/transcribed_text`
  - [ ] Classify command complexity:
    * Simple: direct motor commands (stop, forward, backward, turn)
    * Complex: requires LLM (find X, bring me Y, what do you see?)
  - [ ] Route simple commands directly to behavior tree
  - [ ] Route complex commands to LLM (`/cognitive/llm_request`)
  - [ ] Log routing decisions
- [ ] Create simple command mapping:
  ```python
  SIMPLE_COMMANDS = {
    "stop": {"action": "stop"},
    "go forward": {"action": "move", "direction": "forward"},
    "turn left": {"action": "turn", "direction": "left"},
    "turn right": {"action": "turn", "direction": "right"},
    "go back": {"action": "move", "direction": "backward"}
  }
  ```
- [ ] Test command classification accuracy

### 8.3 Blackboard Implementation
- [ ] Implement blackboard manager
- [ ] Define blackboard schema:
  ```
  - robot_pose (geometry_msgs/PoseStamped)
  - robot_orientation (from IMU)
  - semantic_map (list of objects with poses)
  - current_mission (string)
  - current_goal (geometry_msgs/PoseStamped)
  - navigation_status (enum: idle, moving, stuck, arrived)
  - audio_status (enum: listening, processing, speaking)
  - system_health (dict: CPU, GPU, RAM, temperature)
  - error_log (list of recent errors)
  ```
- [ ] Implement blackboard update subscribers:
  - [ ] Subscribe to `/odom` → update robot_pose
  - [ ] Subscribe to `/imu/data` → update orientation
  - [ ] Subscribe to `/perception/objects` → update semantic_map
  - [ ] Subscribe to `/cognitive/intent` → update current_mission
- [ ] Test blackboard updates (latency <10ms)

### 8.4 Core Behavior Tree Design
- [ ] Design main behavior tree structure:
  ```xml
  <BehaviorTree ID="MainLoop">
    <ReactiveSequence>
      <SafetyCheck/>
      <Fallback>
        <EmergencyStop/>
        <Sequence>
          <CheckForNewCommand/>
          <Fallback>
            <ExecuteSimpleCommand/>
            <ExecuteLLMIntent/>
          </Fallback>
        </Sequence>
      </Fallback>
      <MonitorSystemHealth/>
    </ReactiveSequence>
  </BehaviorTree>
  ```
- [ ] Implement safety behaviors:
  - [ ] `SafetyCheck` - check temperature, battery, errors
  - [ ] `EmergencyStop` - detect emergency stop flag
- [ ] Implement command execution behaviors:
  - [ ] `CheckForNewCommand` - check for new intent or transcribed text
  - [ ] `ExecuteSimpleCommand` - execute motor commands directly
  - [ ] `ExecuteLLMIntent` - execute complex intents from LLM

### 8.5 Navigation Behaviors
- [ ] Implement navigation behavior tree:
  ```xml
  <BehaviorTree ID="NavigateToGoal">
    <Sequence>
      <SetGoalFromIntent/>
      <ReactiveSequence>
        <Fallback>
          <IsGoalReached/>
          <Sequence>
            <ComputePath/>
            <Fallback>
              <IsPathClear/>
              <AvoidObstacle/>
            </Fallback>
            <FollowPath/>
          </Sequence>
        </Fallback>
        <DetectStuck/>
      </ReactiveSequence>
    </Sequence>
  </BehaviorTree>
  ```
- [ ] Implement navigation action nodes:
  - [ ] `SetGoalFromIntent` - extract target from LLM intent
  - [ ] `ComputePath` - simple path planning (A* or direct line)
  - [ ] `IsPathClear` - check for obstacles in depth map
  - [ ] `AvoidObstacle` - simple obstacle avoidance (rotate + retry)
  - [ ] `FollowPath` - send velocity commands to `/cmd_vel`
  - [ ] `IsGoalReached` - check distance to goal (<0.5m)
  - [ ] `DetectStuck` - use IMU to detect stuck state

### 8.6 Stuck Detection & Recovery
- [ ] Implement `stuck_detector.py`
  - [ ] Subscribe to `/cmd_vel` (commanded velocity)
  - [ ] Subscribe to `/imu/data` (actual acceleration)
  - [ ] Detect stuck condition:
    * Commanded velocity > 0 for >3 seconds
    * IMU acceleration < threshold for >3 seconds
  - [ ] Publish stuck flag to blackboard
- [ ] Implement recovery behavior:
  ```xml
  <BehaviorTree ID="RecoverFromStuck">
    <Sequence>
      <StopMotors/>
      <MoveBackward duration="1.0"/>
      <RotateRandom angle="45-135"/>
      <IncrementStuckCounter/>
      <Fallback>
        <IsStuckCountLessThan max="3"/>
        <RequestHumanHelp/>
      </Fallback>
    </Sequence>
  </BehaviorTree>
  ```
- [ ] Test stuck detection and recovery

### 8.7 Dialogue Manager
- [ ] Implement `dialogue_manager.py`
  - [ ] Subscribe to `/cognitive/intent` and blackboard
  - [ ] Generate status updates:
    * "Navigating to red ball"
    * "I'm stuck, trying to recover"
    * "I found the object"
  - [ ] Generate clarification questions:
    * "Which object do you mean?"
    * "I can't find that. Can you be more specific?"
  - [ ] Publish to `/audio/tts_request`
  - [ ] Implement dialogue state machine:
    * Idle → Command Received → Executing → Report Status → Idle
- [ ] Design response templates:
  ```python
  RESPONSES = {
    "navigating": "I'm on my way to {target}",
    "arrived": "I've arrived at {target}",
    "stuck": "I'm having trouble moving. Let me try another way.",
    "object_found": "I found the {object}",
    "object_not_found": "I can't find a {object} nearby",
    "clarification": "I see multiple {objects}. Which one?"
  }
  ```
- [ ] Test dialogue flow

### 8.8 Behavior Tree Executor
- [ ] Implement `behavior_tree_executor.py`
  - [ ] ROS2 node structure
  - [ ] Load behavior trees from XML files
  - [ ] Execute main behavior tree loop (10 Hz)
  - [ ] Publish behavior tree status for visualization
  - [ ] Implement graceful shutdown
- [ ] Create `config/behavior_tree_config.xml`
- [ ] Test behavior tree execution

### 8.9 Integration Testing
- [ ] Test complete behavior flow:
  - Wake word → ASR → Command Router → Behavior Tree → Motors
  - Wake word → ASR → LLM → Intent → Behavior Tree → Navigation
- [ ] Test dialogue integration:
  - Robot provides status updates via TTS
  - Robot asks clarification questions
- [ ] Test error handling:
  - Stuck detection and recovery
  - Object not found scenarios
  - Emergency stop

**Deliverables**:
- Working behavior tree system
- Command routing (simple vs complex)
- Navigation behaviors with stuck detection
- Dialogue management
- Behavior tree executor node
- Integration tests passing

---

## Phase 9: Web Interface & Monitoring (Week 13)

### 9.1 Web Server Backend
- [ ] Create `web_interface_nodes` ROS2 package
- [ ] Implement `web_server.py`
  - [ ] FastAPI application setup
  - [ ] WebSocket support for real-time data
  - [ ] CORS configuration for development
  - [ ] Static file serving
  - [ ] RESTful API endpoints
- [ ] Implement ROS2 bridge node
  - [ ] Subscribe to all monitoring topics
  - [ ] Buffer latest data (1-second window)
  - [ ] Push data to WebSocket clients
  - [ ] Rate limiting (1-10 Hz configurable)
- [ ] Create API endpoints:
  ```python
  GET  /api/status          # System health summary
  GET  /api/robot/pose      # Current robot pose
  GET  /api/perception      # Latest detections
  GET  /api/audio/status    # Audio pipeline state
  GET  /api/system/metrics  # CPU/GPU/RAM/Temp
  POST /api/emergency_stop  # Trigger emergency stop
  POST /api/command         # Send text command
  ```
- [ ] Test API endpoints with curl/Postman

**Tests Required**:
```python
# tests/test_web_server.py
- Test FastAPI initialization
- Test WebSocket connections
- Test API endpoint responses
- Test concurrent client connections
- Test data rate limiting
```

### 9.2 System Monitoring Node
- [ ] Implement `system_monitor.py`
  - [ ] Monitor CPU usage (per core)
  - [ ] Monitor GPU usage and memory
  - [ ] Monitor RAM usage
  - [ ] Monitor temperatures (CPU, GPU, thermal zones)
  - [ ] Monitor disk usage
  - [ ] Monitor network stats
  - [ ] Publish to `/system/metrics` topic (10 Hz)
- [ ] Implement `node_monitor.py`
  - [ ] Track active ROS2 nodes
  - [ ] Monitor topic publication rates
  - [ ] Detect node failures
  - [ ] Log node restarts
  - [ ] Publish to `/system/node_status` topic
- [ ] Create custom messages:
  ```python
  # monitoring_msgs/SystemMetrics.msg
  float32 cpu_percent
  float32[] cpu_cores_percent
  float32 gpu_percent
  float32 gpu_memory_used_mb
  float32 ram_percent
  float32 ram_used_gb
  float32 cpu_temp
  float32 gpu_temp
  float32 disk_percent

  # monitoring_msgs/NodeStatus.msg
  string[] active_nodes
  string[] failed_nodes
  diagnostic_msgs/DiagnosticArray diagnostics
  ```

### 9.3 Frontend Development
- [ ] Create `web_interface_nodes/static/` directory
- [ ] Implement HTML structure (`index.html`)
  - [ ] Navigation sidebar
  - [ ] Dashboard layout (grid system)
  - [ ] Camera feed panel
  - [ ] Map visualization panel
  - [ ] System metrics panel
  - [ ] Audio status panel
  - [ ] Command input panel
  - [ ] Log viewer panel
- [ ] Implement CSS styling (`style.css`)
  - [ ] Responsive design (mobile-friendly)
  - [ ] Dark theme (easier on eyes)
  - [ ] Status indicators (colors for health)
  - [ ] Animation for live updates
- [ ] Implement JavaScript functionality (`app.js`)
  - [ ] WebSocket connection management
  - [ ] Real-time data updates
  - [ ] Camera feed display (MJPEG stream)
  - [ ] Interactive map rendering (Canvas/SVG)
  - [ ] System metrics charts (Chart.js)
  - [ ] Command input and submission
  - [ ] Log scrolling and filtering
  - [ ] Connection status indicator

**Key Features**:
```javascript
// app.js structure
class RobotDashboard {
  - connectWebSocket()
  - updateCameraFeed()
  - updateSystemMetrics()
  - updateRobotPose()
  - updatePerceptionData()
  - updateAudioStatus()
  - sendCommand()
  - handleEmergencyStop()
  - updateLogs()
}
```

### 9.4 Data Visualization
- [ ] Implement camera feed streaming
  - [ ] Create MJPEG streamer node
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Compress frames (JPEG, quality: 80)
  - [ ] Throttle to 5 FPS for web
  - [ ] HTTP endpoint: `/api/camera/stream`
- [ ] Implement object detection overlay
  - [ ] Draw bounding boxes on camera feed
  - [ ] Add labels and confidence scores
  - [ ] Color code by object class
- [ ] Implement depth map visualization
  - [ ] Convert depth to colormap (JET/TURBO)
  - [ ] Optional toggle on camera feed
- [ ] Implement 2D map visualization
  - [ ] Render occupancy grid from RTAB-Map
  - [ ] Show robot pose (arrow/triangle)
  - [ ] Show detected objects (markers)
  - [ ] Show planned path (if navigation active)
  - [ ] Zoom and pan controls
- [ ] Implement system metrics charts
  - [ ] Real-time line charts (Chart.js)
  - [ ] CPU usage over time (60 seconds)
  - [ ] GPU usage over time
  - [ ] Temperature over time
  - [ ] RAM usage gauge

### 9.5 Control Interface
- [ ] Implement text command input
  - [ ] Text input field
  - [ ] Submit button
  - [ ] Command history (last 10 commands)
  - [ ] Send to `/api/command` endpoint
  - [ ] Display robot response
- [ ] Implement emergency stop button
  - [ ] Large red button (prominent)
  - [ ] Confirmation dialog
  - [ ] Call `/api/emergency_stop`
  - [ ] Visual feedback on activation
- [ ] Implement manual control (optional)
  - [ ] Arrow keys for movement
  - [ ] Slider for speed control
  - [ ] Send directly to `/cmd_vel`
  - [ ] Safety timeout (auto-stop after 1s)

### 9.6 Configuration Interface
- [ ] Implement settings panel
  - [ ] Toggle web UI features on/off
  - [ ] Adjust data refresh rates
  - [ ] Camera feed quality settings
  - [ ] Log verbosity settings
  - [ ] Save settings to browser localStorage
- [ ] Implement system controls
  - [ ] Start/stop specific nodes (via ROS2 lifecycle)
  - [ ] Enable/disable LLM loading
  - [ ] Enable/disable perception models
  - [ ] Trigger map save/load

### 9.7 Logging and Diagnostics
- [ ] Implement log viewer
  - [ ] Display ROS2 logs in real-time
  - [ ] Filter by log level (DEBUG, INFO, WARN, ERROR)
  - [ ] Filter by node name
  - [ ] Search functionality
  - [ ] Auto-scroll toggle
  - [ ] Export logs button
- [ ] Implement diagnostics panel
  - [ ] Show node health status (green/yellow/red)
  - [ ] Show topic publication rates
  - [ ] Show message latencies
  - [ ] Show error counts per node
  - [ ] Alert notifications for critical errors

### 9.8 Web Server Integration
- [ ] Create `launch/web_interface_launch.py`
- [ ] Test web server with all nodes running
- [ ] Test multiple simultaneous clients (3+ browsers)
- [ ] Test on mobile devices (phone/tablet)
- [ ] Optimize for low bandwidth (<1 Mbps)
- [ ] Test disconnection and reconnection
- [ ] Profile web server resource usage (target: <200 MB RAM)

### 9.9 Security Considerations
- [ ] Implement basic authentication (optional)
  - [ ] Username/password login
  - [ ] Session management
  - [ ] JWT tokens for API
- [ ] Implement HTTPS (optional, for production)
  - [ ] SSL certificate setup
  - [ ] Redirect HTTP to HTTPS
- [ ] Rate limiting on API endpoints
  - [ ] Prevent command spam
  - [ ] Prevent DoS attacks
- [ ] Input validation
  - [ ] Sanitize command inputs
  - [ ] Validate API parameters
  - [ ] Prevent injection attacks

### 9.10 Performance Optimization
- [ ] Optimize WebSocket data transmission
  - [ ] Use binary format for large data (images)
  - [ ] Compress JSON messages
  - [ ] Batch small updates
- [ ] Optimize camera streaming
  - [ ] Adaptive quality based on bandwidth
  - [ ] Skip frames if clients are slow
- [ ] Optimize frontend rendering
  - [ ] Use requestAnimationFrame for smooth updates
  - [ ] Throttle/debounce event handlers
  - [ ] Lazy load heavy components
- [ ] Test with network latency simulation
  - [ ] 50ms, 100ms, 500ms delays
  - [ ] Verify UI remains responsive

**Deliverables**:
- Working web server (FastAPI + WebSocket)
- System monitoring node
- Complete web dashboard with:
  - Live camera feed with overlays
  - Interactive 2D map
  - System metrics visualization
  - Command input interface
  - Emergency stop button
  - Real-time logs
- Mobile-responsive design
- Performance optimized (<200 MB RAM)
- Documentation for web interface usage

---

## Phase 10: System Integration & Testing (Weeks 14-15)

### 9.1 Full System Launch
- [ ] Create `launch/full_system_launch.py`
  - [ ] Launch all perception nodes
  - [ ] Launch audio pipeline nodes
  - [ ] Launch localization and SLAM
  - [ ] Launch cognitive core (lazy loaded)
  - [ ] Launch behavioral architecture
  - [ ] Launch web interface
  - [ ] Launch monitoring nodes
- [ ] Create launch configuration options:
  - [ ] `--minimal` - motors + wake word only (emergency mode)
  - [ ] `--no-llm` - skip LLM loading (simple commands only)
  - [ ] `--no-web` - disable web interface (save RAM)
  - [ ] `--perception-only` - camera + perception for testing
- [ ] Test full system startup (<30 seconds)
- [ ] Test graceful shutdown (all nodes stop cleanly)
- [ ] Test web interface access during startup

### 9.2 End-to-End Testing
- [ ] Create test scenarios:

  **Scenario 1: Simple Navigation**
  - [ ] User says wake word
  - [ ] User says "go forward"
  - [ ] Robot moves forward for 1 meter
  - [ ] Robot stops and says "Done"

  **Scenario 2: Object Finding**
  - [ ] User says wake word
  - [ ] User says "find the red ball"
  - [ ] Robot rotates to scan environment
  - [ ] Robot detects ball with YOLO
  - [ ] Robot navigates to ball
  - [ ] Robot says "I found the red ball"

  **Scenario 3: Complex Command**
  - [ ] User says wake word
  - [ ] User says "bring me that cup on the table"
  - [ ] LLM interprets command
  - [ ] Robot navigates to table
  - [ ] Robot stops near cup
  - [ ] Robot says "I'm at the cup, but I can't pick it up yet"

  **Scenario 4: Stuck Recovery**
  - [ ] Robot encounters obstacle while navigating
  - [ ] Stuck detector triggers after 3 seconds
  - [ ] Robot backs up and rotates
  - [ ] Robot retries navigation
  - [ ] Robot says "I got stuck but found another way"

  **Scenario 5: Clarification**
  - [ ] User says wake word
  - [ ] User says "go to the ball"
  - [ ] Multiple balls detected
  - [ ] Robot asks "Which ball? The red one or the blue one?"
  - [ ] User responds "the red one"
  - [ ] Robot navigates to red ball

- [ ] Test each scenario 5+ times
- [ ] Measure success rate (target: >80%)
- [ ] Document failure modes
- [ ] Monitor via web interface during tests
- [ ] Verify web interface shows correct real-time data

### 9.3 Performance Profiling
- [ ] Profile full system under load:
  - [ ] CPU usage per core (target: <80% average)
  - [ ] GPU usage (target: <90%)
  - [ ] RAM usage (target: <7.5 GB)
  - [ ] Temperature (target: <80°C sustained)
- [ ] Measure latencies:
  - [ ] Wake word to ASR complete (<3 seconds)
  - [ ] ASR to LLM response (<5 seconds for complex)
  - [ ] LLM response to TTS start (<1 second)
  - [ ] Command to motor action (<2 seconds for simple)
  - [ ] Object detection to navigation start (<5 seconds)
- [ ] Profile individual nodes:
  - [ ] Identify bottlenecks
  - [ ] Optimize hot paths
  - [ ] Reduce memory allocations
- [ ] Create performance dashboard

### 9.4 Stress Testing
- [ ] Run system for extended periods:
  - [ ] 1-hour continuous operation
  - [ ] 4-hour continuous operation
  - [ ] 8-hour continuous operation (overnight)
- [ ] Monitor for:
  - [ ] Memory leaks (RAM should be stable)
  - [ ] Performance degradation
  - [ ] Thermal throttling
  - [ ] Error accumulation
- [ ] Test under high load:
  - [ ] Multiple rapid commands
  - [ ] Continuous navigation
  - [ ] Frequent LLM invocations

### 9.5 Failure Mode Testing
- [ ] Test hardware failures:
  - [ ] USB microphone disconnect during recording
  - [ ] USB speaker disconnect during playback
  - [ ] Camera disconnect during operation
  - [ ] UART communication loss
- [ ] Test software failures:
  - [ ] Model inference failure (corrupted engine)
  - [ ] ROS2 node crash recovery
  - [ ] Out of memory condition
  - [ ] Emergency stop during mission
- [ ] Test edge cases:
  - [ ] Very long user commands (>30 seconds)
  - [ ] Completely dark environment
  - [ ] Blank wall environment (no visual features)
  - [ ] Very noisy audio environment
  - [ ] Rapid rotation causing motion blur
- [ ] Verify recovery mechanisms work

### 9.6 User Acceptance Testing
- [ ] Recruit 3-5 test users
- [ ] Provide basic instructions (wake word, example commands)
- [ ] Observe user interactions:
  - [ ] Command understanding rate
  - [ ] User frustration points
  - [ ] Unexpected use cases
- [ ] Collect feedback:
  - [ ] Voice quality and naturalness
  - [ ] Response time satisfaction
  - [ ] Command understanding accuracy
  - [ ] Overall experience rating
- [ ] Iterate based on feedback

**Deliverables**:
- Full system launch file tested
- End-to-end scenarios passing (>80% success)
- Performance benchmarks documented
- Stress testing results
- Failure modes tested and recovery verified
- User acceptance testing completed

---

## Phase 11: Optimization & Documentation (Weeks 16-17)

### 10.1 Performance Optimization
- [ ] Analyze profiling data from Phase 9.3
- [ ] Optimize bottlenecks:
  - [ ] Reduce memory allocations in hot paths
  - [ ] Optimize image copy operations (use zero-copy where possible)
  - [ ] Tune ROS2 QoS settings for throughput
  - [ ] Optimize TensorRT engines (INT8 calibration if needed)
  - [ ] Reduce Python overhead (Cython for critical functions)
- [ ] Optimize model loading:
  - [ ] Cache TensorRT engines in shared memory
  - [ ] Preload frequently used models
  - [ ] Optimize LLM loading time
- [ ] Optimize power consumption:
  - [ ] Tune GPU clock frequencies
  - [ ] Implement power modes (performance vs efficiency)
- [ ] Re-run benchmarks and verify improvements

### 10.2 Memory Optimization
- [ ] Implement aggressive memory management:
  - [ ] Tune model unloading thresholds
  - [ ] Optimize blackboard size (limit history)
  - [ ] Reduce image buffer sizes
  - [ ] Implement buffer pooling
- [ ] Test memory management under various scenarios:
  - [ ] Low memory conditions
  - [ ] Rapid model swapping
  - [ ] Long-term operation
- [ ] Document memory usage patterns

### 10.3 Code Quality & Refactoring
- [ ] Code review for all nodes
- [ ] Refactor complex functions
- [ ] Add type hints to all Python code
- [ ] Add docstrings to all classes and functions
- [ ] Fix linting issues (pylint, flake8)
- [ ] Add logging statements for debugging
- [ ] Remove dead code and TODOs

### 10.4 Documentation
- [ ] Update architecture document with final implementation details
- [ ] Create user manual:
  - [ ] Getting started guide
  - [ ] Hardware setup instructions
  - [ ] Software installation guide
  - [ ] Configuration guide
  - [ ] Troubleshooting section
  - [ ] FAQ
- [ ] Create developer documentation:
  - [ ] System architecture overview
  - [ ] ROS2 node documentation
  - [ ] Message definitions
  - [ ] Configuration parameters
  - [ ] Extending the system
- [ ] Create deployment guide:
  - [ ] Docker deployment
  - [ ] Systemd service setup
  - [ ] Auto-start on boot
  - [ ] Remote monitoring setup
- [ ] Document known limitations:
  - [ ] Performance constraints
  - [ ] Hardware requirements
  - [ ] Unsupported scenarios
- [ ] Create video demonstrations:
  - [ ] System overview
  - [ ] Example interactions
  - [ ] Setup walkthrough

### 10.5 Testing & Validation
- [ ] Run full test suite:
  - [ ] Unit tests (>80% coverage)
  - [ ] Integration tests
  - [ ] End-to-end tests
  - [ ] Performance tests
- [ ] Fix any failing tests
- [ ] Add missing tests
- [ ] Create continuous integration pipeline:
  - [ ] Automated testing on commit
  - [ ] Build verification
  - [ ] Linting checks

### 10.6 Deployment Preparation
- [ ] Create Docker container:
  - [ ] Multi-stage Dockerfile for optimized image size
  - [ ] Include all dependencies and models
  - [ ] Configure GPU access
  - [ ] Test container on fresh Jetson
- [ ] Create systemd services:
  - [ ] Main robot service (auto-start on boot)
  - [ ] Health monitoring service
  - [ ] Log rotation service
- [ ] Create backup and restore scripts:
  - [ ] Configuration backup
  - [ ] Model backup
  - [ ] Map data backup
- [ ] Create remote monitoring setup:
  - [ ] SSH access configuration
  - [ ] Web interface for remote status
  - [ ] Remote shutdown capability
- [ ] Security hardening:
  - [ ] Disable unnecessary services
  - [ ] Configure firewall rules
  - [ ] Set up secure SSH keys
  - [ ] Remove default passwords

### 10.7 Final Validation
- [ ] Run complete test suite one final time
- [ ] Perform fresh installation on clean Jetson
- [ ] Verify all documentation is accurate
- [ ] Test with external users (3+ people)
- [ ] Create final performance report
- [ ] Document any remaining known issues

**Deliverables**:
- Optimized system (20-30% performance improvement)
- Complete documentation (user + developer)
- Deployment package (Docker + systemd)
- Final validation report
- Video demonstrations
- Ready for production use

---

## Phase 12: Advanced Features & Polish (Weeks 18-19) [OPTIONAL]

### 11.1 Advanced Navigation
- [ ] Implement path planning with obstacle avoidance
- [ ] Add dynamic replanning
- [ ] Implement wall following
- [ ] Add exploration mode (autonomous mapping)
- [ ] Test in complex environments

### 11.2 Improved Dialogue
- [ ] Add conversation history (last 5 exchanges)
- [ ] Implement context-aware responses
- [ ] Add personality to robot responses
- [ ] Support multi-turn conversations
- [ ] Add emotion detection from voice tone

### 11.3 Object Manipulation (Future Work)
- [ ] Design gripper interface (placeholder)
- [ ] Add object grasping behaviors to tree
- [ ] Implement approach and align behaviors
- [ ] Document integration points for future hardware

### 12.4 Web Interface Enhancements
- [ ] Add interactive map view with zoom/pan
- [ ] Add manual control interface (teleoperation)
- [ ] Add configuration editor (modify YAML files from web)
- [ ] Add mission replay (recorded sessions playback)
- [ ] Add performance graphs and statistics
- [ ] Add mobile app using same API (React Native/Flutter)

### 12.5 Multi-Robot Support (Future Work)
- [ ] Design multi-robot communication protocol
- [ ] Add robot discovery mechanism
- [ ] Implement task coordination
- [ ] Test with 2 robots (if available)

**Deliverables**:
- Advanced features implemented
- Enhanced user experience
- Foundation for future extensions

---

## Testing Summary

### Unit Tests (Target: >80% coverage)
- Hardware abstraction layer
- Sensor data processing
- Model inference wrappers
- ROS2 message conversions
- Behavior tree conditions and actions
- Utility functions

### Integration Tests
- UART communication pipeline
- Camera to perception pipeline
- Audio capture to TTS pipeline
- Localization (IMU + Visual Odom + EKF)
- Perception to behavior tree
- Complete audio-to-action flow

### Performance Tests
- Model inference benchmarks
- End-to-end latency measurements
- Memory usage profiling
- CPU/GPU utilization monitoring
- Thermal performance under load
- Long-duration stability tests

### System Tests
- End-to-end scenarios (5+ scenarios)
- Failure mode testing
- Recovery mechanism validation
- User acceptance testing
- Stress testing (8+ hours)

---

## Risk Management

### High-Risk Items
1. **RAM Budget Exceeded**
   - **Mitigation**: Aggressive model quantization (INT8/INT4), lazy loading, model swapping
   - **Contingency**: Use smaller models (Phi-2 instead of LLaMA-2, Whisper Tiny only)

2. **Visual Odometry Failure**
   - **Mitigation**: Multiple fallback modes (IMU-only, dead reckoning)
   - **Contingency**: Add wheel encoders to Wave Rover (hardware modification)

3. **Audio Latency Too High**
   - **Mitigation**: Optimize each pipeline stage, use faster-whisper, optimize TTS
   - **Contingency**: Accept higher latency, set user expectations

4. **LLM Inference Too Slow**
   - **Mitigation**: INT4 quantization, smaller model (Phi-2), optimize prompts
   - **Contingency**: Cloud-based LLM fallback (requires internet)

5. **Thermal Throttling**
   - **Mitigation**: Optimize workloads, reduce model sizes, add cooling
   - **Contingency**: Reduce frame rates, disable web interface, active cooling fan

### Medium-Risk Items
1. **TensorRT Conversion Issues**
   - **Mitigation**: Use ONNX as fallback, extensive testing
   - **Contingency**: Run models in ONNX Runtime (slower but functional)

2. **SLAM Drift Accumulation**
   - **Mitigation**: Frequent loop closure, semantic landmarks
   - **Contingency**: Periodic manual recalibration, external localization (markers)

3. **USB Device Reliability**
   - **Mitigation**: Health monitoring, auto-reconnection
   - **Contingency**: Use Jetson built-in audio (if available)

### Low-Risk Items
1. **ROS2 Communication Overhead**
   - **Mitigation**: Optimize QoS settings, use composition

2. **Model Accuracy Degradation**
   - **Mitigation**: Validation during conversion, accuracy tests

3. **Power Supply Issues**
   - **Mitigation**: Use recommended power supply, monitor voltage

---

## Resource Allocation

### Time Budget by Phase
- Phase 0 (Setup): 1 week
- Phase 1 (Hardware Validation): 1 week
- Phase 2 (Core Infrastructure): 2 weeks
- Phase 3 (Model Conversion): 1 week
- Phase 4 (Perception Integration): 1 week
- Phase 5 (Audio Pipeline): 1 week
- Phase 6 (SLAM & Localization): 1 week
- Phase 7 (Cognitive Core): 2 weeks
- Phase 8 (Behavioral Architecture): 2 weeks
- Phase 9 (Web Interface & Monitoring): 1 week
- Phase 10 (System Integration): 2 weeks
- Phase 11 (Optimization & Docs): 2 weeks
- **Total: 17 weeks**
- Phase 12 (Advanced Features): 2 weeks [OPTIONAL]

### Hardware Requirements
- **Development**: Jetson Orin Nano (8GB), Wave Rover, IMX219 camera, USB audio devices
- **Storage**: 256GB+ NVMe SSD (models, maps, logs)
- **Networking**: WiFi or Ethernet for development
- **Cooling**: Active cooling recommended for sustained operation
- **Power**: 15W+ power supply

### Software Tools
- **OS**: Ubuntu 20.04/22.04 with JetPack
- **IDE**: VS Code with Remote-SSH
- **Version Control**: Git + GitHub
- **Containerization**: Docker
- **Documentation**: Sphinx or MkDocs
- **Monitoring**: htop, nvtop, jtop (jetson-stats)

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Robot responds to wake word (>95% detection rate)
- [ ] Robot understands simple voice commands (>90% accuracy)
- [ ] Robot can navigate to visible objects (>80% success)
- [ ] Robot provides voice feedback for status
- [ ] System runs for 1+ hour without crashes
- [ ] End-to-end latency <5 seconds for complex commands
- [ ] Safe operation (emergency stop works, no collisions)

### Full System Goals
- [ ] Complex command understanding with LLM (>90% accuracy)
- [ ] Semantic SLAM with object tracking
- [ ] Stuck detection and recovery (>80% success)
- [ ] Multi-turn conversations
- [ ] 8+ hour continuous operation
- [ ] RAM usage <7.5 GB
- [ ] Temperature <80°C sustained
- [ ] User satisfaction rating >4/5

### Performance Targets
| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Wake Word Detection | <100ms | <200ms |
| Speech-to-Text | <2s for 5s audio | <4s |
| Text-to-Speech | <500ms for 20 words | <1s |
| LLM Inference | <3s | <5s |
| Object Detection | 20+ FPS | 15+ FPS |
| Depth Estimation | 15+ FPS | 10+ FPS |
| End-to-End Latency | <5s | <8s |
| RAM Usage | <7.5 GB | <8 GB |
| CPU Usage | <80% average | <95% |
| Temperature | <80°C | <85°C |

---

## Maintenance & Support Plan

### Regular Maintenance
- **Weekly**: Review error logs, check system health
- **Monthly**: Update dependencies, security patches
- **Quarterly**: Model retraining (if needed), performance tuning

### Backup Strategy
- **Daily**: Map data, configuration files
- **Weekly**: Full system backup
- **Before Updates**: Complete snapshot

### Monitoring
- Continuous system health monitoring
- Automated alerts for critical errors
- Performance metrics logging
- User interaction logging (for debugging)

### Update Strategy
- Test updates in development environment
- Staged rollout (test → production)
- Rollback plan for failed updates
- Version control for all components

---

## Appendix A: Hardware Pin Connections

### Jetson Orin Nano Connections
```
Camera: MIPI CSI-2 port (J10)
UART: Pin 8 (TX), Pin 10 (RX), Pin 6 (GND)
USB: Multiple USB 3.0 ports for audio devices
Power: DC barrel jack (5V 3A minimum)
NVMe: M.2 Key M slot for SSD
```

### Wave Rover UART Protocol
```
Baud Rate: 115200
Data Bits: 8
Stop Bits: 1
Parity: None
Format: JSON strings
```

---

## Appendix B: ROS2 Topic Structure

### Sensor Topics
```
/camera/raw                    # sensor_msgs/Image (NVMM)
/camera/undistorted           # sensor_msgs/Image
/camera_info                  # sensor_msgs/CameraInfo
/imu/data                     # sensor_msgs/Imu (50 Hz)
/audio/raw                    # audio_msgs/AudioData (16 kHz)
```

### Perception Topics
```
/perception/objects           # vision_msgs/Detection2DArray
/perception/depth             # sensor_msgs/Image
/perception/pointcloud        # sensor_msgs/PointCloud2
```

### Localization Topics
```
/odom                         # nav_msgs/Odometry (fused)
/odom_raw                     # nav_msgs/Odometry (dead reckoning)
/rtabmap/odom                 # nav_msgs/Odometry (visual)
/rtabmap/mapData              # rtabmap_ros/MapData
/rtabmap/grid_map             # nav_msgs/OccupancyGrid
```

### Audio Topics
```
/audio/wake_word_detected     # std_msgs/Bool
/audio/transcribed_text       # std_msgs/String
/audio/tts_request            # std_msgs/String
/audio/tts_output             # audio_msgs/AudioData
```

### Cognitive Topics
```
/cognitive/intent             # cognitive_msgs/Intent
/cognitive/world_state        # cognitive_msgs/WorldState
```

### Control Topics
```
/cmd_vel                      # geometry_msgs/Twist (20 Hz)
/motor_status                 # custom_msgs/MotorStatus
/emergency_stop               # std_srvs/Trigger (service)
```

---

## Appendix C: Configuration Files Reference

### `config/uart_config.yaml`
```yaml
uart:
  port: "/dev/ttyTHS0"
  baud_rate: 115200
  timeout: 1.0

motor_control:
  command_rate: 20  # Hz
  max_speed: 0.5    # PWM percentage
  wheelbase: 0.2    # meters

watchdog:
  timeout: 0.5      # seconds
```

### `config/camera_config.yaml`
```yaml
camera:
  width: 640
  height: 480
  fps: 30
  calibration_file: "config/camera_calibration.yaml"

undistortion:
  enabled: true
  use_gpu: true
```

### `config/audio_config.yaml`
```yaml
microphone:
  device_index: 0
  sample_rate: 16000
  chunk_size: 1024

speaker:
  device_index: 1
  sample_rate: 22050

wake_word:
  model_path: "models/wake_word/openWakeWord.onnx"
  threshold: 0.5
  cooldown: 2.0  # seconds
```

### `config/memory_management.yaml`
```yaml
thresholds:
  warning: 0.80   # 80% RAM
  critical: 0.85  # 85% RAM
  emergency: 0.90 # 90% RAM

strategies:
  warning: log_warning
  critical: unload_llm
  emergency: unload_all_except_motors

models:
  llm:
    lazy_load: true
    unload_timeout: 300  # seconds (5 minutes)

  perception:
    always_loaded: false
    priority: high
```

---

## Appendix D: Useful Commands

### Jetson Monitoring
```bash
# Monitor system resources
jtop

# Monitor GPU usage
nvtop

# Check temperatures
watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Monitor power
sudo tegrastats
```

### ROS2 Commands
```bash
# List all topics
ros2 topic list

# Echo topic data
ros2 topic echo /camera/raw

# Check node status
ros2 node list

# Check topic frequency
ros2 topic hz /imu/data

# Visualize in RViz2
rviz2
```

### Testing Commands
```bash
# Run unit tests
colcon test --packages-select perception_nodes

# Run with coverage
colcon test --packages-select perception_nodes --pytest-args --cov

# Run specific test
python3 -m pytest tests/test_uart_motor_controller.py -v
```

### Deployment Commands
```bash
# Build Docker image
docker build -t robot-assistant:latest .

# Run Docker container
docker run --runtime nvidia --network host \
  -v /dev:/dev --privileged \
  robot-assistant:latest

# Start systemd service
sudo systemctl start robot-assistant

# Check service status
sudo systemctl status robot-assistant

# View logs
journalctl -u robot-assistant -f
```

---

## Appendix E: Troubleshooting Guide

### Issue: Camera not detected
**Symptoms**: No `/dev/video*` devices
**Solutions**:
1. Check camera cable connection
2. Verify JetPack installation: `dmesg | grep CSI`
3. Test with: `nvgstcapture-1.0`

### Issue: UART communication fails
**Symptoms**: No response from Wave Rover
**Solutions**:
1. Check UART connections (TX, RX, GND)
2. Verify baud rate (115200)
3. Check permissions: `sudo chmod 666 /dev/ttyTHS0`
4. Test with: `screen /dev/ttyTHS0 115200`

### Issue: Out of memory errors
**Symptoms**: Nodes crash with OOM
**Solutions**:
1. Check RAM usage: `free -h`
2. Verify swap is active: `swapon --show`
3. Enable model unloading in config
4. Reduce model sizes (use smaller LLM)

### Issue: Thermal throttling
**Symptoms**: Performance drops over time
**Solutions**:
1. Check temperature: `jtop`
2. Improve cooling (add heatsink/fan)
3. Reduce workload (lower frame rates)
4. Enable power efficiency mode

### Issue: Audio device not found
**Symptoms**: PyAudio device errors
**Solutions**:
1. List devices: `arecord -l` and `aplay -l`
2. Check USB connections
3. Update device indices in config
4. Test with: `arecord -d 5 test.wav && aplay test.wav`

### Issue: LLM inference very slow
**Symptoms**: >10 second response time
**Solutions**:
1. Verify INT4 quantization
2. Check RAM usage (may be swapping)
3. Use smaller model (Phi-2)
4. Reduce context length

---

## Conclusion

This implementation plan provides a comprehensive, phase-by-phase approach to building a fully autonomous, locally-operated AI robot assistant on the NVIDIA Jetson Orin Nano. The plan emphasizes:

1. **Hardware-first validation** to catch issues early
2. **Incremental integration** to maintain stability
3. **Thorough testing** at each phase
4. **Performance optimization** for edge deployment
5. **Robust error handling** for production use

**Key Success Factors**:
- Disciplined adherence to the phase structure
- Regular performance profiling and optimization
- Comprehensive testing (unit, integration, system)
- Proper memory management with model swapping
- Multiple fallback mechanisms for robustness

**Timeline Summary**:
- **Weeks 1-2**: Setup and hardware validation
- **Weeks 3-4**: Core infrastructure (ROS2 nodes)
- **Weeks 5-6**: Model conversion and perception
- **Weeks 7-8**: Audio pipeline and SLAM
- **Weeks 9-12**: Cognitive core and behaviors
- **Weeks 13-16**: Integration, testing, and documentation

With this plan, you will have a production-ready, voice-controlled robot assistant capable of understanding natural language, navigating autonomously, and operating entirely offline on edge hardware.

**Next Steps**:
1. Review and approve this plan
2. Set up development environment (Phase 0)
3. Begin hardware validation (Phase 1)
4. Track progress using GitHub Projects board
5. Regular team check-ins (weekly recommended)

Good luck with your robot assistant project! 🤖
