# Implementation Plan & Development Checklist
## Local AI Robot Assistant Project

**Project Timeline**: 12-16 weeks
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
- [x] Flash Jetson Orin Nano with JetPack SDK
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

**Deliverables**:
- GitHub repo initialized
- Jetson fully configured
- Hardware inventory documented

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
- [x] Document baud rate and communication protocol
- [x] Create `hardware_tests/test_waveroever_uart.py`

**Test Script Requirements**:
```python
# hardware_tests/test_waveroever_uart.py
- Test connection establishment
- Test JSON parsing of responses
- Test all motor control commands
- Test IMU data format
- Test communication error handling
- Benchmark communication latency
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
```

### 1.3 Camera Calibration (Critical)
- [x] Print checkerboard calibration pattern
- [x] Capture 20-30 calibration images
- [x] Run OpenCV calibration script
- [x] Generate camera intrinsics matrix
- [x] Generate distortion coefficients
- [x] Validate calibration with test images
- [x] Save calibration to `config/camera_calibration.yaml`
- [x] Create `hardware_tests/calibrate_camera.py`
- [x] Create `hardware_tests/test_undistortion.py`

**Test Script Requirements**:
```python
# hardware_tests/calibrate_camera.py
- Automated checkerboard detection
- Calibration optimization
- Reprojection error calculation
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
- Measure audio latency
- Test noise levels
```

### 1.5 Power & Thermal Testing
- [x] Test Jetson power consumption under idle
- [x] Test power consumption under full load
- [x] Monitor temperature during extended operation
- [x] Test thermal throttling behavior
- [x] Verify cooling solution adequacy
- [x] Create `hardware_tests/test_thermal_power.py`

**Deliverables**:
- All hardware validated
- Test scripts for each component
- Hardware test results documented
- Camera calibration file generated

---

## Phase 2: Core Infrastructure (Weeks 3-4)

### 2.1 ROS2 Workspace Setup
- [x] Create ROS2 workspace structure
- [x] Create all package directories
- [x] Set up package.xml files for each package
- [x] Set up CMakeLists.txt or setup.py files
- [x] Create custom message definitions
- [x] Build workspace and verify
- [x] Set up colcon build configuration
- [x] Create launch file directory structure

### 2.2 UART Communication Node
- [ ] Implement `uart_motor_controller.py`
  - [ ] Serial port initialization
  - [ ] JSON command builder
  - [ ] JSON response parser
  - [ ] ROS2 node structure
  - [ ] Subscribe to `/cmd_vel` topic
  - [ ] Publish to `/motor_status` topic
  - [ ] Implement differential drive kinematics
  - [ ] Add watchdog timer
  - [ ] Add emergency stop service
- [ ] Implement `uart_imu_node.py`
  - [ ] Periodic IMU query (20 Hz)
  - [ ] JSON response parser
  - [ ] Publish to `/imu/data` topic
  - [ ] Data validation
  - [ ] Error handling
- [ ] Create unit tests for both nodes
- [ ] Create integration test for UART package
- [ ] Document UART protocol in package README

**Tests Required**:
```python
# tests/test_uart_motor_controller.py
- Test JSON command generation
- Test velocity to wheel speed conversion
- Test watchdog timer
- Test emergency stop
- Mock serial communication

# tests/test_uart_imu_node.py
- Test IMU data parsing
- Test ROS message conversion
- Test error handling
```

### 2.3 Camera Pipeline (DeepStream-Accelerated)
- [ ] Implement `camera_driver.py`
  - [ ] DeepStream pipeline setup with nvarguscamerasrc
  - [ ] ROS2 node structure
  - [ ] Publish raw images to `/camera/raw` (using NVMM buffers)
  - [ ] Implement hardware-accelerated frame rate control
  - [ ] Add camera info publisher
  - [ ] Utilize GPU memory for zero-copy operations
- [ ] Implement `image_undistort_node.py`
  - [ ] Load calibration from YAML
  - [ ] Subscribe to `/camera/raw`
  - [ ] Apply GPU-accelerated undistortion with DeepStream
  - [ ] Publish to `/camera/undistorted`
  - [ ] Add performance monitoring for GPU usage
- [ ] Create unit tests
- [ ] Create integration test
- [ ] Benchmark processing latency and GPU memory usage

**Tests Required**:
```python
# tests/test_camera_driver.py
- Test DeepStream pipeline initialization
- Test hardware-accelerated frame publishing
- Test frame rate consistency with NVMM memory
- Test GPU memory usage optimization

# tests/test_image_undistort.py
- Test calibration loading
- Test GPU-accelerated undistortion algorithm
- Test output image quality with DeepStream
- Benchmark performance vs CPU implementation
```

### 2.4 Configuration Management
- [ ] Create `config/uart_config.yaml`
- [ ] Create `config/camera_config.yaml`
- [ ] Create `config/audio_config.yaml`
- [ ] Create parameter loading utilities
- [ ] Test configuration validation

**Deliverables**:
- Working UART communication
- Working camera pipeline
- Unit tests for all nodes
- Configuration files

---

## Phase 3: Perception Models (Weeks 5-6)

### 3.1 Model Acquisition & Preparation
- [ ] Download YOLOv8n PyTorch model
- [ ] Download FastDepth PyTorch model
- [ ] Download Whisper Tiny model
- [ ] Download Piper TTS model and voice
- [ ] Download openWakeWord model
- [ ] Download LLaMA-2 7B or Gemma 2-7B
- [ ] Organize models in `/models` directory
- [ ] Document model sources and licenses

### 3.2 YOLO Object Detection
- [ ] Convert YOLOv8n to ONNX format
- [ ] Convert ONNX to TensorRT engine (FP16)
- [ ] Implement `object_detector.py`
  - [ ] Load TensorRT engine
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Run inference
  - [ ] Publish detections to `/perception/objects`
  - [ ] Add visualization overlay
- [ ] Benchmark inference time
- [ ] Test on various objects
- [ ] Create unit tests
- [ ] Document supported object classes

**Tests Required**:
```python
# tests/test_object_detector.py
- Test model loading
- Test inference on sample images
- Test detection accuracy
- Benchmark FPS
- Test bounding box format

# manual_tests/test_yolo_realtime.py
- Live camera feed test
- Visual validation of detections
- FPS monitoring
```

### 3.3 Depth Estimation
- [ ] Convert FastDepth to ONNX format
- [ ] Convert ONNX to TensorRT engine (FP16)
- [ ] Implement `depth_estimator.py`
  - [ ] Load TensorRT engine
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Run inference
  - [ ] Publish depth maps to `/perception/depth`
  - [ ] Add depth colormap visualization
- [ ] Benchmark inference time
- [ ] Test depth accuracy with known distances
- [ ] Create unit tests

**Tests Required**:
```python
# tests/test_depth_estimator.py
- Test model loading
- Test inference on sample images
- Test depth range validation
- Benchmark FPS

# manual_tests/test_depth_accuracy.py
- Compare estimated depth with measured depth
- Test at various distances (0.5m - 5m)
- Visualize depth maps
```

### 3.4 Point Cloud Generation
- [ ] Implement point cloud generation from depth
- [ ] Use camera intrinsics for back-projection
- [ ] Test point cloud accuracy
- [ ] Publish to `/perception/pointcloud`
- [ ] Visualize in RViz2

**Deliverables**:
- All perception models converted to TensorRT
- Working object detection node
- Working depth estimation node
- Model performance benchmarks
- Test results documented

---

## Phase 4: Audio Pipeline (Weeks 7-8)

### 4.1 Audio Capture & Playback
- [ ] Implement `audio_capture_node.py`
  - [ ] PyAudio initialization
  - [ ] Configure USB microphone
  - [ ] Continuous audio streaming
  - [ ] Publish to `/audio/raw`
  - [ ] Add audio buffer management
- [ ] Implement `audio_playback_node.py`
  - [ ] PyAudio initialization
  - [ ] Configure USB speakers
  - [ ] Subscribe to `/audio/tts_output`
  - [ ] Queue-based playback
  - [ ] Handle playback interruptions
- [ ] Test audio latency
- [ ] Test simultaneous capture/playback

**Tests Required**:
```python
# tests/test_audio_capture.py
- Test device initialization
- Test audio streaming
- Test buffer management

# tests/test_audio_playback.py
- Test device initialization
- Test audio playback
- Test queue management
```

### 4.2 Wake Word Detection
- [ ] Install openWakeWord library
- [ ] Implement `wake_word_detector_node.py`
  - [ ] Load wake word model
  - [ ] Subscribe to `/audio/raw`
  - [ ] Run continuous detection
  - [ ] Publish to `/audio/wake_word_detected`
  - [ ] Add detection confidence threshold
- [ ] Train or configure custom wake word
- [ ] Test false positive rate
- [ ] Test detection latency
- [ ] Optimize for low CPU usage

**Tests Required**:
```python
# tests/test_wake_word.py
- Test model loading
- Test detection on sample audio
- Benchmark CPU usage
- Measure detection latency

# manual_tests/test_wake_word_accuracy.py
- Test with multiple speakers
- Test in various noise levels
- Measure false positive/negative rates
```

### 4.3 Speech-to-Text (Whisper)
- [ ] Convert Whisper Tiny to TensorRT/ONNX
- [ ] Implement VAD (Voice Activity Detection)
- [ ] Implement `stt_node.py`
  - [ ] Load Whisper model
  - [ ] Subscribe to wake word trigger
  - [ ] Capture audio segment with VAD
  - [ ] Run transcription
  - [ ] Publish to `/audio/transcribed_text`
  - [ ] Add timeout handling
- [ ] Test transcription accuracy
- [ ] Test with various accents/speaking styles
- [ ] Benchmark inference time
- [ ] Optimize for low latency

**Tests Required**:
```python
# tests/test_stt.py
- Test model loading
- Test transcription on sample audio
- Test VAD functionality
- Benchmark latency

# manual_tests/test_stt_accuracy.py
- Test with various commands
- Test in noisy environments
- Calculate Word Error Rate (WER)
```

### 4.4 Text-to-Speech (Piper)
- [ ] Install Piper TTS
- [ ] Download and test voice model
- [ ] Implement `tts_node.py`
  - [ ] Load Piper model
  - [ ] Subscribe to `/audio/tts_request`
  - [ ] Synthesize speech
  - [ ] Publish to `/audio/tts_output`
  - [ ] Add speech rate control
- [ ] Test voice quality
- [ ] Test synthesis latency
- [ ] Test various sentence lengths
- [ ] Optimize for low latency

**Tests Required**:
```python
# tests/test_tts.py
- Test model loading
- Test synthesis on sample text
- Benchmark latency
- Test voice quality

# manual_tests/test_tts_naturalness.py
- Test various sentence types
- Test punctuation handling
- User feedback on naturalness
```

### 4.5 Audio Pipeline Integration
- [ ] Create `audio_pipeline_launch.py`
- [ ] Test complete audio flow (mic → wake word → ASR → TTS → speaker)
- [ ] Measure end-to-end latency
- [ ] Test error recovery
- [ ] Optimize resource usage

**Deliverables**:
- Working audio pipeline
- All audio models optimized
- Audio tests passing
- Latency benchmarks documented

---

## Phase 5: SLAM & Localization (Weeks 9-10)

### 5.1 Robot Localization Setup
- [ ] Install `robot_localization` package
- [ ] Create `localization_config.yaml`
- [ ] Configure EKF parameters
- [ ] Set up sensor inputs (/imu/data, /visual_odom)
- [ ] Test odometry fusion
- [ ] Tune EKF parameters
- [ ] Create launch file

**Tests Required**:
```python
# tests/test_localization.py
- Test EKF initialization
- Test sensor fusion accuracy
- Test with simulated data
- Benchmark performance
