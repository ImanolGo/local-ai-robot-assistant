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
# ✅ COMPLETED: Full thermal/power validation documented
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
- [x]                  Document thermal/power validation results: `thermal_power_validation_report.md`
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
- [x] Download YOLOv11n PyTorch model from Ultralytics
- [x] Download Depth Anything V2 Small PyTorch model
- [x] Download Whisper Tiny model from OpenAI
- [x] Download Piper TTS model and voice files
- [x] Download openWakeWord models
- [x] Download **Gemma 3n E2B** (5B parameters, 2B effective footprint)
- [x] Organize models in `/models` directory
- [x] Document model sources and licenses in `docs/model_credits.md`

### 3.2 Vision Model Conversion (TensorRT)
- [x] Implement `tools/conversion/convert_yolo.py`
  - [x] Direct PyTorch to TensorRT conversion using Ultralytics export
  - [x] Convert YOLOv11n to TensorRT engine (FP16) - skip ONNX intermediate step
  - [x] Validate output accuracy (mAP drop <2%)
  - [x] Benchmark inference time (target: <50ms on Jetson)
  - [x] Save engine to `models/yolo_trt/YOLOv11n_fp16.engine`

- [x] Implement `tools/convert_depth.py`
  - [x] Export Depth Anything V2 Small to ONNX format
  - [x] Convert ONNX to TensorRT engine (FP16)
  - [x] Validate depth map quality and compare to original model
  - [x] Benchmark inference time (target: <35ms for 30+ FPS)
  - [x] Save engine to `models/depth_trt/depth_anything_v2_s_fp16.engine`

- [x] Document conversion process in `docs/guides/model_conversion.md`

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
- [ ] ALTERNATIVE: Implement `tools/conversion/convert_whisper.py`
  - [ ] Export Whisper Tiny to ONNX
  - [ ] Convert to TensorRT (FP16)
  - [ ] Validate Word Error Rate (WER)
  - [ ] Compare performance with faster-whisper
- [ ] Set up Piper TTS
  - [ ] Download Piper binary and voice files
  - [ ] Test synthesis quality
  - [ ] Benchmark latency (target: <500ms for 20 words)

### 3.4 Multimodal LLM Setup (Gemma 3n E2B)
- [ ] Install HuggingFace Transformers 4.53.0+
- [ ] Download Gemma 3n E2B model from HuggingFace:
  - [ ] Model: `google/gemma-3n-e2b` (5B parameters, 2B effective footprint)
  - [ ] Verify model supports multimodal input (text, audio, image)
  - [ ] Test model loading with transformers library
- [ ] Test inference performance:
  - [ ] Text-only inference (target: <2s for typical response)
  - [ ] Multimodal inference (target: <4s for complex scene analysis)
  - [ ] Memory usage (constant 2GB VRAM footprint)
- [ ] Create preprocessing pipelines:
  - [ ] Image preprocessing (256x256, 512x512, 768x768 resolutions)
  - [ ] Audio preprocessing (6.25 tokens/second encoding)
  - [ ] Text tokenization with 32K context window
- [ ] Save model to `models/gemma_3n_e2b/`
- [ ] Create `docs/guides/gemma_3n_setup.md` with setup instructions

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

# manual_tests/test_tools/conversion/c_realtime.py
- Live camera feed test
- Visual validation of detections
- FPS monitoring over 5 minutes
```

### 4.2 Depth Estimation Node
- [ ] Implement `depth_estimator.py`
  - [ ] Load Depth Anything V2 Small TensorRT engine from Phase 3.2
  - [ ] Subscribe to `/camera/undistorted`
  - [ ] Run inference using TensorRT runtime
  - [ ] Publish depth maps to `/perception/depth`
  - [ ] Add depth colormap visualization
  - [ ] Implement depth range normalization
- [ ] Benchmark inference time (target: 30+ FPS)
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
- Working depth estimation node (30+ FPS)
- Working point cloud generation
- Perception pipeline integrated and tested
- Performance benchmarks documented

---

## Phase 5: Audio Detection Pipeline (Week 7)

### 5.1 Audio Capture & Playback Infrastructure
- [ ] Implement `audio_capture_node.py`
  - [ ] PyAudio initialization with USB microphone configuration
  - [ ] Continuous audio streaming at 16 kHz
  - [ ] Publish to `/audio/raw` topic (audio_msgs/AudioData)
  - [ ] Circular buffer management (5-second rolling buffer)
  - [ ] USB device health monitoring and auto-reconnection
  - [ ] Configurable sample rate and channels from `config/audio_config.yaml`
- [ ] Implement `audio_playback_node.py`
  - [ ] PyAudio initialization with USB speakers configuration
  - [ ] Subscribe to `/audio/tts_output`
  - [ ] Queue-based playback system with priority handling
  - [ ] Handle playback interruptions (emergency stop, new commands)
  - [ ] Volume normalization and audio quality optimization
  - [ ] Monitor playback errors and device status
- [ ] Test audio latency (target: <200ms round-trip)
- [ ] Test simultaneous capture/playback without feedback
- [ ] Test USB device reconnection and hot-swapping

**Tests Required**:
```python
# tests/test_audio_capture.py
- Test device initialization and configuration
- Test audio streaming at 16 kHz with quality metrics
- Test circular buffer management under load
- Test USB device disconnection/reconnection handling
- Test multi-threading performance

# tests/test_audio_playback.py
- Test device initialization and speaker configuration
- Test audio playback with various formats
- Test queue management with priority systems
- Test interruption handling and graceful recovery
- Test volume control and audio quality
```

### 5.2 Wake Word Detection ("Hey Jarvis")

- [ ] Install openWakeWord library and dependencies
- [ ] Implement `wake_word_detector_node.py`
  - [ ] Load wake word model (ONNX format for "Hey Jarvis")
  - [ ] Subscribe to `/audio/raw` with real-time processing
  - [ ] Run continuous detection in dedicated thread
  - [ ] Publish to `/audio/wake_word_detected` (std_msgs/Bool + confidence)
  - [ ] Add detection confidence threshold (configurable, default: 0.6)
  - [ ] Implement cooldown period to prevent multiple triggers (default: 2 seconds)
  - [ ] Always-on operation with minimal resource footprint
- [ ] Train or fine-tune custom wake word model for "Hey Jarvis"
- [ ] Test false positive rate (target: <1 per hour in quiet environment)
- [ ] Test detection latency (target: <100ms from word completion)
- [ ] Optimize for ultra-low CPU usage (target: <5% continuously)
- [ ] Test robustness across different voices, accents, and distances

**Tests Required**:
```python
# tests/test_wake_word.py
- Test model loading and initialization performance
- Test detection accuracy on diverse audio samples
- Benchmark continuous CPU usage (target: <5%)
- Measure detection latency and confidence scores
- Test cooldown period functionality

# manual_tests/test_wake_word_robustness.py
- Test with multiple speakers (5+ people, various ages/genders)
- Test at different distances (1m, 3m, 5m from microphone)
- Test in various noise levels (quiet, TV background, conversation)
- Test false positive rate with similar-sounding words
- Test different speaking speeds and volumes
- Measure detection accuracy in different room acoustics
```

### 5.3 Voice Activity Detection (VAD)

- [ ] Install VAD libraries (webrtcvad and/or silero-vad)
- [ ] Implement `vad_node.py`
  - [ ] Load VAD model (webrtcvad primary, silero-vad alternative)
  - [ ] Subscribe to `/audio/raw` and `/audio/wake_word_detected`
  - [ ] Dynamic enable/disable based on system state
  - [ ] Real-time speech/silence detection with configurable thresholds
  - [ ] Publish voice activity status to `/audio/vad_status`
  - [ ] Maintain audio buffers for complete utterance capture
  - [ ] Implement smart timeout for silence detection (default: 2 seconds)
  - [ ] Prevent feedback loops during robot speech output
- [ ] Configure VAD sensitivity for different environments
- [ ] Test speech segmentation accuracy (target: >95% correct segmentation)
- [ ] Test detection latency (target: <50ms)
- [ ] Optimize resource usage (target: <2% CPU when active)
- [ ] Test with various background noise levels

**Tests Required**:
```python
# tests/test_vad.py
- Test VAD model loading and initialization
- Test speech/silence detection accuracy
- Test dynamic enable/disable functionality
- Benchmark VAD processing latency and CPU usage
- Test audio buffer management under various conditions

# manual_tests/test_vad_robustness.py
- Test speech segmentation with different speaking styles
- Test performance in noisy environments (SNR: 20dB, 10dB, 5dB)
- Test timeout behavior with long pauses
- Test buffer overflow handling with very long utterances
- Test feedback loop prevention during robot speech
```

### 5.4 Enhanced Speech-to-Text with VAD Integration

- [ ] Set up faster-whisper (PRIMARY) with optimizations
  - [ ] Install faster-whisper library with CUDA support
  - [ ] Test model loading time and memory usage
  - [ ] Configure INT8 quantization for performance
- [ ] Implement enhanced `stt_node.py`
  - [ ] Load Whisper model (faster-whisper or TensorRT engine)
  - [ ] Subscribe to `/audio/wake_word_detected` and `/audio/vad_status`
  - [ ] Integrate VAD for intelligent audio capture
  - [ ] Capture complete utterances with pre/post-speech buffers
  - [ ] Run transcription on VAD-segmented audio
  - [ ] Publish to `/audio/transcribed_text` (std_msgs/String + confidence)
  - [ ] Add comprehensive timeout handling (max 15 seconds)
  - [ ] Implement noise suppression and audio preprocessing
  - [ ] Handle overlapping wake word detections gracefully
- [ ] Test transcription accuracy (target: WER <8% for clean speech)
- [ ] Test with various accents, speaking styles, and command types
- [ ] Benchmark inference time (target: <2s for 5s audio, real-time factor <0.4x)
- [ ] Optimize for low latency and consistent performance

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

### 5.5 Text-to-Speech (Piper) with State Management

- [ ] Install Piper TTS with ONNX runtime support
- [ ] Download and validate voice model (recommended: en_US-lessac-medium)
- [ ] Implement enhanced `tts_node.py`
  - [ ] Load Piper model with optimized ONNX inference
  - [ ] Subscribe to `/audio/tts_request` (std_msgs/String + priority)
  - [ ] Synthesize speech with real-time streaming
  - [ ] Publish to `/audio/tts_output` (audio_msgs/AudioData)
  - [ ] Implement speech rate and volume control via parameters
  - [ ] Add volume normalization and audio quality enhancement
  - [ ] Coordinate with VAD node to signal speech start/end
  - [ ] Handle speech interruptions and queue management
- [ ] Test voice quality and naturalness (subjective evaluation)
- [ ] Test synthesis latency (target: <500ms for 20 words)
- [ ] Test various sentence types, lengths, and punctuation handling
- [ ] Optimize for consistent low latency performance

**Tests Required**:
```python
# tests/test_tts.py
- Test model loading and ONNX inference setup
- Test synthesis quality on diverse text samples (20+ sentences)
- Benchmark latency across different text lengths
- Test voice parameter control (rate, volume, pitch)
- Test concurrent requests and queue management

# manual_tests/test_tts_integration.py
- Test various sentence types (questions, commands, statements)
- Test punctuation and special character handling
- User feedback on voice naturalness (5+ participants)
- Test long text synthesis (100+ words) with streaming
- Test coordination with VAD for feedback loop prevention
```

### 5.6 Comprehensive Audio Detection Pipeline Integration

- [ ] Implement `audio_detection_pipeline.py` - Central state machine coordinator
  - [ ] Coordinate all audio nodes (capture, wake word, VAD, STT, TTS, playback)
  - [ ] Implement the 6-step pipeline state machine:
    - Step 1: Always listening for "Hey Jarvis" (LISTENING_WAKE_WORD)
    - Step 2: VAD activation for speech capture (CAPTURING_COMMAND)
    - Step 3: Whisper transcription (PROCESSING_SPEECH)
    - Step 4: Command forwarding to cognitive core (COMMAND_SENT)
    - Step 5: VAD disabled during robot speech (ROBOT_SPEAKING)
    - Step 6: Return to wake word listening (READY_FOR_NEXT)
  - [ ] Implement feedback loop prevention logic
  - [ ] Handle state transitions and error recovery
  - [ ] Manage audio buffers across pipeline stages
  - [ ] Publish pipeline status to `/audio/pipeline_status`
- [ ] Create `launch/audio_detection_pipeline_launch.py`
  - [ ] Launch all audio nodes in correct order
  - [ ] Set up inter-node communication topics
  - [ ] Configure parameters from `config/audio_config.yaml`
- [ ] Test complete audio detection flow:
  - [ ] Wake word → VAD activation → Speech capture → Transcription
  - [ ] TTS request → Speech synthesis → Playback → VAD disable/enable
  - [ ] Error recovery and timeout handling
- [ ] Implement comprehensive integration tests:
  - [ ] Test end-to-end latency (target: <4 seconds wake-to-response)
  - [ ] Test feedback loop prevention effectiveness
  - [ ] Test concurrent user interruptions and overlapping commands
  - [ ] Test long-duration operation (1 hour+ continuous operation)
  - [ ] Test resource usage under sustained load
- [ ] Optimize pipeline performance:
  - [ ] Minimize audio buffer copying between nodes
  - [ ] Optimize thread management and CPU affinity
  - [ ] Tune timeout values and detection thresholds
  - [ ] Implement smart power management for low-activity periods

**Integration Tests Required**:
```python
# tests/test_audio_detection_pipeline.py
- Test complete pipeline state machine transitions
- Test feedback loop prevention during robot speech
- Test error recovery from node failures
- Test concurrent wake word detections handling
- Benchmark end-to-end latency and resource usage

# manual_tests/test_full_audio_interaction.py
- Test realistic conversation scenarios (10+ interactions)
- Test interruption handling (user speaks while robot responds)
- Test multiple consecutive commands without wake words
- Test pipeline recovery from audio device disconnections
- Test performance degradation over extended use (2+ hours)
- Test with multiple users and overlapping speech
```

**Deliverables**:
- Complete audio detection pipeline with 6-step state machine
- Wake word detection for "Hey Jarvis" (<5% CPU continuously)
- Voice Activity Detection with feedback loop prevention
- Speech-to-text transcription (WER <8% for clean speech)
- Text-to-speech synthesis with natural voice output
- Central pipeline coordinator with state management
- Comprehensive integration tests and performance benchmarks
- End-to-end latency <4 seconds (wake word to robot response)
- 1+ hour continuous operation capability
- Audio device hot-swap and error recovery
- Documentation of all performance metrics and tuning parameters

---

## Phase 5.5: Enhanced Multimodal Audio Processing (Week 7.5)

### 5.5.1 Direct Audio Processing for Gemma 3n
- [ ] Implement `multimodal_audio_processor.py`
  - [ ] Subscribe to `/audio/raw` for direct audio input
  - [ ] Implement audio preprocessing for Gemma 3n (6.25 tokens/second encoding)
  - [ ] Create audio buffering for multimodal processing
  - [ ] Publish encoded audio to `/audio/gemma_3n_ready`
  - [ ] Support environmental sound analysis
  - [ ] Handle multi-speaker scenario processing
- [ ] Create audio encoding utilities:
  - [ ] Audio segmentation (1-10 second clips)
  - [ ] Format conversion for Gemma 3n input
  - [ ] Quality validation and preprocessing
  - [ ] Real-time streaming capabilities

### 5.5.2 Environmental Audio Analysis
- [ ] Implement environmental sound classification:
  - [ ] Background noise detection (crowd, traffic, music)
  - [ ] Speaker identification for multi-speaker scenarios
  - [ ] Emotional tone analysis from voice characteristics
  - [ ] Urgency detection from audio cues
- [ ] Test environmental audio understanding:
  - [ ] Ambient sound classification accuracy (target: >80%)
  - [ ] Multi-speaker scenario handling
  - [ ] Background noise robustness
  - [ ] Real-time processing capability

### 5.5.3 Audio Context Integration
- [ ] Implement audio context coordination:
  - [ ] Coordinate between traditional ASR and direct audio processing
  - [ ] Intelligent mode switching based on command complexity
  - [ ] Audio context preservation for multimodal reasoning
  - [ ] Latency optimization for real-time operation
- [ ] Test integrated audio processing:
  - [ ] Compare ASR vs direct audio processing accuracy
  - [ ] Test mode switching reliability
  - [ ] Benchmark end-to-end latency with multimodal input
  - [ ] Validate audio context preservation

**Deliverables**:
- Direct audio processing pipeline for Gemma 3n
- Environmental sound analysis capabilities
- Coordinated ASR and multimodal audio processing
- Audio context integration with visual processing
- Comprehensive multimodal audio testing

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

## Phase 7: Gemma 3n Multimodal Cognitive Core (Weeks 9-10)

### 7.1 Gemma 3n E2B Setup
- [ ] Install HuggingFace Transformers 4.53.0+ (Phase 3.4 continuation)
- [ ] Load Gemma 3n E2B model from `models/gemma_3n_e2b/`
- [ ] Verify multimodal capabilities:
  - [ ] Text processing (baseline functionality)
  - [ ] Image understanding (256x256, 512x512, 768x768 resolutions)
  - [ ] Audio processing (6.25 tokens/second encoding)
  - [ ] 32K context window across all modalities
- [ ] Create `cognitive_core_nodes` ROS2 package
- [ ] Implement model loading utilities with constant 2GB footprint
- [ ] Test model initialization time (target: <30s first load)

### 7.2 Multimodal Data Pipeline
- [ ] Implement `multimodal_processor.py`
  - [ ] Subscribe to `/audio/transcribed_text` (text input)
  - [ ] Subscribe to `/audio/raw` (direct audio input for complex scenarios)
  - [ ] Subscribe to `/camera/undistorted` (image input)
  - [ ] Subscribe to world state from SLAM/perception
  - [ ] Coordinate multimodal input processing
  - [ ] Manage input prioritization and batching
- [ ] Implement `vision_processor.py`
  - [ ] Image preprocessing for Gemma 3n (resize, normalize)
  - [ ] Support multiple resolutions based on task complexity
  - [ ] Encode images to 256 tokens per image
  - [ ] Handle real-time snapshot capture for goal assessment
- [ ] Implement `audio_processor.py`
  - [ ] Audio preprocessing for Gemma 3n (6.25 tokens/second)
  - [ ] Direct audio encoding bypassing traditional ASR
  - [ ] Handle environmental sound analysis
  - [ ] Support multi-speaker scenario processing

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

### 7.3 Gemma 3n Interface Node
- [ ] Implement `gemma_3n_multimodal_interface.py`
  - [ ] Load Gemma 3n E2B model with HuggingFace Transformers
  - [ ] Subscribe to `/audio/transcribed_text` (text input)
  - [ ] Subscribe to `/audio/raw` (direct audio for complex scenarios)
  - [ ] Subscribe to `/camera/snapshot` (visual input)
  - [ ] Subscribe to world state from serializer
  - [ ] Implement multimodal prompt construction
  - [ ] Run multimodal inference with 32K context
  - [ ] Parse structured JSON output (enhanced intent format)
  - [ ] Publish intent to `/cognitive/multimodal_intent` (custom msg)
  - [ ] Publish natural language response to `/audio/tts_request`
  - [ ] Implement goal assessment capabilities
  - [ ] Support real-time strategy evaluation
- [ ] Design enhanced multimodal prompt template:
  ```
  You are an AI robot assistant with multimodal capabilities.
  Current robot state: {world_state}
  Current image: <image>
  Audio context: <audio>
  User command: {transcribed_text}

  Task: Provide structured intent and assess current situation.
  Output format: {
    "action": "...", "target": "...", "parameters": {...},
    "visual_confirmation": true/false,
    "goal_assessment": "...",
    "strategy_evaluation": "..."
  }
  ```
- [ ] Implement mode switching:
  - [ ] Text-only mode for simple commands (fast inference)
  - [ ] Multimodal mode for complex scene understanding
- [ ] Test conversation history integration (last 3 exchanges)

### 7.4 Enhanced Intent Message Definition
- [ ] Create custom ROS2 message: `cognitive_msgs/MultimodalIntent`
  ```
  string action  # navigate, pickup, search, stop, etc.
  string target  # object name or location
  string[] parameters  # additional parameters
  float32 confidence  # Gemma 3n confidence (0-1)
  bool visual_confirmation  # visual goal verification available
  string audio_context  # environmental audio analysis
  string strategy_assessment  # navigation/approach evaluation
  string goal_status  # completion assessment
  ```
- [ ] Test message serialization and ROS2 integration

### 7.5 Multimodal Testing
- [ ] Create test dataset of multimodal commands (100+ examples):
  - Text-only: "go forward", "stop", "find the red ball"
  - Visual: "are the lights on?", "what do you see?", "am I close to the target?"
  - Audio context: "is there background noise?", "who is speaking?"
  - Complex multimodal: "navigate to the person who just called my name"
- [ ] Test Gemma 3n multimodal understanding:
  - [ ] Text command accuracy (target: >95% for clear commands)
  - [ ] Visual scene understanding (target: >85% object recognition)
  - [ ] Audio context analysis (target: >80% environmental classification)
  - [ ] Goal assessment capabilities (target: >90% completion verification)
  - [ ] JSON output format compliance
  - [ ] Context awareness (uses world state + visual + audio)
- [ ] Test conversational abilities:
  - [ ] Multi-turn dialogue with visual context
  - [ ] Clarification questions with scene understanding
  - [ ] Status updates with visual confirmation
- [ ] Benchmark multimodal performance:
  - [ ] Text-only inference (target: <2s)
  - [ ] Multimodal inference (target: <4s)
  - [ ] Memory footprint (constant 2GB VRAM)
  - [ ] Model loading time (target: <30s)

### 7.6 Simplified Memory Management
- [ ] Implement `selective_memory_manager.py`
  - [ ] Monitor system RAM usage
  - [ ] Manage Gemma 3n model loading (constant 2GB footprint)
  - [ ] Coordinate with perception models for optimal performance
  - [ ] Implement intelligent model prioritization:
    - Keep Gemma 3n loaded during complex tasks
    - Temporarily reduce perception model frequency if needed
    - Maintain constant memory usage for predictable performance
- [ ] Test simplified memory management:
  - [ ] Verify stable 2GB Gemma 3n footprint
  - [ ] Test concurrent operation with perception models
  - [ ] Measure overall system stability
- [ ] Update `config/memory_management.yaml`:
  ```yaml
  gemma_3n:
    constant_footprint: 2GB  # Predictable memory usage
    loading_timeout: 30s     # Maximum model load time

  thresholds:
    warning: 0.75   # 75% RAM usage (improved margin)
    critical: 0.85  # 85% RAM usage
    emergency: 0.90 # 90% RAM usage

  strategies:
    warning: log_warning
    critical: reduce_perception_frequency
    emergency: emergency_mode_motors_only
  ```

**Deliverables**:
- Working Gemma 3n E2B multimodal integration
- Multimodal data pipeline (text, audio, vision)
- Enhanced intent message with multimodal context
- Goal assessment and strategy evaluation capabilities
- Simplified memory management (constant 2GB footprint)
- Comprehensive multimodal testing (>90% accuracy)
- Visual scene understanding and audio context analysis

---

## Phase 8: Behavioral Architecture (Weeks 11-12)

### 8.1 BehaviorTree.CPP Setup
- [ ] Install BehaviorTree.CPP library
- [ ] Create `behavioral_nodes` ROS2 package
- [ ] Set up BehaviorTree.CPP ROS2 integration
- [ ] Create blackboard data structure
- [ ] Test basic behavior tree execution

### 8.2 Enhanced Multimodal Command Router
- [ ] Implement `adaptive_command_router.py`
  - [ ] Subscribe to `/audio/transcribed_text`
  - [ ] Subscribe to `/audio/raw` (for direct audio context)
  - [ ] Subscribe to `/camera/snapshot` (for visual context)
  - [ ] Classify command complexity and modality requirements:
    - Simple: direct motor commands (stop, forward, backward, turn)
    - Text-only complex: requires Gemma 3n text processing (find X, navigate to Y)
    - Multimodal: requires visual/audio analysis (what do you see?, are lights on?, navigate to the person speaking)
  - [ ] Route commands intelligently:
    - Simple commands → directly to behavior tree
    - Text-only complex → Gemma 3n text mode
    - Multimodal commands → Gemma 3n multimodal mode
  - [ ] Coordinate multimodal data collection before routing
  - [ ] Log routing decisions and modality selection
- [ ] Create enhanced command mapping:
  ```python
  SIMPLE_COMMANDS = {
    "stop": {"action": "stop"},
    "go forward": {"action": "move", "direction": "forward"},
    "turn left": {"action": "turn", "direction": "left"},
    "turn right": {"action": "turn", "direction": "right"},
    "go back": {"action": "move", "direction": "backward"}
  }

  MULTIMODAL_TRIGGERS = [
    "what do you see", "are the lights", "is the", "check if",
    "look at", "find the person", "navigate to the", "am I close"
  ]
  ```
- [ ] Test multimodal command classification accuracy

### 8.3 Enhanced Multimodal Blackboard Implementation
- [ ] Implement enhanced blackboard manager
- [ ] Define enhanced blackboard schema with multimodal context:
  ```
  # Traditional robot state
  - robot_pose (geometry_msgs/PoseStamped)
  - robot_orientation (from IMU)
  - semantic_map (list of objects with poses)
  - current_mission (string)
  - current_goal (geometry_msgs/PoseStamped)
  - navigation_status (enum: idle, moving, stuck, arrived)
  - audio_status (enum: listening, processing, speaking)
  - system_health (dict: CPU, GPU, RAM, temperature)
  - error_log (list of recent errors)

  # NEW: Multimodal context
  - visual_scene_state (latest Gemma 3n image analysis)
  - audio_environment_state (ambient sound analysis)
  - goal_completion_status (visual verification results)
  - strategy_assessment (navigation approach evaluation)
  - multimodal_confidence_scores (reliability per modality)
  - scene_change_detection (environmental change notifications)
  ```
- [ ] Implement enhanced blackboard update subscribers:
  - [ ] Subscribe to `/odom` → update robot_pose
  - [ ] Subscribe to `/imu/data` → update orientation
  - [ ] Subscribe to `/perception/objects` → update semantic_map
  - [ ] Subscribe to `/cognitive/multimodal_intent` → update current_mission
  - [ ] Subscribe to Gemma 3n multimodal analysis results
  - [ ] Update multimodal context in real-time
- [ ] Test enhanced blackboard updates (latency <10ms)

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

### 8.7 Enhanced Multimodal Dialogue Manager
- [ ] Implement `multimodal_dialogue_manager.py`
  - [ ] Subscribe to `/cognitive/multimodal_intent` and enhanced blackboard
  - [ ] Generate contextually aware status updates:
    - "I can see the red ball, navigating toward it"
    - "I hear background noise, but I'm continuing"
    - "I've reached the target - visual confirmation successful"
  - [ ] Generate multimodal clarification questions:
    - "I see multiple objects, which one do you mean?" (with visual context)
    - "I can't see that clearly, should I move closer?"
    - "There's a lot of noise, can you repeat that?"
  - [ ] Publish enhanced responses to `/audio/tts_request`
  - [ ] Implement multimodal dialogue state machine with goal assessment

### 8.8 Multimodal Behavior Tree Nodes
- [ ] Implement enhanced multimodal behavior nodes:
  - [ ] `VisualGoalVerification` - Use Gemma 3n to verify task completion
  - [ ] `SceneAssessment` - Analyze current visual scene for navigation
  - [ ] `StrategyEvaluator` - Evaluate current approach using vision
  - [ ] `AudioContextMonitor` - Monitor environmental audio changes
  - [ ] `MultimodalStuckDetection` - Visual confirmation of stuck state
  - [ ] `GoalProgressMonitor` - Continuous visual progress assessment
- [ ] Create multimodal behavior tree templates:
  ```xml
  <BehaviorTree ID="MultimodalNavigation">
    <Sequence>
      <VisualGoalVerification/>
      <SceneAssessment/>
      <StrategyEvaluator/>
      <Parallel>
        <NavigateToTarget/>
        <GoalProgressMonitor/>
        <AudioContextMonitor/>
      </Parallel>
      <VisualGoalVerification final="true"/>
    </Sequence>
  </BehaviorTree>
  ```
- [ ] Test multimodal behavior integration
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
- Working multimodal behavior tree system
- Enhanced command routing (simple vs text-only vs multimodal)
- Multimodal navigation behaviors with visual goal verification
- Enhanced dialogue management with multimodal context
- Goal assessment and strategy evaluation capabilities
- Behavior tree executor with multimodal node support
- Comprehensive integration tests with visual and audio context
- Real-time scene monitoring and adaptive behavior

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
  - [ ] Monitor RAM usage (with Gemma 3n constant 2GB tracking)
  - [ ] Monitor temperatures (CPU, GPU, thermal zones)
  - [ ] Monitor disk usage
  - [ ] Monitor network stats
  - [ ] Monitor Gemma 3n model performance metrics
  - [ ] Publish to `/system/metrics` topic (10 Hz)
- [ ] Implement `node_monitor.py`
  - [ ] Track active ROS2 nodes
  - [ ] Monitor multimodal topic publication rates
  - [ ] Detect node failures
  - [ ] Log node restarts
  - [ ] Monitor Gemma 3n cognitive core health
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
  float32 gemma_3n_vram_mb  # Should be constant ~2048MB
  float32 gemma_3n_inference_latency_ms
  float32 multimodal_processing_fps

  # monitoring_msgs/NodeStatus.msg
  string[] active_nodes
  string[] failed_nodes
  bool gemma_3n_cognitive_core_healthy
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
  - [ ] Multimodal interaction panel
  - [ ] Gemma 3n cognitive status panel
  - [ ] Command input panel
  - [ ] Log viewer panel
- [ ] Implement CSS styling (`style.css`)
  - [ ] Responsive design (mobile-friendly)
  - [ ] Dark theme (easier on eyes)
  - [ ] Status indicators (colors for health)
  - [ ] Animation for live updates
  - [ ] Multimodal processing indicators
- [ ] Implement JavaScript functionality (`app.js`)
  - [ ] WebSocket connection management
  - [ ] Real-time data updates
  - [ ] Camera feed display (MJPEG stream)
  - [ ] Interactive map rendering (Canvas/SVG)
  - [ ] System metrics charts (Chart.js)
  - [ ] Gemma 3n performance metrics
  - [ ] Multimodal interaction visualization
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
  - updateGemma3nStatus()
  - updateMultimodalProcessing()
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
- [ ] Implement multimodal processing visualization
  - [ ] Show current modalities being processed (text/audio/vision)
  - [ ] Display Gemma 3n context window usage (32K tokens)
  - [ ] Show multimodal token encoding rates
  - [ ] Visualize cross-modal attention patterns
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
  - [ ] Gemma 3n VRAM usage (constant 2GB indicator)
  - [ ] Multimodal processing latency charts

### 9.5 Control Interface
- [ ] Implement multimodal command input
  - [ ] Text input field
  - [ ] Audio recording button (voice commands)
  - [ ] Image upload for visual queries
  - [ ] Submit button
  - [ ] Command history (last 10 commands)
  - [ ] Send to `/api/command` endpoint
  - [ ] Display robot response (multimodal)
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
  - [ ] Gemma 3n model parameters
  - [ ] Multimodal processing preferences
  - [ ] Save settings to browser localStorage
- [ ] Implement system controls
  - [ ] Start/stop specific nodes (via ROS2 lifecycle)
  - [ ] Enable/disable Gemma 3n cognitive core
  - [ ] Enable/disable perception models
  - [ ] Switch multimodal processing modes
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
- System monitoring node with Gemma 3n tracking
- Complete web dashboard with:
  - Live camera feed with overlays
  - Interactive 2D map
  - System metrics visualization
  - Multimodal command input interface (text/audio/image)
  - Gemma 3n cognitive status panel
  - Multimodal processing visualization
  - Emergency stop button
  - Real-time logs
- Mobile-responsive design
- Performance optimized (<200 MB RAM)
- Documentation for multimodal web interface usage

---

## Phase 10: System Integration & Testing (Weeks 14-15)

### 9.1 Full System Launch
- [ ] Create `launch/full_system_launch.py`
  - [ ] Launch all perception nodes
  - [ ] Launch multimodal audio pipeline nodes
  - [ ] Launch localization and SLAM
  - [ ] Launch Gemma 3n cognitive core (lazy loaded)
  - [ ] Launch enhanced multimodal behavioral architecture
  - [ ] Launch web interface with multimodal support
  - [ ] Launch monitoring nodes
- [ ] Create launch configuration options:
  - [ ] `--minimal` - motors + wake word only (emergency mode)
  - [ ] `--no-gemma` - skip Gemma 3n loading (simple commands only)
  - [ ] `--no-web` - disable web interface (save RAM)
  - [ ] `--perception-only` - camera + perception for testing
  - [ ] `--text-only` - disable multimodal (text commands only)
- [ ] Test full system startup (<30 seconds)
- [ ] Test graceful shutdown (all nodes stop cleanly)
- [ ] Test web interface access during startup

### 9.2 End-to-End Testing
- [ ] Create multimodal test scenarios:

  **Scenario 1: Simple Navigation**
  - [ ] User says wake word
  - [ ] User says "go forward"
  - [ ] Robot moves forward for 1 meter
  - [ ] Robot stops and says "Done"

  **Scenario 2: Object Finding with Multimodal Input**
  - [ ] User says wake word
  - [ ] User says "find the red ball" OR shows image of red ball
  - [ ] Robot rotates to scan environment
  - [ ] Robot detects ball with YOLO
  - [ ] Robot navigates to ball
  - [ ] Robot says "I found the red ball"

  **Scenario 3: Complex Multimodal Command**
  - [ ] User says wake word
  - [ ] User says "bring me that cup on the table" with image upload
  - [ ] Gemma 3n processes text+vision input (multimodal)
  - [ ] Robot navigates to table
  - [ ] Robot stops near cup
  - [ ] Robot says "I'm at the cup, but I can't pick it up yet"

  **Scenario 4: Visual Question Answering**
  - [ ] User says wake word
  - [ ] User asks "what do you see?" while pointing camera at objects
  - [ ] Gemma 3n processes camera feed (vision tokens)
  - [ ] Robot describes visible objects
  - [ ] Robot provides contextual information

  **Scenario 5: Audio-Visual Context**
  - [ ] User plays audio while robot sees environment
  - [ ] User asks "relate what you hear to what you see"
  - [ ] Gemma 3n processes audio+vision simultaneously
  - [ ] Robot provides contextual multimodal response

  **Scenario 6: Stuck Recovery**
  - [ ] Robot encounters obstacle while navigating
  - [ ] Stuck detector triggers after 3 seconds
  - [ ] Robot backs up and rotates
  - [ ] Robot retries navigation
  - [ ] Robot says "I got stuck but found another way"

  **Scenario 7: Multimodal Clarification**
  - [ ] User says wake word
  - [ ] User says "go to the ball" with ambiguous image
  - [ ] Multiple balls detected
  - [ ] Robot asks "Which ball? The red one or the blue one?"
  - [ ] User responds with text OR points in image
  - [ ] Robot navigates to specified ball

- [ ] Test each scenario 5+ times
- [ ] Measure success rate (target: >80%)
- [ ] Test multimodal context window efficiency (32K tokens)
- [ ] Verify Gemma 3n constant 2GB VRAM usage
- [ ] Document failure modes
- [ ] Monitor via web interface during tests
- [ ] Verify web interface shows correct real-time multimodal data

### 9.3 Performance Profiling
- [ ] Profile full system under load:
  - [ ] CPU usage per core (target: <80% average)
  - [ ] GPU usage (target: <90%)
  - [ ] RAM usage (target: <7.5 GB)
  - [ ] Gemma 3n VRAM usage (should be constant 2GB)
  - [ ] Temperature (target: <80°C sustained)
- [ ] Measure multimodal latencies:
  - [ ] Wake word to ASR complete (<3 seconds)
  - [ ] ASR to Gemma 3n text processing (<2 seconds)
  - [ ] Image processing to Gemma 3n vision tokens (<1 second)
  - [ ] Audio encoding to Gemma 3n audio tokens (<0.5 seconds)
  - [ ] Multimodal context processing (<3 seconds for 32K tokens)
  - [ ] Gemma 3n response generation (<5 seconds for complex)
  - [ ] Response to TTS start (<1 second)
  - [ ] Command to motor action (<2 seconds for simple)
  - [ ] Object detection to navigation start (<5 seconds)
- [ ] Profile individual nodes:
  - [ ] Identify multimodal processing bottlenecks
  - [ ] Optimize hot paths in Gemma 3n pipeline
  - [ ] Reduce memory allocations
  - [ ] Monitor token encoding efficiency
- [ ] Create multimodal performance dashboard

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
- Optimized multimodal system (20-30% performance improvement)
- Complete documentation (user + developer) with Gemma 3n guides
- Deployment package (Docker + systemd) with multimodal support
- Final validation report with multimodal performance metrics
- Video demonstrations showcasing multimodal capabilities
- Ready for production use with revolutionary AI assistant features

---

## Phase 12: Advanced Multimodal Features & Polish (Weeks 18-19) [OPTIONAL]

### 11.1 Advanced Navigation with Multimodal Guidance
- [ ] Implement path planning with visual obstacle recognition
- [ ] Add dynamic replanning with multimodal feedback
- [ ] Implement wall following with audio cues
- [ ] Add exploration mode with multimodal environment description
- [ ] Test in complex environments with multimodal command handling

### 11.2 Enhanced Multimodal Dialogue
- [ ] Add multimodal conversation history (text+audio+vision)
- [ ] Implement cross-modal context awareness
- [ ] Add personality to robot responses across modalities
- [ ] Support multi-turn multimodal conversations
- [ ] Add emotion detection from voice tone and facial expressions

### 11.3 Advanced Multimodal Object Interaction (Future Work)
- [ ] Design gripper interface with vision-guided grasping (placeholder)
- [ ] Add multimodal object grasping behaviors to tree
- [ ] Implement visual approach and alignment behaviors
- [ ] Add voice-guided manipulation commands
- [ ] Document integration points for future multimodal hardware

### 12.4 Web Interface Multimodal Enhancements
- [ ] Add interactive map view with multimodal annotations
- [ ] Add multimodal control interface (text/voice/visual commands)
- [ ] Add configuration editor for Gemma 3n parameters
- [ ] Add multimodal mission replay (recorded sessions with all modalities)
- [ ] Add Gemma 3n performance graphs and token usage statistics
- [ ] Add mobile app with multimodal input capabilities (React Native/Flutter)

### 12.5 Multi-Robot Multimodal Coordination (Future Work)
- [ ] Design multimodal communication protocol between robots
- [ ] Add robot discovery with visual/audio identification
- [ ] Implement multimodal task coordination
- [ ] Test collaborative multimodal behaviors with 2 robots (if available)

**Deliverables**:
- Advanced multimodal features implemented
- Revolutionary AI assistant user experience
- Foundation for future multimodal extensions
- Showcase of Gemma 3n capabilities on edge devices

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
   - **Mitigation**: Gemma 3n's constant 2GB VRAM footprint eliminates model swapping complexity
   - **Contingency**: Use lighter perception models, reduce web interface functionality

2. **Multimodal Processing Latency**
   - **Mitigation**: Optimize token encoding pipelines, async processing, efficient context management
   - **Contingency**: Fallback to text-only mode, reduce multimodal context window

3. **Visual Odometry Failure**
   - **Mitigation**: Multiple fallback modes (IMU-only, dead reckoning)
   - **Contingency**: Add wheel encoders to Wave Rover (hardware modification)

4. **Gemma 3n Context Window Overflow**
   - **Mitigation**: Intelligent context pruning, sliding window management, token compression
   - **Contingency**: Reset context periodically, prioritize recent interactions

5. **Thermal Throttling with Multimodal Processing**
   - **Mitigation**: Balance workloads across modalities, efficient scheduling, thermal monitoring
   - **Contingency**: Reduce multimodal complexity, disable vision processing, active cooling fan

### Medium-Risk Items
1. **HuggingFace Transformers Compatibility**
   - **Mitigation**: Use tested versions (4.53.0+), extensive integration testing
   - **Contingency**: Pin dependencies, maintain known-working configuration

2. **Multimodal Token Encoding Efficiency**
   - **Mitigation**: Optimize preprocessing pipelines, batch processing where possible
   - **Contingency**: Reduce token resolution, simplify modality interactions

3. **SLAM Drift Accumulation**
   - **Mitigation**: Frequent loop closure, semantic landmarks
   - **Contingency**: Periodic manual recalibration, external localization (markers)

4. **USB Device Reliability**
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
- Phase 5 (Audio Detection Pipeline): 1.5 weeks
- Phase 6 (SLAM & Localization): 1 week
- Phase 7 (Cognitive Core): 2 weeks
- Phase 8 (Behavioral Architecture): 2 weeks
- Phase 9 (Web Interface & Monitoring): 1 week
- Phase 10 (System Integration): 2 weeks
- Phase 11 (Optimization & Docs): 1.5 weeks
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
- [ ] Robot processes basic visual inputs (image uploads via web interface)
- [ ] Robot can navigate to visible objects (>80% success)
- [ ] Robot provides voice feedback for status
- [ ] System runs for 1+ hour without crashes
- [ ] End-to-end latency <5 seconds for complex commands
- [ ] Safe operation (emergency stop works, no collisions)

### Full System Goals
- [ ] Multimodal command understanding with Gemma 3n (>90% accuracy)
- [ ] Simultaneous text, audio, and vision processing
- [ ] Cross-modal context awareness and reasoning
- [ ] Semantic SLAM with object tracking
- [ ] Stuck detection and recovery (>80% success)
- [ ] Multi-turn multimodal conversations
- [ ] 8+ hour continuous operation
- [ ] RAM usage <7.5 GB (with constant 2GB Gemma 3n VRAM)
- [ ] Temperature <80°C sustained
- [ ] User satisfaction rating >4.5/5 for multimodal interactions

### Performance Targets
| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Wake Word Detection | <100ms | <200ms |
| Speech-to-Text | <2s for 5s audio | <4s |
| Text-to-Speech | <500ms for 20 words | <1s |
| LLM Inference | <3s | <5s |
| Object Detection | 20+ FPS | 15+ FPS |
| Depth Estimation | 30+ FPS | 20+ FPS |
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
