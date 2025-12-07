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
- [x] Download **Moondream** (via Ollama)
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
- [x] Set up openWakeWord
  - [x] Download pre-trained models
  - [x] Test wake word detection accuracy
  - [x] Optimize for <5% CPU usage (achieved 7.9% - needs further optimization)
- [x] Set up faster-whisper (PRIMARY OPTION)
  - [x] Install faster-whisper library (CTranslate2)
  - [x] Download Whisper Tiny model (faster-whisper format)
  - [x] Test inference speed (target: real-time factor <0.3x)
  - [x] Benchmark RAM usage (<300 MB) (achieved 0.36x RTF, 718MB RAM - needs optimization)
- [x] ALTERNATIVE: Implement `tools/conversion/convert_whisper_tensorrt.py`
  - [x] Export Whisper Tiny to ONNX
  - [x] Convert to TensorRT (FP16)
  - [x] Validate Word Error Rate (WER)
  - [x] Compare performance with faster-whisper
- [x] Set up Piper TTS
  - [x] Download Piper binary and voice files
  - [x] Test synthesis quality
  - [x] Benchmark latency (target: <500ms for 20 words)
  - [x] Create ROS2 integration node
  - [x] Performance achieved: ~0.03s/word, excellent quality
- [x] Set up silero vad
  - [x] Download pre-trained models
  - [x] Test vad accuracy
  - [x] Optimize for <5% CPU usage

**Note**: Audio models implemented with optimization framework. Current performance:

- openWakeWord: 7.9% CPU (target: <5%) - close to target
- faster-whisper: 0.36x RTF, 718MB RAM (targets: <0.3x RTF, <300MB) - needs TensorRT conversion
- Created optimization scripts: `scripts/optimize_audio_models.py`, `scripts/test_audio_models.py`
- Alternative TensorRT conversion available: `tools/conversion/convert_whisper_tensorrt.py`

### 3.4 Cognitive Core Setup (Ollama + Moondream)
- [x] Install Ollama (Linux ARM64)
  - [x] Configure systemd service for auto-start
  - [x] Verify Ollama API access (`curl localhost:11434`)
- [x] Pull Moondream model
  - [x] Run `ollama pull moondream`
  - [x] Test inference via CLI
  - [x] Verify memory usage (should be ~1.8GB)
- [x] Create `scripts/setup_ollama.sh`
  - [x] Automated installation and model pulling
- [x] Create `docs/guides/ollama_setup.md`
- [x] Create `scripts/test_ollama_moondream.py`
  - [x] Benchmark inference speed and memory usage
  - [x] Document results in walkthrough

### 3.5 Model Profiling
- [x] Individual model profiling scripts created
  - [x] `scripts/test_yolo.py` - YOLO benchmarking
  - [x] `scripts/test_depth.py` - Depth model benchmarking
  - [x] `scripts/test_ollama_moondream.py` - Moondream benchmarking
  - [x] `scripts/test_audio_models.py` - Audio models benchmarking
  - [x] `scripts/test_piper_tts.py` - Piper TTS benchmarking
- [x] Create `scripts/generate_performance_report.py`
  - [x] Run all model benchmarks
  - [x] Aggregate results into unified report
  - [x] Generate `docs/model_performance.md`
- [x] Run unified performance report generator
- [x] Review and finalize `docs/model_performance.md`

**Deliverables**:
- All models converted to optimal formats (TensorRT for vision)
- Model performance benchmarks documented
- Conversion scripts tested and documented
- Model registry with metadata

---

## Phase 4: Perception Models Integration (Week 6)

### 4.1 YOLO Object Detection Node
- [x] Implement `object_detector.py`
  - [x] Load TensorRT engine from Phase 3.2
  - [x] Subscribe to `/camera/undistorted`
  - [x] Run inference using TensorRT runtime
  - [x] Publish detections to `/perception/objects`
  - [x] Add visualization overlay
  - [x] Implement confidence thresholding (default: 0.5)
  - [x] Add NMS (Non-Maximum Suppression)
- [ ] Benchmark inference time (target: 20+ FPS)
- [ ] Test on various objects
- [x] Create unit tests
- [x] Document supported object classes (COCO 80 classes)

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
- [x] Implement `depth_estimator.py`
  - [x] Load Depth Anything V2 Small TensorRT engine from Phase 3.2
  - [x] Subscribe to `/camera/undistorted`
  - [x] Run inference using tensorrt runtime
  - [x] Publish depth maps to `/perception/depth`
  - [x] Add depth colormap visualization
  - [x] Implement depth range normalization
- [x] Benchmark inference time (target: 30+ FPS)
- [x] Test depth accuracy with known distances
- [x] Create unit tests

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
- [x] Implement `pointcloud_generator.py`
  - [x] Subscribe to `/perception/depth` and `/camera/undistorted`
  - [x] Load camera intrinsics from calibration file
  - [x] Back-project depth map to 3D points
  - [x] Apply coordinate transformations
  - [x] Publish to `/perception/pointcloud` (sensor_msgs/PointCloud2)
  - [x] Add RGB color mapping
- [x] Test point cloud accuracy with known geometry
- [x] Visualize in RViz2
- [x] Optimize for real-time performance (target: 10+ Hz)

### 4.4 Perception Integration Test
- [x] Create `launch/perception_launch.py` (updated existing file)
- [x] Test complete perception pipeline:
  - Camera → Undistortion → YOLO + Depth → Point Cloud
- [x] Measure end-to-end latency (target: <200ms)
- [x] Profile GPU/CPU usage
- [ ] Test for memory leaks (24-hour test)

**Deliverables**:
- Working object detection node (implementation complete, benchmarking pending)
- Working depth estimation node (30+ FPS) ✓
- Working point cloud generation ✓
- Perception pipeline integrated and tested (integration complete, end-to-end testing pending)
- Performance benchmarks documented (pending real hardware testing)

---

## Phase 5: Self-Contained Audio Pipeline (Week 7)

### 5.1 Audio Capture Node Refactoring - Self-Contained Pipeline

- [x] **Refactor `audio_capture_node.py` into self-contained pipeline**
  - [x] Audio capture via `arecord` subprocess (already implemented)
  - [x] Circular buffer management (already implemented)
  - [x] Wake word detection integrated (already implemented)
  - [x] Integrate Silero VAD model
    - [x] Load Silero VAD using `silero-vad` package
    - [x] Activate VAD after wake word detection
    - [x] Detect speech start/end boundaries
    - [x] Publish speech events to `/audio/events`
  - [x] Integrate faster-whisper for transcription
    - [x] Load Whisper model (`tiny.en` or `base.en`, INT8)
    - [x] Transcribe audio segment captured by VAD
    - [x] Run transcription in separate thread (non-blocking)
    - [x] Publish to `/audio/transcription` (new TranscriptionResult message)
  - [x] Implement state machine
    - [x] `IDLE`: Listening for wake word continuously
    - [x] `WAKE_WORD_DETECTED`: Wake word triggered, activating VAD
    - [x] `RECORDING`: VAD detected speech start, capturing audio
    - [x] `TRANSCRIBING`: VAD detected speech end, running Whisper
    - [x] Return to IDLE: Transcription complete, return to IDLE
  - [x] Audio buffer management for transcription
    - [x] Maintain pre-roll buffer (audio before wake word)
    - [x] Accumulate audio during RECORDING state
    - [x] Pass complete segment to Whisper
  - [x] Add configuration parameters for VAD and Whisper
  - [x] Add timeout handling (max recording duration)
  - [x] Add error recovery and model failure handling

**Configuration Updates**:
```yaml
# config/audio_config.yaml - NEW sections
pipeline:
  vad:
    model: "silero"
    threshold: 0.5
    min_speech_duration_ms: 250
    min_silence_duration_ms: 500
    speech_pad_ms: 30

  speech_to_text:
    model_size: "tiny.en"
    compute_type: "int8"
    device: "cpu"
    cpu_threads: 4
    beam_size: 5
    language: "en"
    max_recording_duration: 15
```

**Tests Required**:
```python
# tests/test_audio_pipeline.py (NEW)
- Test VAD model initialization
- Test Whisper model initialization
- Test state machine transitions
- Test audio buffer management
- Test VAD speech detection
- Test transcription accuracy
- Test timeout handling
- Test error recovery

# Integration test with test_audio_models.py patterns
- Use existing test audio files (HeyRover.wav, TheRainInSpain.wav)
- Verify wake word → VAD → transcription flow
```

### 5.2 Message Type Definitions

- [x] Create `TranscriptionResult.msg`
  - [x] Define message fields (text, confidence, duration, language)
  - [x] Update `CMakeLists.txt` in `robot_interfaces`
  - [x] Rebuild workspace (`colcon build --packages-select robot_interfaces`)

**New Message**:
```msg
# robot_interfaces/msg/TranscriptionResult.msg
std_msgs/Header header
string text                    # Transcribed text
float32 confidence            # Overall confidence score (0.0-1.0)
float32 duration              # Audio duration in seconds
string language               # Detected language (e.g., "en")
```

### 5.3 Audio Playback Node (Streamlined with Integrated TTS)

- [x] **Refactor `audio_playback_node.py`** to integrate Piper TTS directly
  - [x] **Remove audio data subscription** (no more `/audio/tts_output`)
  - [x] **Add text subscription** to `/audio/tts_request` (std_msgs/String)
  - [x] **Add event subscription** to `/audio/events` (AudioEvent)
  - [x] **Initialize Piper TTS model** in the node
    - [x] Load ONNX model and voice configuration
    - [x] Use same Piper setup as previous tts_node
    - [x] Lazy loading: only load when first TTS request arrives
  - [x] **Implement text-to-audio synthesis**
    - [x] Subscribe to text messages
    - [x] Synthesize audio using Piper ONNX inference
    - [x] Queue synthesized audio for playback
    - [x] Handle synthesis errors gracefully
  - [x] **Implement event-driven notification sounds**
    - [x] Load notification audio files on startup:
      - `assets/audio/notify_asc.wav` (wake word detected)
      - `assets/audio/notify_desc.wav` (speech ended)
    - [x] Subscribe to `/audio/events`
    - [x] Play notification sounds based on event types:
      - `wake_word_detected` → `notify_asc.wav`
      - `speech_ended` → `notify_desc.wav`
    - [x] Prioritize notifications (high priority in queue)
  - [x] **Maintain existing features**
    - [x] Queue-based playback system with priorities
    - [x] Volume normalization
    - [x] Publish playback events to `/audio/events`
    - [x] Reconnection logic for hardware failures

**Message Types**:
- **Input**: `std_msgs/String` on `/audio/tts_request` (text to synthesize)
- **Input**: `robot_interfaces/AudioEvent` on `/audio/events` (trigger notifications)
- **Output**: `robot_interfaces/AudioEvent` on `/audio/events` (playback status)


### 5.4 Text-to-Speech Node (DEPRECATED)

- [x] `tts_node.py` **DEPRECATED** - functionality moved to audio_playback_node
  - [ ] Update documentation to reflect deprecation
  - [ ] Keep file for reference but remove from launch files
  - [ ] All TTS functionality now in `audio_playback_node.py`

### 5.5 Integration Testing

**Verification Tests (Completed):**
- [x] Test imports and message types
- [x] Verify model libraries available (openWakeWord, Silero VAD, faster-whisper)
- [x] Verify configuration file structure
- [x] Verify successful build and compilation

**Real-Time Testing (Pending):**
- [ ] Test complete audio pipeline flow:
  - [ ] Wake word detection → VAD activation → Speech capture → Transcription
  - [ ] Verify no audio streaming over ROS2 (check topic list)
  - [ ] Verify only control messages published
- [ ] Test end-to-end latency (target: <3 seconds wake-to-transcription)
- [ ] Test resource usage:
  - [ ] CPU usage <20% during idle (wake word only)
  - [ ] Memory usage <1GB total
  - [ ] No memory leaks over 10 minutes
- [ ] Test edge cases:
  - [ ] Very short speech (<1 second)
  - [ ] Long speech (>10 seconds)
  - [ ] Multiple wake words in quick succession
  - [ ] Background noise handling

**Integration Tests Required**:
```python
# scripts/test_audio_pipeline_quick.py (COMPLETED)
- ✓ Test imports and message types
- ✓ Verify model libraries
- ✓ Verify configuration

# scripts/test_audio_pipeline_integration.py (TODO)
- Test with pre-recorded audio files
- Simulate wake word + speech
- Verify event sequence
- Verify transcription accuracy
- Measure end-to-end latency

# Manual testing (TODO)
- Real-time microphone testing
- Various speaking styles and accents
- Different noise levels
- Long-duration operation (1+ hour)
```

**Deliverables**:
- ✅ Self-contained audio processing pipeline in single node
- ✅ Wake word detection (continuous, <5% CPU target)
- ✅ VAD integration (Silero VAD)
- ✅ Speech-to-text transcription (faster-whisper)
- ✅ State machine for pipeline flow
- ✅ Only lightweight control messages published (no audio streaming)
- ⏳ End-to-end latency <3 seconds (pending real-time testing)
- ⏳ Comprehensive tests (unit tests complete, integration pending)
- ✅ Updated documentation (architecture, implementation plan)

**Status**: ✅ Implementation complete, ready for real-time testing

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

## Phase 7: Cognitive Core (Ollama + Moondream) (Weeks 9-10)

### 7.1 Ollama Client Node
- [ ] Implement `cognitive_client_node.py`
  - [ ] Subscribe to `/audio/transcribed_text` (from Whisper)
  - [ ] Subscribe to `/camera/undistorted` (for visual context)
  - [ ] Subscribe to `/odom` and `/perception/objects` (for world context)
  - [ ] Implement HTTP client for Ollama API (`http://localhost:11434/api/generate`)
  - [ ] Implement Base64 image encoding
  - [ ] Construct structured prompts for Moondream
  - [ ] Parse JSON responses from Moondream
  - [ ] Publish intents to `/cognitive/intent`
  - [ ] Handle API timeouts and errors
- [ ] Create prompt templates:
  ```python
  SYSTEM_PROMPT = (
      "Context: {world_context}. "
      "User Instruction: {user_prompt}. "
      "Based on the image and context, output a JSON object with: "
      "{'action': string, 'target': string, 'explanation': string}."
  )
  ```

### 7.2 Intent Parsing & Validation
- [ ] Implement `json_parser.py`
  - [ ] Extract JSON from potential markdown blocks
  - [ ] Validate against schema (action, target, explanation)
  - [ ] Handle hallucinated fields
- [ ] Define Intent Message:
  ```
  # cognitive_msgs/Intent.msg
  string action
  string target
  string explanation
  float32 confidence
  bool visual_verification_required
  ```

### 7.3 Visual Verification Logic
- [ ] Implement verification prompts:
  - [ ] "Is the goal [X] achieved in this image? Answer boolean."
  - [ ] "Do you see [X] in the center of the frame?"
- [ ] Test verification accuracy with Moondream
- [ ] Benchmark latency for verification calls (target: <2s)

### 7.4 Cognitive Integration Tests
- [ ] Test end-to-end flow:
  - [ ] Transcribed Text + Image -> Ollama -> Intent
- [ ] Test with various scenes and commands
- [ ] Measure total latency (Whisper + Moondream)
- [ ] Verify memory stability (Ollama resident in RAM)

**Deliverables**:
- Working `cognitive_client_node`
- Robust JSON parsing for Moondream
- Visual verification capabilities
- Integration tests passing

---

## Phase 8: Behavioral Architecture (Weeks 11-12)

### 8.1 BehaviorTree.CPP Setup
- [ ] Install BehaviorTree.CPP library
- [ ] Create `behavioral_nodes` ROS2 package
- [ ] Set up BehaviorTree.CPP ROS2 integration
- [ ] Create blackboard data structure
- [ ] Test basic behavior tree execution

### 8.2 Command Router & Cognitive Bridge
- [ ] Implement `command_router.py`
  - [ ] Subscribe to `/audio/transcribed_text`
  - [ ] Regex match for simple commands (stop, move forward)
  - [ ] Route complex commands to `cognitive_client_node`
  - [ ] Log routing decisions
- [ ] Create command mapping:
  ```python
  SIMPLE_COMMANDS = {
    "stop": {"action": "stop"},
    "go forward": {"action": "move", "direction": "forward"},
    "turn left": {"action": "turn", "direction": "left"},
    "turn right": {"action": "turn", "direction": "right"},
    "go back": {"action": "move", "direction": "backward"}
  }
  ```

### 8.3 Blackboard Implementation
- [ ] Define blackboard schema:
  ```
  # Robot state
  - robot_pose (geometry_msgs/PoseStamped)
  - battery_level (float)
  - current_intent (cognitive_msgs/Intent)
  - navigation_status (enum)
  - last_visual_verification (bool)
  ```
- [ ] Implement blackboard update subscribers

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
            <ExecuteCognitiveIntent/>
          </Fallback>
        </Sequence>
      </Fallback>
      <MonitorSystemHealth/>
    </ReactiveSequence>
  </BehaviorTree>
  ```
- [ ] Implement `ExecuteCognitiveIntent`:
  - [ ] Read intent from blackboard
  - [ ] Set goal for navigation
  - [ ] Trigger visual verification if needed

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

### 8.6 Stuck Detection & Recovery
- [ ] Implement `stuck_detector.py`
  - [ ] Monitor cmd_vel vs IMU acceleration
- [ ] Implement recovery behavior (back up, rotate)

### 8.7 Multimodal Behavior Nodes
- [ ] Implement `VisualVerificationNode`:
  - [ ] Call `cognitive_client_node` with verification prompt
  - [ ] Return SUCCESS/FAILURE based on boolean response
- [ ] Implement `SceneAssessmentNode`:
  - [ ] Ask "Is the path clear?" or similar if stuck

### 8.8 Behavior Tree Executor
- [ ] Implement `behavior_tree_executor.py`
  - [ ] Load XML trees
  - [ ] Tick tree at 10Hz
- [ ] Test integration with `cognitive_client_node`

### 8.9 Integration Testing
- [ ] Test complete flow:
  - Voice -> Whisper -> Router -> Ollama -> Intent -> BT -> Action
- [ ] Verify latency constraints
- [ ] Test emergency stop override

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
- [ ] Implement `web_server.py` (FastAPI + WebSocket)
- [ ] Implement ROS2 bridge node
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

### 9.2 System Monitoring Node
- [ ] Implement `system_monitor.py`
  - [ ] Monitor CPU/GPU/RAM usage
  - [ ] Monitor Ollama service status
  - [ ] Monitor Moondream memory footprint (~1.8GB)
  - [ ] Publish to `/system/metrics`

### 9.3 Frontend Development
- [ ] Implement HTML/CSS/JS dashboard
  - [ ] Camera feed with overlays
  - [ ] System metrics charts
  - [ ] Command input (Text/Voice)
  - [ ] Ollama/Moondream status indicator
  - [ ] Log viewer

### 9.4 Data Visualization
- [ ] Implement camera feed streaming (MJPEG)
- [ ] Implement object detection overlay
- [ ] Implement 2D map visualization

### 9.5 Control Interface
- [ ] Implement multimodal command input
- [ ] Implement emergency stop button

### 9.6 Web Server Integration
- [ ] Create `launch/web_interface_launch.py`
- [ ] Test with multiple clients

**Deliverables**:
- Working web server and dashboard
- System monitoring with Ollama tracking
- Real-time camera feed and metrics

---

## Phase 10: System Integration & Testing (Weeks 14-15)

### 10.1 Full System Launch
- [ ] Create `launch/full_system_launch.py`
  - [ ] Launch perception, audio, localization
  - [ ] Launch `cognitive_client_node`
  - [ ] Launch behavior tree
  - [ ] Launch web interface
- [ ] Test full system startup (<30s)

### 10.2 End-to-End Testing
- [ ] Create test scenarios:

  **Scenario 1: Simple Navigation**
  - [ ] User says "go forward"
  - [ ] Robot moves forward

  **Scenario 2: Object Finding**
  - [ ] User says "find the red ball"
  - [ ] Whisper transcribes
  - [ ] Ollama analyzes scene
  - [ ] Robot navigates to ball

  **Scenario 3: Visual Verification**
  - [ ] Robot arrives at target
  - [ ] Robot asks Ollama "Am I there?"
  - [ ] Ollama confirms "Yes"
  - [ ] Robot stops and reports success

**Deliverables**:
- Full system launch file
- Verified end-to-end scenarios
- Performance report (Latency, RAM usage)

  **Scenario 4: Visual Question Answering**
  - [ ] User says wake word
  - [ ] User asks "what do you see?" while pointing camera at objects
  - [ ] Moondream processes camera feed
  - [ ] Robot describes visible objects
  - [ ] Robot provides contextual information

  **Scenario 5: Stuck Recovery**
  - [ ] Robot encounters obstacle while navigating
  - [ ] Stuck detector triggers after 3 seconds
  - [ ] Robot backs up and rotates
  - [ ] Robot retries navigation
  - [ ] Robot says "I got stuck but found another way"

  **Scenario 6: Multimodal Clarification**
  - [ ] User says wake word
  - [ ] User says "go to the ball" with ambiguous image
  - [ ] Multiple balls detected
  - [ ] Robot asks "Which ball? The red one or the blue one?"
  - [ ] User responds with text
  - [ ] Robot navigates to specified ball

- [ ] Test each scenario 5+ times
- [ ] Measure success rate (target: >80%)
- [ ] Test Ollama context efficiency
- [ ] Verify Moondream constant ~1.8GB VRAM usage
- [ ] Document failure modes
- [ ] Monitor via web interface during tests
- [ ] Verify web interface shows correct real-time multimodal data

### 9.3 Performance Profiling
- [ ] Profile full system under load:
  - [ ] CPU usage per core (target: <80% average)
  - [ ] GPU usage (target: <90%)
  - [ ] RAM usage (target: <7.5 GB)
  - [ ] Moondream VRAM usage (should be constant ~1.8GB)
  - [ ] Temperature (target: <80°C sustained)
- [ ] Measure multimodal latencies:
  - [ ] Wake word to ASR complete (<3 seconds)
  - [ ] ASR to Ollama text processing (<2 seconds)
  - [ ] Image encoding to Base64 (<0.5 second)
  - [ ] Ollama context processing (<3 seconds)
  - [ ] Moondream response generation (<5 seconds for complex)
  - [ ] Response to TTS start (<1 second)
  - [ ] Command to motor action (<2 seconds for simple)
  - [ ] Object detection to navigation start (<5 seconds)
- [ ] Profile individual nodes:
  - [ ] Identify multimodal processing bottlenecks
  - [ ] Optimize Ollama request pipeline
  - [ ] Reduce memory allocations
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
- Complete documentation (user + developer) with Ollama/Moondream guides
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
- [ ] Add configuration editor for Moondream parameters
- [ ] Add multimodal mission replay (recorded sessions with all modalities)
- [ ] Add Moondream performance graphs and token usage statistics
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
- Showcase of Moondream capabilities on edge devices

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
   - **Mitigation**: Moondream's constant ~1.8GB VRAM footprint eliminates model swapping complexity
   - **Contingency**: Use lighter perception models, reduce web interface functionality

2. **Multimodal Processing Latency**
   - **Mitigation**: Optimize token encoding pipelines, async processing, efficient context management
   - **Contingency**: Fallback to text-only mode, reduce multimodal context window

3. **Visual Odometry Failure**
   - **Mitigation**: Multiple fallback modes (IMU-only, dead reckoning)
   - **Contingency**: Add wheel encoders to Wave Rover (hardware modification)

4. **Moondream Context Window Overflow**
   - **Mitigation**: Intelligent context pruning, sliding window management, token compression
   - **Contingency**: Reset context periodically, prioritize recent interactions

5. **Thermal Throttling with Multimodal Processing**
   - **Mitigation**: Balance workloads across modalities, efficient scheduling, thermal monitoring
   - **Contingency**: Reduce multimodal complexity, disable vision processing, active cooling fan

### Medium-Risk Items
1. **Ollama Integration Stability**
   - **Mitigation**: Use tested versions, extensive integration testing
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
- [ ] Multimodal command understanding with Moondream (>90% accuracy)
- [ ] Simultaneous text, audio, and vision processing
- [ ] Cross-modal context awareness and reasoning
- [ ] Semantic SLAM with object tracking
- [ ] Stuck detection and recovery (>80% success)
- [ ] Multi-turn multimodal conversations
- [ ] 8+ hour continuous operation
- [ ] RAM usage <7.5 GB (with constant ~1.8GB Moondream VRAM)
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
