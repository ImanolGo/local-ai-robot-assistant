# GitHub Copilot Instructions for Local AI Robot Assistant

## Project Context
This is an autonomous AI robot assistant running on NVIDIA Jetson Orin Nano with ROS2. All AI processing happens locally on-device.

## Tech Stack
- **Platform**: NVIDIA Jetson Orin Nano (8GB RAM)
- **OS**: Ubuntu 22.04 with JetPack SDK
- **Framework**: ROS2 Humble
- **Languages**: Python 3.10+ (primary), C++ (performance-critical)
- **AI Models**: TensorRT-optimized (YOLO, FastDepth, Whisper, LLaMA-2)

## Critical Constraints
- **Memory Budget**: Maximum 7.5GB RAM usage
- **Performance**: Real-time processing required (≥10 FPS for vision, <2s for voice)
- **Privacy**: 100% local processing, no cloud APIs
- **Hardware**: Limited to single Jetson device

## Coding Standards

### Python
- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 100 characters
- Use docstrings (Google style)
- Prefer async/await for I/O operations

### ROS2 Nodes
- Each node should be single-responsibility
- Use descriptive topic names with proper namespacing
- Always handle node shutdown gracefully
- Log important events using ROS2 logger
- Include node parameter configuration

### Error Handling
- Always validate hardware connections
- Implement retry logic for UART/serial communication
- Handle model loading failures gracefully
- Add meaningful error messages
- Use try-except blocks for external calls

### Performance Optimization
- Use TensorRT for all deep learning inference
- Minimize data copies between nodes
- Use shared memory for large data (images)
- Profile before optimizing
- Cache expensive computations

### Testing
- Write unit tests for all new functions (target: 70%+ coverage)
- Mock hardware dependencies in tests
- Include integration tests for multi-node features
- Document test scenarios in docstrings

## Hardware-Specific Patterns

### UART Communication
Always use this pattern for Wave Rover:
```python
import serial
import json

class WaveRoverController:
    def __init__(self, port='/dev/ttyTHS0', baudrate=115200):
        self.serial = serial.Serial(port, baudrate, timeout=1)

    def send_command(self, cmd_dict):
        """Send JSON command with error handling."""
        try:
            json_str = json.dumps(cmd_dict) + '\n'
            self.serial.write(json_str.encode())
            return True
        except Exception as e:
            self.get_logger().error(f"UART error: {e}")
            return False
```

### Camera Processing
Always undistort images before AI processing:
```python
# Load calibration
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Undistort
undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
```

### TensorRT Inference
Use this pattern for optimized inference:
```python
import tensorrt as trt
import pycuda.driver as cuda

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
```

## Common Patterns

### ROS2 Node Template
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Declare parameters
        self.declare_parameter('param_name', default_value)

        # Initialize publishers/subscribers
        self.publisher = self.create_publisher(MsgType, 'topic', 10)
        self.subscription = self.create_subscription(
            MsgType, 'topic', self.callback, 10)

        # Initialize timers
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('Node initialized')

    def callback(self, msg):
        """Handle incoming messages."""
        pass

    def timer_callback(self):
        """Periodic processing."""
        pass

    def destroy_node(self):
        """Cleanup on shutdown."""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Async Audio Processing
```python
import asyncio
import numpy as np
import sounddevice as sd

async def process_audio_stream():
    """Process audio in non-blocking manner."""
    async def audio_callback(indata, frames, time, status):
        if status:
            print(f'Audio status: {status}')
        # Process audio
        await process_audio_chunk(indata.copy())

    with sd.InputStream(callback=audio_callback):
        await asyncio.sleep(float('inf'))
```

## File-Specific Guidance

### When working in `perception_nodes/`:
- Always process undistorted images
- Use TensorRT for inference
- Publish detection results with timestamps
- Include confidence thresholds as parameters

### When working in `audio_interface_nodes/`:
- Use PyAudio or sounddevice for I/O
- Implement proper buffer management
- Handle device disconnections gracefully
- Minimize latency in processing pipeline

### When working in `actuation_nodes/`:
- Validate all UART commands before sending
- Implement watchdog timers for safety
- Convert Twist messages to differential drive properly
- Log all motor commands for debugging

### When working in `behavioral_nodes/`:
- Use BehaviorTree.CPP patterns
- Keep actions atomic and testable
- Update Blackboard consistently
- Handle action failures with fallbacks

## Documentation Requirements

Every function should have:
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description.

    Detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Example:
        >>> result = function_name(val1, val2)
        >>> print(result)
        expected_output
    """
```

## Testing Patterns

### Unit Test Template
```python
import unittest
from unittest.mock import Mock, patch

class TestMyNode(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.node = MyNode()

    def tearDown(self):
        """Clean up after test."""
        self.node.destroy_node()

    def test_functionality(self):
        """Test specific functionality."""
        # Arrange
        input_data = ...
        expected = ...

        # Act
        result = self.node.process(input_data)

        # Assert
        self.assertEqual(result, expected)

    @patch('module.external_dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependencies."""
        mock_dep.return_value = test_value
        # Test code
```

## Integration Testing
```python
import rclpy
from rclpy.executors import MultiThreadedExecutor

def test_node_integration():
    """Test multiple nodes together."""
    rclpy.init()

    node1 = Node1()
    node2 = Node2()

    executor = MultiThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)

    try:
        # Run test scenario
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()
```

## Performance Considerations

### Memory Management
- Use `del` to explicitly free large objects
- Profile memory usage with memory_profiler
- Implement model lazy loading if RAM constrained
- Use generators for large data processing

### GPU Optimization
- Batch operations when possible
- Keep data on GPU between operations
- Use async CUDA streams for parallel processing
- Monitor GPU memory with nvidia-smi

### Latency Reduction
- Minimize ROS2 message serialization overhead
- Use intra-process communication where possible
- Profile with ros2 tracing tools
- Optimize critical paths first

## Common Pitfalls to Avoid

❌ Don't use `localStorage` or `sessionStorage` in web artifacts (not supported)
❌ Don't assume infinite memory (8GB limit is hard)
❌ Don't use blocking I/O in ROS2 callbacks
❌ Don't forget to undistort fisheye camera images
❌ Don't ignore hardware connection failures
❌ Don't load all models simultaneously (memory)
❌ Don't use wheel odometry (Wave Rover has no encoders)

✅ Do validate all sensor data
✅ Do implement proper error recovery
✅ Do log important events
✅ Do write tests
✅ Do profile before optimizing
✅ Do handle hardware disconnections
✅ Do use visual odometry from SLAM

## Reference Documentation

- Architecture: `docs/architecture.md`
- PRD: `docs/prd.md`
- Implementation Plan: `docs/implementation_plan.md`
- Status: `STATUS.md`
- API Docs: `docs/api/`

## Project-Specific Vocabulary

- **Wave Rover**: Robot chassis with differential drive, no encoders
- **IMU**: Inertial Measurement Unit (9-axis, accessed via UART)
- **Fisheye distortion**: 160° FOV camera requires calibration and undistortion
- **TensorRT**: NVIDIA's inference optimization framework
- **NanoLLM**: NVIDIA's framework for running LLMs on Jetson
- **RTAB-Map**: Visual SLAM system we use for mapping
- **Blackboard**: Shared state in behavior trees
- **EKF**: Extended Kalman Filter for sensor fusion

## When to Ask for Human Review

Please request human review when:
- Modifying hardware communication protocols
- Changing core behavior tree logic
- Implementing safety-critical features (e.g., emergency stop)
- Making significant architectural changes
- Unsure about hardware constraints
- Implementing new AI model integrations
