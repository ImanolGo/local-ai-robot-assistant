# UART Communication Protocol Documentation

This document describes the UART communication protocol implementation for the Wave Rover chassis, including the motor controller and IMU nodes.

## Overview

The UART communication system provides bidirectional communication between the NVIDIA Jetson Orin Nano and the Wave Rover ESP32 controller. It enables:

- Motor control with differential drive kinematics
- IMU data acquisition from the 9-axis sensor
- Chassis state monitoring
- Safety features (emergency stop, watchdog timer)

## Hardware Setup

### Connection Details
- **Serial Port**: `/dev/ttyTHS1` (UART1 on Jetson Orin Nano)
- **Baud Rate**: 115200 bps
- **Data Format**: 8N1 (8 data bits, no parity, 1 stop bit)
- **Protocol**: JSON over serial

### Physical Connections
```
Jetson Orin Nano    Wave Rover ESP32
-----------------   ----------------
GPIO 14 (TXD)   ->  RX
GPIO 15 (RXD)   <-  TX
GND             --  GND
```

## Protocol Specification

### Message Format
All commands and responses use JSON format with UTF-8 encoding, terminated by newline (`\n`).

**Command Format**:
```json
{"T": <command_type>, "<param1>": <value1>, "<param2>": <value2>}
```

**Response Format**: Varies by command type

### Command Types

#### 1. Motor Speed Control (T:1)
Controls left and right wheel speeds independently.

**Command**:
```json
{"T": 1, "L": <left_speed>, "R": <right_speed>}
```

**Parameters**:
- `L`: Left wheel speed (-0.5 to +0.5)
- `R`: Right wheel speed (-0.5 to +0.5)
- Values represent PWM percentage: 0.5 = 100%, 0.25 = 50%

**Response**: None (unless continuous feedback enabled)

**Example**:
```json
{"T": 1, "L": 0.3, "R": 0.3}  # Move forward at 60% speed
{"T": 1, "L": 0.2, "R": -0.2} # Turn in place (left)
{"T": 1, "L": 0.0, "R": 0.0}  # Stop
```

#### 2. PWM Input Control (T:11)
Direct PWM control for debugging purposes.

**Command**:
```json
{"T": 11, "L": <left_pwm>, "R": <right_pwm>}
```

**Parameters**:
- `L`: Left motor PWM (typically 0-200)
- `R`: Right motor PWM (typically 0-200)

**Example**:
```json
{"T": 11, "L": 100, "R": 100}  # Set both motors to PWM 100
```

#### 3. ROS Control (T:13) - UGV01 Only
Linear and angular velocity control (requires encoder feedback).

**Command**:
```json
{"T": 13, "X": <linear_vel>, "Z": <angular_vel>}
```

**Parameters**:
- `X`: Linear velocity (m/s)
- `Z`: Angular velocity (rad/s)

**Note**: Not applicable for Wave Rover (no encoders)

#### 4. IMU Data Query (T:126)
Request IMU sensor data.

**Command**:
```json
{"T": 126}
```

**Response**:
```json
{
  "roll": <degrees>,
  "pitch": <degrees>,
  "yaw": <degrees>,
  "AccX": <g_force>,
  "AccY": <g_force>,
  "AccZ": <g_force>,
  "GyroX": <deg_per_sec>,
  "GyroY": <deg_per_sec>,
  "GyroZ": <deg_per_sec>
}
```

**Example Response**:
```json
{
  "roll": 2.3,
  "pitch": -1.1,
  "yaw": 45.7,
  "AccX": 0.02,
  "AccY": 0.98,
  "AccZ": 0.15,
  "GyroX": 0.1,
  "GyroY": -0.2,
  "GyroZ": 0.05
}
```

#### 5. Continuous Feedback (T:131)
Enable/disable continuous chassis data streaming.

**Command**:
```json
{"T": 131, "cmd": <mode>}
```

**Parameters**:
- `cmd`: 1 to enable, 0 to disable

**Response**: Continuous stream of chassis data

#### 6. OLED Display (T:3)
Control the onboard OLED display.

**Command**:
```json
{"T": 3, "lineNum": <line>, "Text": "<message>"}
```

**Parameters**:
- `lineNum`: Display line number (0-3)
- `Text`: Message to display (max 21 characters)

**Example**:
```json
{"T": 3, "lineNum": 0, "Text": "Robot Active"}
```

## ROS2 Integration

### Motor Controller Node

**Package**: `actuation_nodes`
**Node**: `uart_motor_controller`

#### Subscribed Topics
- `/cmd_vel` (geometry_msgs/Twist) - Robot velocity commands
- `/motor_command` (robot_interfaces/MotorCommand) - Direct motor control

#### Published Topics
- `/motor_status` (robot_interfaces/ChassisState) - Motor status information
- `/chassis_state` (robot_interfaces/ChassisState) - Complete chassis state
- `/odom_raw` (nav_msgs/Odometry) - Raw odometry estimate

#### Services
- `/emergency_stop` (robot_interfaces/EmergencyStop) - Emergency motor stop
- `/set_mode` (robot_interfaces/SetMode) - Set operation mode

#### Parameters
```yaml
uart:
  port: "/dev/ttyTHS1"
  baudrate: 115200
  timeout: 1.0

motor:
  command_rate: 20.0          # Hz
  watchdog_timeout: 0.5       # seconds
  max_speed: 0.5              # wheel speed limit
  wheelbase: 0.16             # meters
  max_linear_velocity: 0.3    # m/s
  max_angular_velocity: 1.0   # rad/s
```

### IMU Node

**Package**: `localization_nodes`
**Node**: `uart_imu_node`

#### Published Topics
- `/imu/data` (sensor_msgs/Imu) - Processed IMU data with covariances
- `/imu/raw` (custom) - Raw IMU data from Wave Rover

#### Parameters
```yaml
uart:
  port: "/dev/ttyTHS1"
  baudrate: 115200

imu:
  query_rate: 20.0            # Hz
  validate_data: true
  acceleration_limit: 50.0    # m/s²

frame_id: "imu_link"
```

## Safety Features

### Watchdog Timer
- **Function**: Automatically stops motors if no commands received
- **Timeout**: Configurable (default 0.5 seconds)
- **Behavior**: Sends stop command (`{"T":1, "L":0.0, "R":0.0}`)

### Emergency Stop
- **Trigger**: ROS2 service call or node shutdown
- **Behavior**: Immediate motor stop and command blocking
- **Recovery**: Manual service call to deactivate

### Command Rate Limiting
- **Purpose**: Prevent UART overflow and ensure stable control
- **Default Rate**: 20 Hz for motor commands
- **Implementation**: Timer-based command transmission

## Differential Drive Kinematics

### Forward Kinematics
Convert wheel speeds to robot velocities:
```
v = (v_left + v_right) / 2
ω = (v_right - v_left) / wheelbase
```

### Inverse Kinematics
Convert robot velocities to wheel speeds:
```
v_left = v - (ω × wheelbase) / 2
v_right = v + (ω × wheelbase) / 2
```

**Parameters**:
- `wheelbase`: 0.16 meters (Wave Rover specification)
- `max_wheel_speed`: Corresponds to motor's maximum safe speed

## Usage Examples

### Basic Motor Control
```python
import rclpy
from geometry_msgs.msg import Twist

# Create publisher
twist_pub = node.create_publisher(Twist, '/cmd_vel', 10)

# Move forward
twist = Twist()
twist.linear.x = 0.2  # 0.2 m/s forward
twist.angular.z = 0.0
twist_pub.publish(twist)

# Turn left
twist.linear.x = 0.1
twist.angular.z = 0.5  # 0.5 rad/s left turn
twist_pub.publish(twist)

# Stop
twist.linear.x = 0.0
twist.angular.z = 0.0
twist_pub.publish(twist)
```

### Emergency Stop
```python
from robot_interfaces.srv import EmergencyStop

# Create service client
emergency_client = node.create_client(EmergencyStop, '/emergency_stop')

# Activate emergency stop
request = EmergencyStop.Request()
request.enable_stop = True
request.reason = "Obstacle detected"
future = emergency_client.call_async(request)
```

### IMU Data Subscription
```python
from sensor_msgs.msg import Imu

def imu_callback(msg):
    # Extract orientation (quaternion)
    orientation = msg.orientation

    # Extract angular velocity (rad/s)
    angular_vel = msg.angular_velocity

    # Extract linear acceleration (m/s²)
    linear_acc = msg.linear_acceleration

# Create subscription
imu_sub = node.create_subscription(Imu, '/imu/data', imu_callback, 10)
```

## Testing

### Unit Tests
```bash
# Run motor controller tests
python3 -m pytest src/actuation_nodes/test/test_uart_motor_controller.py -v

# Run IMU node tests
python3 -m pytest src/localization_nodes/test/test_uart_imu_node.py -v
```

### Integration Tests
```bash
# Run with actual hardware
python3 integration_tests/test_uart_integration.py --port /dev/ttyTHS1

# Run hardware communication test only
python3 integration_tests/test_uart_integration.py --hardware-test

# Skip hardware-dependent tests
python3 integration_tests/test_uart_integration.py --skip-hardware
```

### Manual Testing
```bash
# Test basic UART communication
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --auto

# Interactive testing
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1

# Test specific functionality
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test motor
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test imu
```

## Troubleshooting

### Common Issues

#### Serial Port Access
```bash
# Check port permissions
ls -l /dev/ttyTHS1

# Add user to dialout group
sudo usermod -a -G dialout $USER

# Test port availability
python3 -c "import serial; serial.Serial('/dev/ttyTHS1', 115200)"
```

#### Communication Failures
1. **Check physical connections**
2. **Verify baud rate matches Wave Rover settings**
3. **Test with hardware test script**
4. **Check system logs**: `journalctl -u your-node-name`

#### Performance Issues
1. **Reduce command rate** if UART buffer overflows
2. **Check CPU usage** during operation
3. **Monitor memory usage** for memory leaks
4. **Verify power supply** stability

### Debug Mode
Enable detailed logging:
```yaml
logging:
  log_commands: true      # Log all sent commands
  log_responses: true     # Log all responses
  log_errors: true        # Log communication errors
```

### Error Codes
- **Serial connection failed**: Check port and permissions
- **JSON parse error**: Malformed response from Wave Rover
- **Timeout error**: No response within expected time
- **Watchdog triggered**: No commands received within timeout

## Performance Specifications

### Latency
- **Command latency**: < 10ms (UART + processing)
- **IMU update rate**: 20 Hz (50ms period)
- **Motor command rate**: 20 Hz (50ms period)

### Throughput
- **UART bandwidth**: 115200 bps
- **Typical command size**: ~30 bytes
- **Maximum command rate**: ~380 commands/second (theoretical)

### Resource Usage
- **CPU usage**: < 5% per node (Jetson Orin Nano)
- **Memory usage**: < 50MB per node
- **UART buffer**: Standard Linux serial buffer (4096 bytes)

## Configuration Reference

### Complete UART Configuration
See `config/uart_config.yaml` for full configuration options including:
- Serial communication settings
- Motor control parameters
- IMU processing options
- Safety feature configuration
- ROS2 topic/service names
- Frame ID assignments

### Launch File Example
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='actuation_nodes',
            executable='uart_motor_controller',
            name='uart_motor_controller',
            parameters=['/path/to/uart_config.yaml']
        ),
        Node(
            package='localization_nodes',
            executable='uart_imu_node',
            name='uart_imu_node',
            parameters=['/path/to/uart_config.yaml']
        )
    ])
```

## Related Documentation

- **Wave Rover Hardware Manual**: Motor specifications and wiring
- **ROS2 Interfaces**: Custom message and service definitions
- **Architecture Document**: System-level integration details
- **Hardware Test Scripts**: Validation and debugging tools

---

*For technical support or questions about the UART implementation, please refer to the project repository issues or contact the development team.*
