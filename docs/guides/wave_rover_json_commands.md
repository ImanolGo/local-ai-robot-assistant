# Wave Rover JSON Commands Reference

## Overview

The Waveshare Wave Rover communicates via JSON commands over serial (UART or USB). This document provides a complete reference for all available commands, organized by functionality.

## Serial Communication Setup

### Connection Methods

1. **40-PIN UART Interface** (Recommended for Jetson)
   - Connect to `/dev/ttyTHS1` on Jetson Orin Nano
   - Default baudrate: 115200
   - Hardware flow control: Disabled (RTS/DTR = False)

2. **USB Serial Port**
   - Connect USB cable from robot's slave drive board to host
   - Requires robot disassembly
   - Port varies by system (e.g., `/dev/ttyUSB0`, `COM20`)

### Basic Serial Setup

```python
import serial

ser = serial.Serial('/dev/ttyTHS1', baudrate=115200, dsrdtr=None)
ser.setRTS(False)
ser.setDTR(False)

# Send command
command = {"T": 1, "L": 0.2, "R": 0.2}
ser.write(json.dumps(command).encode() + b'\n')

# Read response
response = ser.readline().decode('utf-8')
```

## Command Categories

### 1. Chassis Movement

#### Speed Control (Recommended)

**Command Type:** `CMD_SPEED_CTRL`

```json
{"T": 1, "L": 0.5, "R": 0.5}
```

- **T**: Command type (1)
- **L**: Left wheel speed (-0.5 to +0.5)
- **R**: Right wheel speed (-0.5 to +0.5)
- **Notes**:
  - Positive values = forward, negative = backward
  - 0.5 = 100% PWM, 0.25 = 50% PWM
  - No encoders on Wave Rover (unlike UGV01)

**Examples:**

```json
{"T": 1, "L": 0.2, "R": 0.2}    // Forward
{"T": 1, "L": -0.2, "R": -0.2}  // Backward
{"T": 1, "L": 0.2, "R": -0.2}   // Turn left (differential)
{"T": 1, "L": 0, "R": 0}        // Stop
```

#### PWM Control (Debug Only)

**Command Type:** `CMD_PWM_INPUT`

```json
{"T": 11, "L": 164, "R": 164}
```

- **T**: Command type (11)
- **L**: Left motor PWM (-255 to +255)
- **R**: Right motor PWM (-255 to +255)
- **Warning**: DC motors may not rotate at low PWM values
- **Use Case**: Debugging only; prefer speed control for normal operation

#### ROS Control (UGV01 Only)

**Command Type:** `CMD_ROS_CTRL`

```json
{"T": 13, "X": 0.1, "Z": 0.3}
```

- **T**: Command type (13)
- **X**: Linear velocity (m/s)
- **Z**: Angular velocity (rad/s)
- **Limitation**: Only available on UGV01 with encoders (not Wave Rover)

### 2. Motor Configuration

#### PID Settings (UGV01 Only)

```json
{"T": 2, "P": 200, "I": 2500, "D": 0, "L": 255}
```

- **T**: Command type (2)
- **P**: Proportional coefficient
- **I**: Integral coefficient
- **D**: Derivative coefficient
- **L**: Windup limit (reserved, not currently used)
- **Limitation**: Only for UGV01 with encoders

### 3. OLED Display Control

#### Set Display Content

```json
{"T": 3, "lineNum": 0, "Text": "Hello World"}
```

- **T**: Command type (3)
- **lineNum**: Line number (0-3, total 4 lines)
- **Text**: Display text content
- **Behavior**: Replaces content on specified line without affecting others

**Examples:**

```json
{"T": 3, "lineNum": 0, "Text": "Status: Active"}
{"T": 3, "lineNum": 1, "Text": "Battery: 85%"}
{"T": 3, "lineNum": 2, "Text": "Mode: Auto"}
{"T": 3, "lineNum": 3, "Text": "Speed: 0.2"}
```

#### Restore Default Display

```json
{"T": -3}
```

- **T**: Command type (-3)
- **Effect**: Restores OLED to show default robot information

### 4. Sensor Data Retrieval

#### IMU Data Request

```json
{"T": 126}
```

- **T**: Command type (126)
- **Returns**: Heading angle, magnetic field, acceleration, attitude, temperature
- **Response Format**: JSON with IMU sensor data

#### Chassis Feedback

**Command Type:** `CMD_BASE_FEEDBACK`

```json
{"T": 130}
```

- **T**: Command type (130)
- **Returns**: Chassis status information
- **Mode**: Request-response (query-based)

### 5. Communication Settings

#### Continuous Feedback Control

```json
{"T": 131, "cmd": 1}  // Enable
{"T": 131, "cmd": 0}  // Disable (default)
```

- **T**: Command type (131)
- **cmd**: 0 = off, 1 = on
- **Default**: Disabled (request-response mode)
- **When Enabled**: Continuous data streaming (ideal for ROS)
- **When Disabled**: Query-based communication

#### Serial Echo Control

```json
{"T": 143, "cmd": 1}  // Enable
{"T": 143, "cmd": 0}  // Disable (default)
```

- **T**: Command type (143)
- **cmd**: 0 = off, 1 = on
- **Effect**: When enabled, echoes all sent commands back through serial

### 6. GPIO Control

#### IO4/IO5 PWM Control

```json
{"T": 132, "IO4": 255, "IO5": 255}
```

- **T**: Command type (132)
- **IO4**: PWM value for IO4 pin (0-255)
- **IO5**: PWM value for IO5 pin (0-255)
- **Use Case**: Control external devices via GPIO pins

### 7. External Module Support

#### Module Type Configuration

```json
{"T": 4, "cmd": 0}
```

- **T**: Command type (4)
- **cmd**: Module type
  - 0: Null (no module)
  - 1: RoArm-M2 (robotic arm)
  - 3: Gimbal (pan-tilt)

#### Pan-Tilt Control

```json
{"T": 133, "X": 45, "Y": 45, "SPD": 0, "ACC": 0}
```

- **T**: Command type (133)
- **X**: Horizontal angle (positive = left, negative = right)
- **Y**: Vertical angle (positive = up, negative = down)
- **SPD**: Speed parameter
- **ACC**: Acceleration parameter
- **Requirement**: Pan-tilt module must be installed

## Integration Examples

### Basic Movement Controller

```python
import json
import serial

class WaveRoverController:
    def __init__(self, port='/dev/ttyTHS1', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1.0)
        self.ser.setRTS(False)
        self.ser.setDTR(False)

    def move(self, left_speed: float, right_speed: float) -> bool:
        """Move robot with differential drive control."""
        cmd = {"T": 1, "L": left_speed, "R": right_speed}
        try:
            self.ser.write(json.dumps(cmd).encode() + b'\n')
            return True
        except Exception as e:
            print(f"Movement error: {e}")
            return False

    def stop(self):
        """Stop robot movement."""
        return self.move(0.0, 0.0)

    def forward(self, speed=0.2):
        """Move forward at specified speed."""
        return self.move(speed, speed)

    def backward(self, speed=0.2):
        """Move backward at specified speed."""
        return self.move(-speed, -speed)

    def turn_left(self, speed=0.2):
        """Turn left in place."""
        return self.move(-speed, speed)

    def turn_right(self, speed=0.2):
        """Turn right in place."""
        return self.move(speed, -speed)

    def get_imu_data(self):
        """Request IMU sensor data."""
        cmd = {"T": 126}
        self.ser.write(json.dumps(cmd).encode() + b'\n')
        # Read response (implement response parsing as needed)
        response = self.ser.readline().decode('utf-8')
        return response
```

### ROS2 Integration Pattern

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import json
import serial

class WaveRoverNode(Node):
    def __init__(self):
        super().__init__('wave_rover_node')

        # Serial setup
        self.serial = serial.Serial('/dev/ttyTHS1', 115200, timeout=1.0)
        self.serial.setRTS(False)
        self.serial.setDTR(False)

        # ROS2 setup
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        # Enable continuous feedback for real-time operation
        self.enable_continuous_feedback()

    def enable_continuous_feedback(self):
        """Enable continuous chassis feedback."""
        cmd = {"T": 131, "cmd": 1}
        self.send_command(cmd)

    def send_command(self, cmd_dict):
        """Send JSON command safely."""
        try:
            json_str = json.dumps(cmd_dict) + '\n'
            self.serial.write(json_str.encode())
            return True
        except Exception as e:
            self.get_logger().error(f"Serial error: {e}")
            return False

    def cmd_vel_callback(self, msg):
        """Convert Twist message to differential drive commands."""
        # Convert linear/angular to left/right wheel speeds
        linear = msg.linear.x
        angular = msg.angular.z

        # Differential drive kinematics
        wheel_base = 0.15  # meters (adjust for actual robot)
        left_speed = linear - (angular * wheel_base / 2.0)
        right_speed = linear + (angular * wheel_base / 2.0)

        # Clamp to valid range
        left_speed = max(-0.5, min(0.5, left_speed))
        right_speed = max(-0.5, min(0.5, right_speed))

        # Send movement command
        cmd = {"T": 1, "L": left_speed, "R": right_speed}
        self.send_command(cmd)
```

## Testing and Validation

### Using the Hardware Test Script

The project includes a comprehensive test script at `hardware_tests/test_waveroever_uart.py`:

```bash
# Interactive mode
python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1

# Automated test suite
python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --auto

# Specific test categories
python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test motor
python hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test imu
```

### Command Validation

Always validate commands before sending:

```python
def validate_speed_command(left, right):
    """Validate speed command parameters."""
    if not (-0.5 <= left <= 0.5):
        raise ValueError(f"Left speed {left} out of range [-0.5, 0.5]")
    if not (-0.5 <= right <= 0.5):
        raise ValueError(f"Right speed {right} out of range [-0.5, 0.5]")
    return True

def create_movement_command(left_speed, right_speed):
    """Create validated movement command."""
    validate_speed_command(left_speed, right_speed)
    return {"T": 1, "L": left_speed, "R": right_speed}
```

## Troubleshooting

### Common Issues

1. **No Serial Response**
   - Check port permissions: `sudo chmod 666 /dev/ttyTHS1`
   - Verify baudrate (115200)
   - Ensure RTS/DTR are disabled
   - Check physical connections

2. **Command Not Executed**
   - Verify JSON format (use `json.dumps()`)
   - Check command type (T parameter)
   - Ensure newline termination (`\n`)
   - Validate parameter ranges

3. **Inconsistent Movement**
   - Wave Rover has no encoders (open-loop control)
   - Speed values are PWM percentages, not absolute speeds
   - Surface conditions affect actual movement
   - Consider IMU feedback for closed-loop control

### Debug Commands

```python
# Enable echo to see sent commands
{"T": 143, "cmd": 1}

# Enable continuous feedback for monitoring
{"T": 131, "cmd": 1}

# Display status on OLED
{"T": 3, "lineNum": 0, "Text": "Debug Mode"}

# Test with minimal PWM
{"T": 11, "L": 50, "R": 50}
```

## Safety Considerations

1. **Speed Limits**: Keep speeds ≤ 0.3 for indoor testing
2. **Emergency Stop**: Always implement `{"T": 1, "L": 0, "R": 0}`
3. **Timeout**: Set serial timeouts to prevent blocking
4. **Error Handling**: Always catch and handle serial exceptions
5. **Hardware Limits**: Respect PWM ranges to avoid motor damage

## Related Documentation

- [Hardware Test Scripts](../../hardware_tests/README.md)
- [UART Configuration](../../config/uart_config.yaml)
- [Wave Rover Controller Node](../../src/actuation_nodes/)
- [Integration Test Examples](../../integration_tests/)

## Reference Links

- [Waveshare Wave Rover Official Documentation](https://www.waveshare.com/wiki/WAVE_ROVER)
- [Serial Communication Best Practices](./testing_strategy.md)
- [ROS2 Integration Guide](./quick_start.md)
