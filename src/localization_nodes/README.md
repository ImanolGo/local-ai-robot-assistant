# IMU Node Documentation

This document describes the IMU node implementation in the `localization_nodes` package for acquiring 9-axis IMU data from the Wave Rover chassis.

## Overview

The `uart_imu_node` provides periodic IMU data acquisition from the Wave Rover's built-in 9-axis IMU sensor via UART communication. It processes raw sensor data and publishes properly formatted ROS2 IMU messages with coordinate frame transformations and covariance matrices.

## Features

- **Periodic IMU Queries**: Configurable rate (default 20 Hz)
- **Data Validation**: Comprehensive validation of received IMU data
- **Coordinate Transformations**: Euler angles to quaternion conversion
- **ROS2 Integration**: Standard sensor_msgs/Imu message format
- **Error Handling**: Robust communication error recovery
- **Thread Safety**: Safe concurrent operation

## Node Configuration

### Parameters

```yaml
uart:
  port: "/dev/ttyTHS1"        # Serial port
  baudrate: 115200            # Baud rate
  timeout: 1.0                # Read timeout (seconds)

imu:
  query_rate: 20.0            # IMU query rate (Hz)
  query_command: 126          # Command type for IMU query
  validate_data: true         # Enable data validation
  acceleration_limit: 50.0    # Max valid acceleration (m/s²)

frame_id: "imu_link"          # IMU coordinate frame
```

### Topics

**Published**:
- `/imu/data` (sensor_msgs/Imu) - Processed IMU data with covariances

**Subscribed**: None

### Services

None

## IMU Data Format

### Raw Data from Wave Rover

The Wave Rover returns IMU data in the following JSON format:

```json
{
  "roll": 5.2,      // Roll angle (degrees)
  "pitch": -2.1,    // Pitch angle (degrees)
  "yaw": 45.8,      // Yaw angle (degrees)
  "AccX": 0.98,     // X acceleration (g)
  "AccY": 0.05,     // Y acceleration (g)
  "AccZ": 0.12,     // Z acceleration (g)
  "GyroX": 0.02,    // X angular velocity (deg/s)
  "GyroY": -0.01,   // Y angular velocity (deg/s)
  "GyroZ": 0.15     // Z angular velocity (deg/s)
}
```

### ROS2 IMU Message

The processed data is published as `sensor_msgs/Imu`:

- **Orientation**: Quaternion computed from Euler angles
- **Angular Velocity**: Converted to rad/s
- **Linear Acceleration**: Converted to m/s² (from g-force)
- **Covariances**: Appropriate uncertainty estimates

## Usage

### Launch the Node

```bash
ros2 run localization_nodes uart_imu_node
```

### With Parameters

```bash
ros2 run localization_nodes uart_imu_node --ros-args \
  -p uart.port:="/dev/ttyTHS1" \
  -p imu.query_rate:=20.0 \
  -p frame_id:="imu_link"
```

### Subscribe to IMU Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class IMUSubscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

    def imu_callback(self, msg):
        # Extract orientation quaternion
        q = msg.orientation
        print(f"Orientation: w={q.w:.3f}, x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}")

        # Extract angular velocity (rad/s)
        omega = msg.angular_velocity
        print(f"Angular velocity: x={omega.x:.3f}, y={omega.y:.3f}, z={omega.z:.3f}")

        # Extract linear acceleration (m/s²)
        acc = msg.linear_acceleration
        print(f"Acceleration: x={acc.x:.3f}, y={acc.y:.3f}, z={acc.z:.3f}")
```

## Implementation Details

### Data Validation

The node performs comprehensive validation of received IMU data:

1. **Required Fields**: Checks for all necessary IMU fields
2. **Data Types**: Validates numeric types for all values
3. **Range Checking**: Ensures angles are within [-180, 180] degrees
4. **Acceleration Limits**: Rejects extreme acceleration values
5. **JSON Format**: Validates proper JSON structure

### Coordinate Frame Conventions

The IMU data follows the standard ROS coordinate conventions:

- **X-axis**: Forward (robot front)
- **Y-axis**: Left (robot left side)
- **Z-axis**: Up (robot top)
- **Roll**: Rotation about X-axis
- **Pitch**: Rotation about Y-axis
- **Yaw**: Rotation about Z-axis

### Error Handling

The node implements robust error handling for:

- **Serial Communication Errors**: Automatic retry and recovery
- **JSON Parse Errors**: Graceful handling of malformed data
- **Data Validation Failures**: Rejection of invalid sensor readings
- **Timeout Handling**: Detection of communication timeouts

## Testing

### Unit Tests

```bash
# Run IMU node unit tests
python3 -m pytest src/localization_nodes/test/test_uart_imu_node.py -v
```

### Integration Tests

```bash
# Test with actual hardware
python3 integration_tests/test_uart_integration.py --port /dev/ttyTHS1

# Test IMU functionality specifically
python3 integration_tests/test_uart_integration.py --port /dev/ttyTHS1 -k test_imu
```

### Manual Testing

```bash
# Test IMU communication directly
python3 hardware_tests/test_waveroever_uart.py --port /dev/ttyTHS1 --test imu

# Monitor IMU data
ros2 topic echo /imu/data
```

## Troubleshooting

### Common Issues

1. **No IMU Data Published**
   - Check serial port connection
   - Verify Wave Rover is powered and responding
   - Check node logs for error messages

2. **Invalid IMU Data**
   - Ensure Wave Rover firmware is up to date
   - Check for electromagnetic interference
   - Verify IMU sensor calibration

3. **High Latency**
   - Reduce query rate if UART buffer overflows
   - Check system CPU usage
   - Verify serial port performance

### Debug Commands

```bash
# Check node status
ros2 node info /uart_imu_node

# Monitor topic publication rate
ros2 topic hz /imu/data

# View detailed IMU messages
ros2 topic echo /imu/data --once

# Check node logs
ros2 log get /uart_imu_node
```

## Performance

### Specifications

- **Update Rate**: Up to 50 Hz (limited by UART bandwidth)
- **Latency**: < 20ms (query + processing)
- **CPU Usage**: < 2% (Jetson Orin Nano)
- **Memory Usage**: < 30MB

### Optimization Tips

1. **Adjust Query Rate**: Lower rate reduces CPU and UART usage
2. **Disable Validation**: For maximum performance (not recommended)
3. **Use Larger Buffers**: Increase serial buffer size if needed

## Related Documentation

- **UART Protocol**: See actuation_nodes/README.md for protocol details
- **Wave Rover Manual**: Hardware specifications and calibration
- **ROS2 IMU Integration**: sensor_msgs/Imu message documentation

---

*This node is part of the Local AI Robot Assistant project. For more information, see the main project documentation.*
