# Chassis state information from Wave Rover
# Parsed from UART continuous feedback mode

std_msgs/Header header

# Battery and power
float32 battery_voltage      # Battery voltage (V)
float32 power_consumption    # Current power consumption (W)

# Motor status
bool motors_enabled          # Are motors currently enabled
float32 left_motor_current   # Left motor current (A)
float32 right_motor_current  # Right motor current (A)

# IMU data (from Wave Rover's 9-axis IMU)
float32 heading_angle        # Heading angle (degrees)
float32 roll                 # Roll angle (degrees)
float32 pitch                # Pitch angle (degrees)
float32 yaw                  # Yaw angle (degrees)

geometry_msgs/Vector3 acceleration     # Linear acceleration (m/s²)
geometry_msgs/Vector3 angular_velocity # Angular velocity (rad/s)
geometry_msgs/Vector3 magnetic_field   # Magnetic field (µT)

# Temperature
float32 temperature          # IMU temperature (°C)

# System status
string status_message        # Human-readable status
bool emergency_stop_active   # Emergency stop state
