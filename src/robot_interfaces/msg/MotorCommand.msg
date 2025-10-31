# Motor command for direct wheel speed control
# Used for UART communication with Wave Rover

float32 left_speed   # Left wheel speed (-0.5 to +0.5)
float32 right_speed  # Right wheel speed (-0.5 to +0.5)
bool enable          # Enable/disable motors
uint32 command_id    # Command ID for tracking
