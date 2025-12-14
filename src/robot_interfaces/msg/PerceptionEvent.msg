std_msgs/Header header

# Event Types
uint8 ENTERED_FOV=1
uint8 LEFT_FOV=2
uint8 MOVED_SIGNIFICANTLY=3
uint8 STATIONARY=4

uint8 event_type

# Object Details
string class_name      # e.g., "person", "cup"
int32 tracking_id      # Persistent ID from tracker (changed to int32 to match YOLO definition)
float32 confidence

# Position (optional, for MOVED events)
geometry_msgs/Point position
