# Object detection result from YOLO

std_msgs/Header header

# Detection details
string object_class          # Object class name (e.g., "person", "bottle")
float32 confidence          # Detection confidence (0.0-1.0)
geometry_msgs/Point32 bbox_center    # Bounding box center (pixel coordinates)
float32 bbox_width          # Bounding box width (pixels)
float32 bbox_height         # Bounding box height (pixels)

# 3D information (if available from depth estimation)
bool has_3d_info            # Whether 3D information is available
geometry_msgs/Point position_3d      # 3D position in camera frame (meters)
float32 distance            # Distance from camera (meters)

# Tracking
uint32 track_id             # Unique ID for tracking across frames
bool is_new_detection       # Whether this is a new detection
