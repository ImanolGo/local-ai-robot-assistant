# Depth image with metadata for point cloud generation

std_msgs/Header header

sensor_msgs/Image depth_image        # Raw depth image
sensor_msgs/CameraInfo camera_info   # Camera intrinsics for back-projection

# Depth information
float32 min_depth           # Minimum depth value (meters)
float32 max_depth           # Maximum depth value (meters)
float32 depth_scale         # Scale factor for depth values
string encoding             # Depth image encoding (e.g., "16UC1", "32FC1")

# Processing metadata
float32 inference_time      # Time taken for depth estimation (seconds)
string model_name           # Name of depth estimation model used
bool is_filtered            # Whether depth image has been filtered
