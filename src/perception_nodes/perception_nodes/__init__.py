"""Computer vision and perception nodes for the Local AI Robot Assistant.

This package contains nodes for:
- Camera capture and calibration
- Image undistortion for fisheye cameras
- Object detection using YOLO
- Depth estimation using FastDepth
- Point cloud generation
"""

__all__ = [
    "camera_driver",
    "image_undistort_node",
    "object_detector",
    "depth_estimator",
]
__version__ = "0.1.0"
