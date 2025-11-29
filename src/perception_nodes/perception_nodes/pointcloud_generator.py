#!/usr/bin/env python3
"""
Point Cloud Generator Node
Generates 3D point clouds from depth maps and RGB images.

Subscribes to:
- /perception/depth (sensor_msgs/Image): Depth map
- /camera/undistorted (sensor_msgs/Image): RGB image
- /camera_info (sensor_msgs/CameraInfo): Camera calibration

Publishes to:
- /perception/pointcloud (sensor_msgs/PointCloud2): RGB point cloud
"""

import array
import time
from typing import Optional, Tuple

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class PointCloudGenerator(Node):
    """Generates RGB point clouds from depth maps and images."""

    def __init__(self):
        super().__init__("pointcloud_generator")

        # Declare parameters
        self.declare_parameter("depth_range_min", 0.1)  # meters
        self.declare_parameter("depth_range_max", 10.0)  # meters
        self.declare_parameter("downsample_factor", 2)  # Reduce point cloud density
        self.declare_parameter("enable_rgb", True)
        self.declare_parameter("publish_rate", 10.0)  # Hz

        # Get parameters
        self.depth_min = self.get_parameter("depth_range_min").value
        self.depth_max = self.get_parameter("depth_range_max").value
        self.downsample = self.get_parameter("downsample_factor").value
        self.enable_rgb = self.get_parameter("enable_rgb").value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera intrinsics (will be set from camera_info)
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.camera_frame = "camera_link"

        # Create camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, 10
        )

        # Create synchronized subscribers for depth and RGB
        self.depth_sub = message_filters.Subscriber(self, Image, "/perception/depth")
        self.rgb_sub = message_filters.Subscriber(self, Image, "/camera/undistorted")

        # Synchronize depth and RGB messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.rgb_sub], queue_size=10, slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.synchronized_callback)

        # Create publisher
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/perception/pointcloud", 10)

        # Performance monitoring
        self.frame_count = 0
        self.total_time = 0.0
        self.fps_timer = self.create_timer(5.0, self.log_performance)

        self.get_logger().info("Point cloud generator initialized")

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from camera info message."""
        if self.fx is None:
            # Camera matrix K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_frame = msg.header.frame_id

            self.get_logger().info(
                f"Camera intrinsics loaded: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}"
            )

    def synchronized_callback(self, depth_msg: Image, rgb_msg: Image):
        """Process synchronized depth and RGB messages."""
        if self.fx is None:
            self.get_logger().warn("Waiting for camera calibration...", throttle_duration_sec=5.0)
            return

        try:
            start_time = time.time()

            # Convert ROS Images to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

            # Ensure images have same dimensions
            if depth_image.shape[:2] != rgb_image.shape[:2]:
                # Resize depth to match RGB
                depth_image = self._resize_depth(depth_image, rgb_image.shape[:2])

            # Generate point cloud
            points = self.generate_pointcloud(depth_image, rgb_image)

            # Convert to PointCloud2 message
            pointcloud_msg = self.create_pointcloud2_msg(points, rgb_msg.header)

            # Publish
            self.pointcloud_pub.publish(pointcloud_msg)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.frame_count += 1
            self.total_time += processing_time

        except Exception as e:
            self.get_logger().error(f"Error generating point cloud: {e}")

    def _resize_depth(self, depth: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize depth map to match target shape."""
        import cv2

        return cv2.resize(
            depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
        )

    def generate_pointcloud(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Generate 3D point cloud from depth map and RGB image.

        Args:
            depth: Depth map (H x W) in meters
            rgb: RGB image (H x W x 3)

        Returns:
            Point cloud array (N x 6) with [x, y, z, r, g, b]
        """
        h, w = depth.shape

        # Create coordinate grids with downsampling
        y_coords, x_coords = np.mgrid[0 : h : self.downsample, 0 : w : self.downsample]

        # Sample depth and RGB at grid points
        depth_sampled = depth[:: self.downsample, :: self.downsample]

        if self.enable_rgb:
            rgb_sampled = rgb[:: self.downsample, :: self.downsample]

        # Filter by depth range
        valid_mask = (depth_sampled > self.depth_min) & (depth_sampled < self.depth_max)

        # Extract valid points
        z = depth_sampled[valid_mask]
        y_valid = y_coords[valid_mask]
        x_valid = x_coords[valid_mask]

        # Back-project to 3D using pinhole camera model
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        # Z = depth
        x_3d = (x_valid - self.cx) * z / self.fx
        y_3d = (y_valid - self.cy) * z / self.fy
        z_3d = z

        # Create point cloud array
        if self.enable_rgb:
            # Extract RGB values (convert BGR to RGB)
            rgb_valid = rgb_sampled[valid_mask]
            r = rgb_valid[:, 2].astype(np.float32)
            g = rgb_valid[:, 1].astype(np.float32)
            b = rgb_valid[:, 0].astype(np.float32)

            # Combine XYZ and RGB
            points = np.column_stack((x_3d, y_3d, z_3d, r, g, b))
        else:
            # XYZ only
            points = np.column_stack((x_3d, y_3d, z_3d))

        return points

    def create_pointcloud2_msg(self, points: np.ndarray, header) -> PointCloud2:
        """
        Create PointCloud2 message from point array.
        Optimized version using direct byte buffer manipulation.

        Args:
            points: Point cloud array (N x 6) with [x, y, z, r, g, b]
            header: ROS message header

        Returns:
            PointCloud2 message
        """
        # Define point cloud fields
        if self.enable_rgb and points.shape[1] == 6:
            # XYZRGB format
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            # Pack RGB into single float32 (via uint32 view)
            # This is a bit magic: ROS expects a float32 that is actually a packed uint32
            rgb_packed = (
                (points[:, 3].astype(np.uint32) << 16)  # R
                | (points[:, 4].astype(np.uint32) << 8)  # G
                | (points[:, 5].astype(np.uint32))  # B
            )
            rgb_float = rgb_packed.view(np.float32)

            # Create output array directly
            # We need (N, 4) float32 array: [x, y, z, rgb]
            # Allocate uninitialized memory for speed
            cloud_data = np.empty((len(points), 4), dtype=np.float32)

            # Fill data
            cloud_data[:, 0] = points[:, 0]  # x
            cloud_data[:, 1] = points[:, 1]  # y
            cloud_data[:, 2] = points[:, 2]  # z
            cloud_data[:, 3] = rgb_float  # rgb

            point_step = 16

        else:
            # XYZ only format
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]

            # Create output array directly
            cloud_data = np.empty((len(points), 3), dtype=np.float32)

            # Fill data
            cloud_data[:, 0] = points[:, 0]  # x
            cloud_data[:, 1] = points[:, 1]  # y
            cloud_data[:, 2] = points[:, 2]  # z

            point_step = 12

        # Create PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.header.frame_id = self.camera_frame

        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = point_step
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
        pointcloud_msg.is_dense = False

        # Optimized assignment: Convert to array.array to bypass slow rclpy validation
        pointcloud_msg.data = array.array("B", cloud_data.tobytes())

        return pointcloud_msg

    def log_performance(self):
        """Log performance metrics."""
        if self.frame_count > 0:
            avg_fps = self.frame_count / 5.0
            avg_latency = (self.total_time / self.frame_count) * 1000  # ms

            self.get_logger().info(f"Performance: {avg_fps:.1f} FPS, {avg_latency:.1f} ms latency")

            # Reset counters
            self.frame_count = 0
            self.total_time = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
