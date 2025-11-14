#!/usr/bin/env python3
"""
Depth Estimation Node for Local AI Robot Assistant
Performs monocular depth estimation using Depth Anything V2 TensorRT

Subscribes to undistorted camera images and publishes depth maps
Optimized for real-time performance on Jetson Orin Nano
"""

import time
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .depth_anything_v2_trt import DepthAnythingV2TRT


class DepthEstimationNode(Node):
    """
    ROS2 node for real-time depth estimation using Depth Anything V2

    Subscribes to:
        - /camera/image_undistorted: Undistorted camera images
        - /camera/camera_info: Camera calibration information

    Publishes:
        - /perception/depth: Raw depth maps
        - /perception/depth_colored: Colored depth visualizations
        - /perception/depth_stats: Depth statistics (min, max, mean)
        - /perception/obstacles: Obstacle detection based on depth
    """

    def __init__(self):
        super().__init__("depth_estimation_node")

        # Declare parameters
        self.declare_parameter("engine_path", "models/depth_trt/depth_anything_v2_small.trt")
        self.declare_parameter("config_path", "models/depth_trt/config.json")
        self.declare_parameter("publish_colored", True)
        self.declare_parameter("publish_stats", True)
        self.declare_parameter("publish_obstacles", True)
        self.declare_parameter("obstacle_threshold_m", 2.0)
        self.declare_parameter("obstacle_roi_height", 0.3)  # Bottom 30% of image
        self.declare_parameter("max_depth_m", 10.0)  # Maximum depth in meters
        self.declare_parameter("frame_skip", 1)  # Process every N-th frame

        # Get parameters
        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.publish_colored = (
            self.get_parameter("publish_colored").get_parameter_value().bool_value
        )
        self.publish_stats = self.get_parameter("publish_stats").get_parameter_value().bool_value
        self.publish_obstacles = (
            self.get_parameter("publish_obstacles").get_parameter_value().bool_value
        )
        self.obstacle_threshold = (
            self.get_parameter("obstacle_threshold_m").get_parameter_value().double_value
        )
        self.obstacle_roi_height = (
            self.get_parameter("obstacle_roi_height").get_parameter_value().double_value
        )
        self.max_depth = self.get_parameter("max_depth_m").get_parameter_value().double_value
        self.frame_skip = self.get_parameter("frame_skip").get_parameter_value().integer_value

        # Initialize depth estimation model
        try:
            self.depth_model = DepthAnythingV2TRT(engine_path, config_path)
            self.get_logger().info(f"Depth model loaded from: {engine_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load depth model: {e}")
            raise

        # CV bridge for image conversion
        self.bridge = CvBridge()

        # Camera info
        self.camera_info: Optional[CameraInfo] = None

        # Frame counting for skipping
        self.frame_count = 0

        # Performance tracking
        self.processing_times = []
        self.last_stats_time = time.time()

        # Publishers
        self.depth_pub = self.create_publisher(Image, "/perception/depth", 10)

        if self.publish_colored:
            self.depth_colored_pub = self.create_publisher(Image, "/perception/depth_colored", 10)

        if self.publish_stats:
            self.depth_stats_pub = self.create_publisher(
                Header, "/perception/depth_stats", 10  # Custom message would be better
            )

        if self.publish_obstacles:
            self.obstacles_pub = self.create_publisher(PolygonStamped, "/perception/obstacles", 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/image_undistorted", self.image_callback, 10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self.camera_info_callback, 10
        )

        # Performance reporting timer
        self.stats_timer = self.create_timer(10.0, self.publish_performance_stats)

        self.get_logger().info("Depth estimation node initialized")

    def camera_info_callback(self, msg: CameraInfo):
        """Handle camera calibration information"""
        self.camera_info = msg

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return

        start_time = time.time()

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Run depth estimation
            depth_map = self.depth_model.infer(cv_image, return_original_size=True)

            # Convert relative depth to metric depth (approximate)
            depth_metric = self._convert_to_metric_depth(depth_map)

            # Publish raw depth map
            self._publish_depth_map(depth_metric, msg.header)

            # Publish colored visualization if enabled
            if self.publish_colored:
                self._publish_colored_depth(depth_map, msg.header)

            # Publish depth statistics if enabled
            if self.publish_stats:
                self._publish_depth_stats(depth_metric, msg.header)

            # Publish obstacle detection if enabled
            if self.publish_obstacles:
                self._publish_obstacles(depth_metric, msg.header)

            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only recent measurements
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _convert_to_metric_depth(self, relative_depth: np.ndarray) -> np.ndarray:
        """
        Convert relative depth to approximate metric depth

        Note: Depth Anything V2 produces relative depth. For accurate metric depth,
        additional calibration with known objects or stereo vision would be needed.
        """
        # Simple linear scaling - this is an approximation
        # In practice, you'd want proper calibration
        depth_normalized = (relative_depth - relative_depth.min()) / (
            relative_depth.max() - relative_depth.min() + 1e-8
        )
        metric_depth = depth_normalized * self.max_depth

        return metric_depth

    def _publish_depth_map(self, depth_map: np.ndarray, header: Header):
        """Publish raw depth map"""
        try:
            # Convert to 32-bit float image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_map.astype(np.float32), encoding="32FC1")
            depth_msg.header = header
            depth_msg.header.frame_id = "camera_frame"

            self.depth_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish depth map: {e}")

    def _publish_colored_depth(self, depth_map: np.ndarray, header: Header):
        """Publish colored depth visualization"""
        try:
            # Create colored visualization
            depth_colored = self.depth_model.visualize_depth(depth_map)

            # Convert to ROS message
            colored_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
            colored_msg.header = header
            colored_msg.header.frame_id = "camera_frame"

            self.depth_colored_pub.publish(colored_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish colored depth: {e}")

    def _publish_depth_stats(self, depth_map: np.ndarray, header: Header):
        """Publish depth statistics"""
        try:
            # Calculate statistics
            valid_depths = depth_map[depth_map > 0]

            if len(valid_depths) > 0:
                # Note: Using Header message as placeholder
                # In production, create custom message type for depth statistics
                stats_msg = Header()
                stats_msg.stamp = header.stamp
                stats_msg.frame_id = f"depth_stats_min_{valid_depths.min():.2f}_max_\
                    {valid_depths.max():.2f}_mean_{valid_depths.mean():.2f}"

                self.depth_stats_pub.publish(stats_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish depth stats: {e}")

    def _publish_obstacles(self, depth_map: np.ndarray, header: Header):
        """Detect and publish obstacles based on depth"""
        try:
            height, width = depth_map.shape

            # Define ROI for obstacle detection (bottom portion of image)
            roi_start = int(height * (1.0 - self.obstacle_roi_height))
            roi_depth = depth_map[roi_start:, :]

            # Find close obstacles
            close_pixels = np.where((roi_depth > 0) & (roi_depth < self.obstacle_threshold))

            if len(close_pixels[0]) > 0:
                # Create obstacle polygon (simplified to bounding box)
                obstacle_msg = PolygonStamped()
                obstacle_msg.header = header
                obstacle_msg.header.frame_id = "camera_frame"

                # Find bounding box of close pixels
                min_y, max_y = close_pixels[0].min(), close_pixels[0].max()
                min_x, max_x = close_pixels[1].min(), close_pixels[1].max()

                # Convert to normalized coordinates
                points = [
                    Point32(
                        x=float(min_x) / width,
                        y=float(roi_start + min_y) / height,
                        z=0.0,
                    ),
                    Point32(
                        x=float(max_x) / width,
                        y=float(roi_start + min_y) / height,
                        z=0.0,
                    ),
                    Point32(
                        x=float(max_x) / width,
                        y=float(roi_start + max_y) / height,
                        z=0.0,
                    ),
                    Point32(
                        x=float(min_x) / width,
                        y=float(roi_start + max_y) / height,
                        z=0.0,
                    ),
                ]

                obstacle_msg.polygon.points = points
                self.obstacles_pub.publish(obstacle_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to publish obstacles: {e}")

    def publish_performance_stats(self):
        """Publish performance statistics"""
        if not self.processing_times:
            return

        # Get model performance stats
        model_stats = self.depth_model.get_performance_stats()

        # Calculate node processing stats
        avg_processing_time = np.mean(self.processing_times)
        _ = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

        current_time = time.time()
        time_since_last = current_time - self.last_stats_time
        frames_processed = len(self.processing_times)
        effective_fps = frames_processed / time_since_last if time_since_last > 0 else 0.0

        self.get_logger().info(
            f"Depth estimation performance - "
            f"Model: {model_stats['avg_fps']:.1f} FPS ({model_stats['avg_inference_time']*1000:.1f}\
                  ms), "
            f"Node: {effective_fps:.1f} FPS ({avg_processing_time*1000:.1f} ms avg)"
        )

        # Reset for next period
        self.processing_times.clear()
        self.last_stats_time = current_time

    def destroy_node(self):
        """Clean up resources on shutdown"""
        try:
            self.depth_model.cleanup()
            self.get_logger().info("Depth estimation node shutdown complete")
        except Exception as e:
            self.get_logger().error(f"Error during shutdown: {e}")

        super().destroy_node()


def main(args=None):
    """Main entry point for depth estimation node"""
    rclpy.init(args=args)

    try:
        node = DepthEstimationNode()

        node.get_logger().info("Depth estimation node started")
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running depth estimation node: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
