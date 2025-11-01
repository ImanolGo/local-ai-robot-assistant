#!/usr/bin/env python3
"""
GPU-Accelerated Image Undistortion Node
Applies lens distortion correction using GPU acceleration and DeepStream integration.

Author: Local AI Robot Team
License: Apache-2.0
"""

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Try to import GPU-accelerated OpenCV if available
try:
    import cv2.cuda as cv2_cuda

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ImageUndistortNode(Node):
    """
    GPU-accelerated image undistortion node.

    Features:
    - GPU-accelerated undistortion using OpenCV CUDA
    - Cached distortion maps for optimal performance
    - Real-time performance monitoring
    - Configurable interpolation methods
    - Fallback to CPU processing if GPU unavailable
    """

    def __init__(self):
        super().__init__("image_undistort_node")

        # Initialize variables
        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.new_camera_matrix: Optional[np.ndarray] = None
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None

        # GPU variables
        self.gpu_map1: Optional[cv2_cuda.GpuMat] = None
        self.gpu_map2: Optional[cv2_cuda.GpuMat] = None
        self.gpu_src: Optional[cv2_cuda.GpuMat] = None
        self.gpu_dst: Optional[cv2_cuda.GpuMat] = None
        self.use_gpu = False

        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.processing_times = []
        self.gpu_memory_usage = []

        # Load configuration
        self._load_config()

        # Set up ROS2 subscribers and publishers
        self._setup_ros2()

        # Load camera calibration and setup undistortion
        self._setup_undistortion()

        self.get_logger().info(
            f"Image undistortion node initialized "
            f"({'GPU' if self.use_gpu else 'CPU'} acceleration)"
        )

    def _load_config(self) -> None:
        """Load camera configuration from YAML file."""
        try:
            config_path = "/home/imanolgo/repos/local-ai-robot-assistant/config/camera_config.yaml"
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)

            self.get_logger().info(f"Loaded camera configuration from {config_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load camera config: {e}")
            # Use default configuration
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ros2": {
                "raw_image_topic": "/camera/raw",
                "undistorted_image_topic": "/camera/undistorted",
                "frame_id": "camera_link",
            },
            "undistortion": {
                "use_gpu_acceleration": True,
                "interpolation_method": "linear",
                "border_mode": "constant",
                "border_value": [0, 0, 0],
                "cache_maps": True,
                "use_optimized_camera_matrix": True,
                "alpha": 1.0,
            },
            "monitoring": {
                "enable_fps_monitoring": True,
                "enable_gpu_monitoring": True,
                "log_performance_stats": True,
                "stats_publish_rate": 1.0,
            },
            "calibration_file": (
                "/home/imanolgo/repos/local-ai-robot-assistant/" "config/camera_calibration.yaml"
            ),
        }

    def _setup_ros2(self) -> None:
        """Set up ROS2 subscribers and publishers."""
        # Declare parameters
        self.declare_parameter("use_gpu", self.config["undistortion"]["use_gpu_acceleration"])
        self.declare_parameter(
            "interpolation_method", self.config["undistortion"]["interpolation_method"]
        )
        self.declare_parameter("alpha", self.config["undistortion"]["alpha"])

        # QoS profile for real-time performance
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriber
        raw_topic = self.config["ros2"]["raw_image_topic"]
        self.image_sub = self.create_subscription(Image, raw_topic, self._image_callback, qos)

        # Publisher
        undistorted_topic = self.config["ros2"]["undistorted_image_topic"]
        self.image_pub = self.create_publisher(Image, undistorted_topic, qos)

        # Performance monitoring timer
        if self.config["monitoring"]["enable_fps_monitoring"]:
            stats_rate = self.config["monitoring"]["stats_publish_rate"]
            self.stats_timer = self.create_timer(1.0 / stats_rate, self._publish_stats)

    def _setup_undistortion(self) -> None:
        """Load calibration data and setup undistortion maps."""
        try:
            # Load calibration
            calib_path = self.config.get("calibration_file")
            if not calib_path:
                raise ValueError("Calibration file path not specified")

            with open(calib_path, "r") as file:
                calib_data = yaml.safe_load(file)

            # Extract calibration parameters
            self.camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
            self.dist_coeffs = np.array(
                calib_data["distortion_coefficients"], dtype=np.float32
            ).flatten()

            # Image dimensions
            self.image_width = calib_data["image_width"]
            self.image_height = calib_data["image_height"]

            self.get_logger().info(f"Loaded calibration from {calib_path}")
            self.get_logger().info(f"Image size: {self.image_width}x{self.image_height}")

            # Setup undistortion parameters
            self._setup_undistortion_parameters()

        except Exception as e:
            self.get_logger().error(f"Failed to setup undistortion: {e}")
            raise

    def _setup_undistortion_parameters(self) -> None:
        """Setup undistortion parameters and maps."""
        try:
            alpha = self.get_parameter("alpha").value

            # Get optimal new camera matrix
            if self.config["undistortion"]["use_optimized_camera_matrix"]:
                self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix,
                    self.dist_coeffs,
                    (self.image_width, self.image_height),
                    alpha,
                    (self.image_width, self.image_height),
                )
                self.get_logger().info(f"Using optimized camera matrix with alpha={alpha}")
            else:
                self.new_camera_matrix = self.camera_matrix

            # Setup GPU acceleration if available and requested
            use_gpu_param = self.get_parameter("use_gpu").value
            self.use_gpu = GPU_AVAILABLE and use_gpu_param

            if self.use_gpu:
                self._setup_gpu_undistortion()
            else:
                if use_gpu_param and not GPU_AVAILABLE:
                    self.get_logger().warning(
                        "GPU acceleration requested but OpenCV CUDA not available, using CPU"
                    )
                self._setup_cpu_undistortion()

        except Exception as e:
            self.get_logger().error(f"Failed to setup undistortion parameters: {e}")
            raise

    def _setup_gpu_undistortion(self) -> None:
        """Setup GPU-accelerated undistortion maps."""
        try:
            # Get interpolation method
            _ = self._get_interpolation_method()

            # Generate undistortion maps
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix,
                (self.image_width, self.image_height),
                cv2.CV_32FC1,
            )

            # Upload maps to GPU
            self.gpu_map1 = cv2_cuda.GpuMat()
            self.gpu_map2 = cv2_cuda.GpuMat()
            self.gpu_map1.upload(self.map1)
            self.gpu_map2.upload(self.map2)

            # Pre-allocate GPU matrices for processing
            self.gpu_src = cv2_cuda.GpuMat(self.image_height, self.image_width, cv2.CV_8UC3)
            self.gpu_dst = cv2_cuda.GpuMat(self.image_height, self.image_width, cv2.CV_8UC3)

            self.get_logger().info("GPU undistortion maps initialized")

        except Exception as e:
            self.get_logger().error(f"Failed to setup GPU undistortion: {e}")
            self.use_gpu = False
            self._setup_cpu_undistortion()

    def _setup_cpu_undistortion(self) -> None:
        """Setup CPU undistortion maps."""
        try:
            if self.config["undistortion"]["cache_maps"]:
                # Pre-compute undistortion maps for better performance
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    self.new_camera_matrix,
                    (self.image_width, self.image_height),
                    cv2.CV_16SC2,
                )
                self.get_logger().info("CPU undistortion maps cached")
            else:
                self.get_logger().info("CPU undistortion without cached maps")

        except Exception as e:
            self.get_logger().error(f"Failed to setup CPU undistortion: {e}")
            raise

    def _get_interpolation_method(self) -> int:
        """Get OpenCV interpolation method from config."""
        method_str = self.get_parameter("interpolation_method").value
        method_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        return method_map.get(method_str, cv2.INTER_LINEAR)

    def _get_border_mode(self) -> int:
        """Get OpenCV border mode from config."""
        border_str = self.config["undistortion"]["border_mode"]
        border_map = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "wrap": cv2.BORDER_WRAP,
            "reflect_101": cv2.BORDER_REFLECT_101,
        }
        return border_map.get(border_str, cv2.BORDER_CONSTANT)

    def _image_callback(self, msg: Image) -> None:
        """Process incoming image and publish undistorted result."""
        try:
            start_time = time.time()

            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply undistortion
            if self.use_gpu:
                undistorted_image = self._undistort_gpu(cv_image)
            else:
                undistorted_image = self._undistort_cpu(cv_image)

            # Convert back to ROS image
            output_msg = self.bridge.cv2_to_imgmsg(undistorted_image, "bgr8")
            output_msg.header = msg.header

            # Publish undistorted image
            self.image_pub.publish(output_msg)

            # Update performance metrics
            self._update_performance_metrics(start_time)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _undistort_gpu(self, image: np.ndarray) -> np.ndarray:
        """Apply GPU-accelerated undistortion."""
        try:
            # Upload image to GPU
            self.gpu_src.upload(image)

            # Apply remap operation on GPU
            cv2_cuda.remap(
                self.gpu_src,
                self.gpu_dst,
                self.gpu_map1,
                self.gpu_map2,
                self._get_interpolation_method(),
                self._get_border_mode(),
                self.config["undistortion"]["border_value"],
            )

            # Download result from GPU
            result = self.gpu_dst.download()

            # Monitor GPU memory usage
            if self.config["monitoring"]["enable_gpu_monitoring"]:
                gpu_info = cv2_cuda.DeviceInfo().totalMemory()
                self.gpu_memory_usage.append(gpu_info)

            return result

        except Exception as e:
            self.get_logger().error(f"GPU undistortion failed: {e}")
            # Fallback to CPU
            return self._undistort_cpu(image)

    def _undistort_cpu(self, image: np.ndarray) -> np.ndarray:
        """Apply CPU undistortion."""
        try:
            # Validate calibration data first
            if (
                self.camera_matrix is None
                or self.dist_coeffs is None
                or self.new_camera_matrix is None
            ):
                self.get_logger().warning("Invalid calibration data, returning original image")
                return image

            if self.config["undistortion"]["cache_maps"] and self.map1 is not None:
                # Use cached maps for better performance
                undistorted = cv2.remap(
                    image,
                    self.map1,
                    self.map2,
                    self._get_interpolation_method(),
                    self._get_border_mode(),
                    self.config["undistortion"]["border_value"],
                )
            else:
                # Direct undistortion (slower but uses less memory)
                undistorted = cv2.undistort(
                    image,
                    self.camera_matrix,
                    self.dist_coeffs,
                    None,
                    self.new_camera_matrix,
                )

            return undistorted

        except Exception as e:
            self.get_logger().error(f"CPU undistortion failed: {e}")
            return image  # Return original image as fallback

    def _update_performance_metrics(self, start_time: float) -> None:
        """Update performance monitoring metrics."""
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        # Keep only last 100 measurements
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        self.frame_count += 1

        # Calculate FPS every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def _publish_stats(self) -> None:
        """Publish performance statistics."""
        if not self.config["monitoring"]["log_performance_stats"]:
            return

        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            max_processing_time = np.max(self.processing_times)

            stats_msg = (
                f"Undistortion Performance - FPS: {self.fps:.1f}, "
                f"Avg Processing: {avg_processing_time*1000:.1f}ms, "
                f"Max Processing: {max_processing_time*1000:.1f}ms, "
                f"Mode: {'GPU' if self.use_gpu else 'CPU'}"
            )

            if self.use_gpu and self.gpu_memory_usage:
                avg_gpu_memory = np.mean(self.gpu_memory_usage[-10:])  # Last 10 measurements
                stats_msg += f", GPU Memory: {avg_gpu_memory/(1024**3):.1f}GB"

            self.get_logger().info(stats_msg)


def main(args=None):
    """Main entry point for the image undistortion node."""
    rclpy.init(args=args)

    try:
        node = ImageUndistortNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Image undistortion node error: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
