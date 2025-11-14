#!/usr/bin/env python3
"""
Depth estimation ROS2 node using Depth Anything V2 with pipeline approach.
Updated to use HuggingFace transformers pipeline for compatibility.
"""

import json
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, Header
from transformers import pipeline

logger = logging.getLogger(__name__)


class DepthAnythingV2Pipeline:
    """
    Depth Anything V2 inference using HuggingFace transformers pipeline.
    This is a fallback implementation while ONNX/TensorRT conversion issues are resolved.
    """

    def __init__(self, model_dir: str):
        """Initialize the pipeline-based depth estimator."""
        self.model_dir = Path(model_dir)
        self.config_path = self.model_dir / "config.json"

        # Load configuration
        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.model_name = self.config.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")
        self.input_size = self.config.get("input_size", 384)
        self.device = "cpu"  # Force CPU for stability

        logger.info(f"Loading Depth Anything V2 pipeline: {self.model_name}")

        # Initialize pipeline
        self.pipe = pipeline(
            "depth-estimation",
            model=self.model_name,
            device=-1,  # CPU
            torch_dtype=torch.float32,
        )

        logger.info("Depth estimation pipeline loaded successfully")

        # Performance tracking
        self.inference_times = []
        self.frame_count = 0

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth map from input image.

        Args:
            image: Input RGB image [H, W, 3] uint8

        Returns:
            depth_map: Depth map [H, W] float32
        """
        start_time = time.time()

        try:
            # Convert to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = PILImage.fromarray(image)

            # Run inference
            result = self.pipe(pil_image)

            # Extract depth map
            if isinstance(result, dict):
                if "predicted_depth" in result:
                    depth_tensor = result["predicted_depth"]
                elif "depth" in result:
                    depth_tensor = result["depth"]
                else:
                    raise ValueError(f"Unexpected result format: {result.keys()}")
            else:
                depth_tensor = result

            # Convert to numpy
            if hasattr(depth_tensor, "numpy"):
                depth_map = depth_tensor.numpy()
            else:
                depth_map = np.array(depth_tensor)

            # Ensure float32
            depth_map = depth_map.astype(np.float32)

            # Record performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.frame_count += 1

            if self.frame_count % 10 == 0:
                avg_time = np.mean(self.inference_times[-10:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(f"Avg inference time: {avg_time:.3f}s, FPS: {fps:.1f}")

            return depth_map

        except Exception as e:
            logger.error(f"Depth prediction failed: {e}")
            # Return dummy depth map on failure
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    def get_stats(self) -> dict:
        """Get inference statistics."""
        if not self.inference_times:
            return {"frames": 0, "avg_fps": 0, "avg_time_ms": 0}

        avg_time = np.mean(self.inference_times)
        return {
            "frames": self.frame_count,
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "avg_time_ms": avg_time * 1000,
            "model_name": self.model_name,
            "device": self.device,
        }


class DepthEstimationNode(Node):
    """ROS2 node for depth estimation using Depth Anything V2."""

    def __init__(self):
        super().__init__("depth_estimation_node")

        # Parameters
        self.declare_parameter(
            "model_dir",
            "/home/imanolgo/repos/local-ai-robot-assistant/models/depth_trt",
        )
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/depth/image_raw")
        self.declare_parameter("pointcloud_topic", "/depth/points")
        self.declare_parameter("publish_pointcloud", False)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("min_depth", 0.1)

        # Get parameters
        model_dir = self.get_parameter("model_dir").value
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.publish_pointcloud = self.get_parameter("publish_pointcloud").value
        self.max_depth = self.get_parameter("max_depth").value
        self.min_depth = self.get_parameter("min_depth").value

        self.get_logger().info("Initializing depth estimation node")
        self.get_logger().info(f"Model directory: {model_dir}")
        self.get_logger().info(f"Input topic: {input_topic}")
        self.get_logger().info(f"Output topic: {output_topic}")

        # Initialize components
        self.bridge = CvBridge()

        # Initialize depth estimator
        try:
            self.depth_estimator = DepthAnythingV2Pipeline(model_dir)
            self.get_logger().info("Depth estimator initialized successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize depth estimator: {e}")
            raise

        # QoS profiles
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, input_topic, self.image_callback, image_qos
        )

        # Publishers
        self.depth_pub = self.create_publisher(Image, output_topic, image_qos)

        if self.publish_pointcloud:
            self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, image_qos)

        # Statistics publisher
        self.stats_pub = self.create_publisher(Float32MultiArray, "/depth/stats", image_qos)

        # Timer for statistics
        self.create_timer(5.0, self.publish_stats)

        self.get_logger().info("Depth estimation node ready")

        # Processing state
        self.processing = False
        self.last_process_time = 0.0

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        if self.processing:
            self.get_logger().debug("Skipping frame - still processing previous")
            return

        self.processing = True
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            # Run depth estimation
            depth_map = self.depth_estimator.predict(cv_image)

            # Normalize depth for visualization and clamp to valid range
            depth_normalized = np.clip(depth_map, self.min_depth, self.max_depth)
            depth_vis = (
                (depth_normalized - self.min_depth) / (self.max_depth - self.min_depth) * 255
            ).astype(np.uint8)

            # Convert to ROS message
            depth_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="mono8")
            depth_msg.header = msg.header
            depth_msg.header.frame_id = msg.header.frame_id

            # Publish depth image
            self.depth_pub.publish(depth_msg)

            # Publish point cloud if enabled
            if self.publish_pointcloud:
                pointcloud_msg = self.create_pointcloud(depth_map, msg.header, cv_image.shape)
                self.pointcloud_pub.publish(pointcloud_msg)

            # Log performance periodically
            process_time = time.time() - start_time
            self.last_process_time = process_time

            if (
                hasattr(self.depth_estimator, "frame_count")
                and self.depth_estimator.frame_count % 20 == 0
            ):
                stats = self.depth_estimator.get_stats()
                self.get_logger().info(
                    f"Processed {stats['frames']} frames, "
                    f"avg FPS: {stats['avg_fps']:.1f}, "
                    f"processing time: {process_time:.3f}s"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

        finally:
            self.processing = False

    def create_pointcloud(
        self, depth_map: np.ndarray, header: Header, image_shape: Tuple[int, int, int]
    ) -> PointCloud2:
        """Create point cloud from depth map."""
        height, width = image_shape[:2]

        # Camera intrinsics (these should be from camera calibration)
        fx = fy = min(width, height)  # Rough approximation
        cx, cy = width // 2, height // 2

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D points
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter valid depths
        valid_mask = (z > self.min_depth) & (z < self.max_depth)
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)

        # Create PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.is_dense = False
        pointcloud_msg.is_bigendian = False

        # Define fields
        pointcloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pointcloud_msg.point_step = 12
        pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width

        # Pack point data
        pointcloud_msg.data = points.astype(np.float32).tobytes()

        return pointcloud_msg

    def publish_stats(self):
        """Publish performance statistics."""
        try:
            stats = self.depth_estimator.get_stats()

            msg = Float32MultiArray()
            msg.data = [
                float(stats["frames"]),
                stats["avg_fps"],
                stats["avg_time_ms"],
                self.last_process_time * 1000,  # Last processing time in ms
            ]

            self.stats_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing stats: {e}")


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = DepthEstimationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
