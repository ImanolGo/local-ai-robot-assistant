#!/usr/bin/env python3
"""
DeepStream-Accelerated Camera Driver for IMX219 Camera
Hardware-accelerated camera capture with NVMM buffers and zero-copy operations.

Author: Local AI Robot Team
License: Apache-2.0
"""

import threading
import time
from typing import Any, Dict, Optional

import gi
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

# GObject introspection version must be set before importing Gst
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

# Initialize GStreamer
Gst.init(None)


class CameraDriver(Node):
    """
    DeepStream-accelerated camera driver for IMX219 camera.

    Features:
    - Hardware-accelerated capture with nvarguscamerasrc
    - NVMM memory buffers for zero-copy operations
    - Real-time frame rate control
    - Camera info publisher with calibration data
    - GPU memory optimization
    - Performance monitoring
    """

    def __init__(self):
        super().__init__("camera_driver")

        # Initialize variables
        self.pipeline: Optional[Gst.Pipeline] = None
        self.appsink: Optional[Gst.Element] = None
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None

        # Performance monitoring
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.processing_times = []

        # Threading
        self.pipeline_thread: Optional[threading.Thread] = None
        self.running = False

        # Load configuration
        self._load_config()

        # Set up ROS2 publishers and parameters
        self._setup_ros2()

        # Load camera calibration
        self._load_calibration()

        # Initialize DeepStream pipeline
        self._setup_deepstream_pipeline()

        # Start the pipeline
        self._start_pipeline()

        self.get_logger().info("Camera driver initialized with DeepStream acceleration")

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
        """Get default camera configuration."""
        return {
            "camera": {
                "device_id": 0,
                "sensor_mode": 0,
                "width": 1640,
                "height": 1232,
                "framerate": 30,
                "flip_method": 0,
            },
            "deepstream": {
                "source_element": "nvarguscamerasrc",
                "nvmm_memory": True,
                "format": "NV12",
                "buffer_pool_size": 4,
                "max_buffers": 8,
                "do_timestamp": True,
            },
            "ros2": {
                "raw_image_topic": "/camera/raw",
                "camera_info_topic": "/camera/camera_info",
                "frame_id": "camera_link",
                "publish_camera_info": True,
            },
            "monitoring": {
                "enable_fps_monitoring": True,
                "enable_gpu_monitoring": True,
                "log_performance_stats": True,
                "stats_publish_rate": 1.0,
            },
        }

    def _setup_ros2(self) -> None:
        """Set up ROS2 publishers and parameters."""
        # Declare parameters
        self.declare_parameter("device_id", self.config["camera"]["device_id"])
        self.declare_parameter("width", self.config["camera"]["width"])
        self.declare_parameter("height", self.config["camera"]["height"])
        self.declare_parameter("framerate", self.config["camera"]["framerate"])
        self.declare_parameter("flip_method", self.config["camera"]["flip_method"])

        # QoS profile for real-time performance
        _ = self.config.get("ros2", {}).get("qos_profile", {})
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        raw_topic = self.config["ros2"]["raw_image_topic"]
        info_topic = self.config["ros2"]["camera_info_topic"]

        self.image_pub = self.create_publisher(Image, raw_topic, qos)

        if self.config["ros2"]["publish_camera_info"]:
            self.camera_info_pub = self.create_publisher(CameraInfo, info_topic, qos)

        # Performance monitoring timer
        if self.config["monitoring"]["enable_fps_monitoring"]:
            stats_rate = self.config["monitoring"]["stats_publish_rate"]
            self.stats_timer = self.create_timer(1.0 / stats_rate, self._publish_stats)

    def _load_calibration(self) -> None:
        """Load camera calibration data."""
        try:
            calib_path = self.config.get(
                "calibration_file",
                "/home/imanolgo/repos/local-ai-robot-assistant/config/camera_calibration.yaml",
            )

            with open(calib_path, "r") as file:
                calib_data = yaml.safe_load(file)

            # Create CameraInfo message
            self.camera_info = CameraInfo()
            self.camera_info.header.frame_id = self.config["ros2"]["frame_id"]
            self.camera_info.width = calib_data["image_width"]
            self.camera_info.height = calib_data["image_height"]

            # Camera matrix (K)
            K = np.array(calib_data["camera_matrix"]).flatten()
            self.camera_info.k = K.tolist()

            # Distortion coefficients (D)
            D = np.array(calib_data["distortion_coefficients"]).flatten()
            self.camera_info.d = D.tolist()
            self.camera_info.distortion_model = "plumb_bob"

            # Projection matrix (P) - same as K for monocular camera with no rectification
            P = np.zeros((3, 4))
            P[:3, :3] = np.array(calib_data["camera_matrix"])
            self.camera_info.p = P.flatten().tolist()

            # Rectification matrix (R) - identity for monocular camera
            self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

            self.get_logger().info(f"Loaded camera calibration from {calib_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load camera calibration: {e}")
            self.camera_info = None

    def _setup_deepstream_pipeline(self) -> None:
        """Set up the DeepStream GStreamer pipeline."""
        try:
            # Get camera parameters
            cam_config = self.config["camera"]
            ds_config = self.config["deepstream"]

            # Get current parameter values
            device_id = self.get_parameter("device_id").value
            width = self.get_parameter("width").value
            height = self.get_parameter("height").value
            framerate = self.get_parameter("framerate").value
            flip_method = self.get_parameter("flip_method").value

            # Create pipeline string for nvarguscamerasrc
            pipeline_str = (
                f"{ds_config['source_element']} "
                f"sensor-id={device_id} "
                f"sensor-mode={cam_config['sensor_mode']} "
                f"do-timestamp={str(ds_config['do_timestamp']).lower()} "
                f"! "
                f"video/x-raw(memory:NVMM), "
                f"width={width}, "
                f"height={height}, "
                f"framerate={framerate}/1, "
                f"format={ds_config['format']} "
                f"! "
                f"nvvidconv flip-method={flip_method} "
                f"! "
                f"video/x-raw, "
                f"width={width}, "
                f"height={height}, "
                f"format=BGRx "
                f"! "
                f"videoconvert "
                f"! "
                f"video/x-raw, format=BGR "
                f"! "
                f"appsink name=appsink emit-signals=true "
                f"max-buffers={ds_config['max_buffers']} "
                f"drop=true sync=false"
            )

            self.get_logger().info(f"Creating DeepStream pipeline: {pipeline_str}")

            # Create pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)

            if not self.pipeline:
                raise RuntimeError("Failed to create GStreamer pipeline")

            # Get appsink element
            self.appsink = self.pipeline.get_by_name("appsink")
            if not self.appsink:
                raise RuntimeError("Failed to get appsink element")

            # Connect to new-sample signal
            self.appsink.connect("new-sample", self._on_new_sample)

            self.get_logger().info("DeepStream pipeline created successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to setup DeepStream pipeline: {e}")
            raise

    def _start_pipeline(self) -> None:
        """Start the GStreamer pipeline."""
        try:
            if not self.pipeline:
                raise RuntimeError("Pipeline not initialized")

            # Set pipeline to playing state
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")

            self.running = True

            # Start pipeline monitoring thread
            self.pipeline_thread = threading.Thread(target=self._monitor_pipeline)
            self.pipeline_thread.daemon = True
            self.pipeline_thread.start()

            self.get_logger().info("DeepStream pipeline started successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to start pipeline: {e}")
            raise

    def _monitor_pipeline(self) -> None:
        """Monitor the pipeline for errors and EOS."""
        if not self.pipeline:
            return

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()

        while self.running:
            message = bus.timed_pop_filtered(
                Gst.CLOCK_TIME_NONE,
                Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING,
            )

            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    self.get_logger().error(f"Pipeline error: {err}, Debug: {debug}")
                    break
                elif message.type == Gst.MessageType.EOS:
                    self.get_logger().info("End of stream received")
                    break
                elif message.type == Gst.MessageType.WARNING:
                    warn, debug = message.parse_warning()
                    self.get_logger().warning(f"Pipeline warning: {warn}, Debug: {debug}")

        bus.remove_signal_watch()

    def _on_new_sample(self, appsink) -> Gst.FlowReturn:
        """Handle new frame from the pipeline."""
        try:
            start_time = time.time()

            # Get the sample
            sample = appsink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.ERROR

            # Get buffer from sample
            buffer = sample.get_buffer()
            if not buffer:
                return Gst.FlowReturn.ERROR

            # Get caps and extract frame info
            caps = sample.get_caps()
            if not caps:
                return Gst.FlowReturn.ERROR

            structure = caps.get_structure(0)
            width = structure.get_int("width")[1]
            height = structure.get_int("height")[1]

            # Map buffer to get data
            map_info = buffer.map(Gst.MapFlags.READ)
            if not map_info[0]:
                return Gst.FlowReturn.ERROR

            try:
                # Convert buffer data to numpy array
                frame_data = np.frombuffer(map_info[1], dtype=np.uint8)
                frame = frame_data.reshape((height, width, 3))

                # Create ROS2 Image message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = self.config["ros2"]["frame_id"]

                image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                image_msg.header = header

                # Publish image
                self.image_pub.publish(image_msg)

                # Publish camera info if available
                if self.camera_info and self.config["ros2"]["publish_camera_info"]:
                    self.camera_info.header = header
                    self.camera_info_pub.publish(self.camera_info)

                # Update performance metrics
                self._update_performance_metrics(start_time)

            finally:
                buffer.unmap(map_info)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")
            return Gst.FlowReturn.ERROR

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

            self.get_logger().info(
                f"Camera Performance - FPS: {self.fps:.1f}, "
                f"Avg Processing: {avg_processing_time*1000:.1f}ms, "
                f"Max Processing: {max_processing_time*1000:.1f}ms"
            )

    def destroy_node(self) -> None:
        """Clean shutdown of the camera driver."""
        self.get_logger().info("Shutting down camera driver...")

        self.running = False

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Wait for monitoring thread
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=2.0)

        super().destroy_node()
        self.get_logger().info("Camera driver shutdown complete")


def main(args=None):
    """Main entry point for the camera driver node."""
    rclpy.init(args=args)

    try:
        node = CameraDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Camera driver error: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
