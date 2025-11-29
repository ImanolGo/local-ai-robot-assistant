#!/usr/bin/env python3
"""
YOLO Object Detection Node - TensorRT Optimized
Uses YOLOv11n with TensorRT via Ultralytics for real-time object detection on Jetson Orin Nano.

Performance Targets:
- FPS: 20+
- Latency: <50ms per frame
- GPU Memory: <400MB
"""

import time
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class YOLODetector:
    """Ultralytics YOLO detector with TensorRT support."""

    def __init__(
        self,
        engine_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize YOLO detector with TensorRT engine using Ultralytics.

        Args:
            engine_path: Path to TensorRT engine file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Load model using Ultralytics (handles TensorRT engines properly)
        self.model = YOLO(engine_path)

        # Load COCO class names
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> List[str]:
        """Load COCO class names."""
        # COCO 80 classes
        return [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Run detection using Ultralytics YOLO (handles preprocessing/inference/postprocessing).

        Args:
            image: BGR image from OpenCV

        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # Run inference with Ultralytics (handles everything internally)
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        boxes = []
        scores = []
        class_ids = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Batch transfer from GPU to CPU (optimized)
                bboxes_np = result.boxes.xyxy.cpu().numpy()  # Shape: (N, 4)
                confs_np = result.boxes.conf.cpu().numpy()  # Shape: (N,)
                classes_np = result.boxes.cls.cpu().numpy().astype(int)  # Shape: (N,)

                # Convert to lists
                for i in range(len(bboxes_np)):
                    boxes.append(bboxes_np[i].tolist())
                    scores.append(float(confs_np[i]))
                    class_ids.append(int(classes_np[i]))

        return boxes, scores, class_ids


class ObjectDetectorNode(Node):
    """ROS2 node for YOLO object detection."""

    def __init__(self):
        super().__init__("object_detector")

        # Declare parameters
        self.declare_parameter(
            "engine_path",
            "/home/imanolgo/repos/local-ai-robot-assistant/models/yolo_trt/yolo11n_fp16.engine",
        )
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.4)
        self.declare_parameter("input_size", [640, 640])
        self.declare_parameter("visualize", True)
        self.declare_parameter("publish_rate", 20.0)  # Hz

        # Get parameters
        engine_path = self.get_parameter("engine_path").value
        confidence_threshold = self.get_parameter("confidence_threshold").value
        nms_threshold = self.get_parameter("nms_threshold").value
        input_size = tuple(self.get_parameter("input_size").value)
        self.visualize = self.get_parameter("visualize").value

        # Initialize detector
        self.get_logger().info(f"Loading YOLO model from {engine_path}")
        self.detector = YOLODetector(engine_path, confidence_threshold, nms_threshold, input_size)
        self.get_logger().info("YOLO model loaded successfully")

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/undistorted", self.image_callback, 10
        )

        # Create publishers
        self.detection_pub = self.create_publisher(Detection2DArray, "/perception/objects", 10)

        if self.visualize:
            self.viz_pub = self.create_publisher(Image, "/perception/objects_viz", 10)

        # Performance monitoring
        self.frame_count = 0
        self.total_time = 0.0
        self.fps_timer = self.create_timer(5.0, self.log_performance)

        self.get_logger().info("Object detector node initialized")

    def image_callback(self, msg: Image):
        """Process incoming images and detect objects."""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Run detection
            start_time = time.time()
            boxes, scores, class_ids = self.detector.detect(cv_image)
            inference_time = time.time() - start_time

            # Update performance metrics
            self.frame_count += 1
            self.total_time += inference_time

            # Create detection message
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            for box, score, class_id in zip(boxes, scores, class_ids):
                detection = Detection2D()
                detection.header = msg.header

                # Set bounding box
                detection.bbox.center.position.x = float((box[0] + box[2]) / 2)
                detection.bbox.center.position.y = float((box[1] + box[3]) / 2)
                detection.bbox.size_x = float(box[2] - box[0])
                detection.bbox.size_y = float(box[3] - box[1])

                # Set class and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id)
                hypothesis.hypothesis.score = score
                detection.results.append(hypothesis)

                detection_array.detections.append(detection)

            # Publish detections
            self.detection_pub.publish(detection_array)

            # Publish visualization if enabled
            if self.visualize:
                viz_image = self.draw_detections(cv_image, boxes, scores, class_ids)
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding="bgr8")
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def draw_detections(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        scores: List[float],
        class_ids: List[int],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        viz_image = image.copy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # Draw box
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{self.detector.class_names[class_id]}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(viz_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                viz_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return viz_image

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
    node = ObjectDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
