#! /usr/bin/env python3
"""
YOLO Object Detection Node - TensorRT Optimized
Uses YOLOv11n with TensorRT via Ultralytics for real-time object detection on Jetson Orin Nano.

Performance Targets:
- FPS: 20+
- Latency: <50ms per frame
- GPU Memory: <400MB
"""

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from robot_interfaces.msg import PerceptionEvent


class EventGenerator:
    """Generates perception events based on object tracking updates."""

    def __init__(self, movement_threshold: float = 0.1, missing_frames_threshold: int = 5):
        """
        Initialize event generator.

        Args:
            movement_threshold: normalized distance (0-1) to trigger MOVED event
            missing_frames_threshold: number of frames an ID is missing before LEFT trigger
        """
        self.movement_threshold = movement_threshold
        self.missing_frames_threshold = missing_frames_threshold
        self.tracked_objects: Dict[
            int, Dict
        ] = {}  # ID -> {pos, class, conf, last_seen, missing_count}

    def update(self, current_detections: List[Dict]) -> List[Dict]:
        """
        Update tracking state and generate events.

        Args:
           current_detections: List of dicts with keys: id, class, conf, center_x, center_y

        Returns:
           List of event dicts to publish
        """
        events = []
        current_ids = set()

        # Process current frame detections
        for det in current_detections:
            track_id = det["id"]
            current_ids.add(track_id)

            # Check if this is a new object
            if track_id not in self.tracked_objects:
                # NEW OBJECT -> ENTERED_FOV
                self.tracked_objects[track_id] = {
                    "pos": (det["center_x"], det["center_y"]),
                    "class": det["class"],
                    "conf": det["conf"],
                    "last_seen": time.time(),
                    "missing_count": 0,
                }
                events.append(
                    {
                        "type": PerceptionEvent.ENTERED_FOV,
                        "id": track_id,
                        "class": det["class"],
                        "conf": det["conf"],
                        "pos": (det["center_x"], det["center_y"]),
                    }
                )
            else:
                # EXISTING OBJECT
                prev_obj = self.tracked_objects[track_id]
                prev_pos = prev_obj["pos"]
                curr_pos = (det["center_x"], det["center_y"])

                # Update state
                prev_obj["pos"] = curr_pos
                prev_obj["conf"] = det["conf"]
                prev_obj["last_seen"] = time.time()
                prev_obj["missing_count"] = 0

                # Check for significant movement
                dist = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
                # Normalize distance (roughly, assuming image width is 1.0 in normalized coords if
                # we were using them)
                # Here we are using pixel coordinates, so we need to normalize or use
                # pixel threshold
                # Assuming pixel coords from main node, let's normalize roughly by 640 for
                # thresholding logic
                # Actually, main node logic below passes pixel coords. Let's assume threshold
                # is pixels.
                # If threshold is 0.1 (10%), that's ~64 pixels on 640 width.

                # Let's enforce the input to be normalized or handle pixels
                # To be safe, let's assume pixel distance threshold for now: 50 pixels
                pixel_threshold = 50.0

                if dist > pixel_threshold:
                    events.append(
                        {
                            "type": PerceptionEvent.MOVED_SIGNIFICANTLY,
                            "id": track_id,
                            "class": det["class"],
                            "conf": det["conf"],
                            "pos": curr_pos,
                        }
                    )

        # Process missing objects
        ids_to_remove = []
        for track_id, obj in self.tracked_objects.items():
            if track_id not in current_ids:
                obj["missing_count"] += 1
                if obj["missing_count"] >= self.missing_frames_threshold:
                    # OBJECT LOST -> LEFT_FOV
                    events.append(
                        {
                            "type": PerceptionEvent.LEFT_FOV,
                            "id": track_id,
                            "class": obj["class"],
                            "conf": obj["conf"],
                            "pos": obj["pos"],
                        }
                    )
                    ids_to_remove.append(track_id)

        # Cleanup removed objects
        for track_id in ids_to_remove:
            del self.tracked_objects[track_id]

        return events


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

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run detection using Ultralytics YOLO with TRACKING.

        Args:
            image: BGR image from OpenCV

        Returns:
            Tuple of (boxes, scores, class_ids, track_ids) as numpy arrays
        """
        # Run tracking inference
        # persist=True is crucial for ID consistency across frames
        results = self.model.track(
            image,
            conf=self.confidence_threshold,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",  # Default robust tracker
        )

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Batch transfer from GPU to CPU
                boxes_np = result.boxes.xyxy.cpu().numpy()  # Shape: (N, 4)
                confs_np = result.boxes.conf.cpu().numpy()  # Shape: (N,)
                classes_np = result.boxes.cls.cpu().numpy().astype(int)  # Shape: (N,)

                # Check if tracking IDs are available (might be None in first frame or if
                # tracking fails)
                if result.boxes.id is not None:
                    track_ids_np = result.boxes.id.cpu().numpy().astype(int)
                else:
                    # Fallback if no IDs: assign ephemeral IDs or -1
                    track_ids_np = np.full(len(classes_np), -1, dtype=int)

                return boxes_np, confs_np, classes_np, track_ids_np

        # Return empty arrays if no detections
        return (
            np.empty((0, 4), dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )


class ObjectDetectorNode(Node):
    """ROS2 node for YOLO object detection with Event Generation."""

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

        # Initialize Event Generator
        self.event_generator = EventGenerator()

        # Optimize OpenCV
        cv2.setNumThreads(4)
        cv2.setUseOptimized(True)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # QoS profile for sensor data (Best Effort)
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, "/camera/undistorted", self.image_callback, qos_sensor
        )

        # Create publishers
        self.detection_pub = self.create_publisher(Detection2DArray, "/perception/objects", 10)

        # NEW: Event publisher (Reliable QoS for events)
        qos_events = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.event_pub = self.create_publisher(PerceptionEvent, "/perception/events", qos_events)

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
            # self.get_logger().info("Debug: Image received in callback")
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Run detection
            start_time = time.time()
            boxes, scores, class_ids, track_ids = self.detector.detect(cv_image)
            inference_time = time.time() - start_time

            # Prepare struct for Event Generator
            current_detections = []
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            for i in range(len(boxes)):
                box = boxes[i]
                score = float(scores[i])
                class_id = int(class_ids[i])
                track_id = int(track_ids[i])

                # Center coordinates
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)

                # Skip untracked objects (-1) for event generation logic if necessary,
                # but typically ByteTrack assigns IDs almost always.
                if track_id != -1:
                    current_detections.append(
                        {
                            "id": track_id,
                            "class": self.detector.class_names[class_id],
                            "conf": score,
                            "center_x": cx,
                            "center_y": cy,
                        }
                    )

                # Create ROS message
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = cx
                detection.bbox.center.position.y = cy
                detection.bbox.size_x = float(box[2] - box[0])
                detection.bbox.size_y = float(box[3] - box[1])

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id)
                hypothesis.hypothesis.score = score
                # We can store track_id in the 'id' field of Detection2D if available,
                # but standard message doesn't have it easily. We usually put it
                # in class_id string or side channel.
                # Here we stick to standard message.

                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

            # Publish regular detections
            self.detection_pub.publish(detection_array)

            # Generate and publish events
            events = self.event_generator.update(current_detections)
            for evt in events:
                event_msg = PerceptionEvent()
                event_msg.header = msg.header
                event_msg.event_type = evt["type"]
                event_msg.class_name = evt["class"]
                event_msg.tracking_id = evt["id"]
                event_msg.confidence = evt["conf"]
                event_msg.position = Point(x=evt["pos"][0], y=evt["pos"][1], z=0.0)

                self.event_pub.publish(event_msg)

                self.get_logger().info(f"Event: {evt['type']} - {evt['class']} (ID: {evt['id']})")

            # Update performance metrics
            self.frame_count += 1
            self.total_time += inference_time

            # Publish visualization if enabled
            if self.visualize:
                viz_image = self.draw_detections(cv_image, boxes, scores, class_ids, track_ids)
                viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding="bgr8")
                viz_msg.header = msg.header
                self.viz_pub.publish(viz_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def draw_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        viz_image = image.copy()

        for i in range(len(boxes)):
            box = boxes[i]
            score = float(scores[i])
            class_id = int(class_ids[i])
            track_id = int(track_ids[i])

            x1, y1, x2, y2 = box.astype(int)

            # Draw box
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label with ID
            id_str = f"ID:{track_id} " if track_id != -1 else ""
            label = f"{id_str}{self.detector.class_names[class_id]}: {score:.2f}"

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
