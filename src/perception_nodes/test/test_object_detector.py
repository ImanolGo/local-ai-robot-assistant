#!/usr/bin/env python3
"""
Unit tests for Object Detector Node.

Tests:
- TensorRT engine loading
- Image preprocessing
- Inference on sample images
- Detection postprocessing
- Bounding box format and coordinates
- Performance benchmarking
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray


class TestYOLODetector(unittest.TestCase):
    """Test cases for YOLO detector inference wrapper."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test environment."""
        self.test_image = self._create_test_image()

    def _create_test_image(self) -> np.ndarray:
        """Create a test BGR image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a rectangle (simulated object)
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
        return img

    @patch("perception_nodes.object_detector.YOLO")
    def test_detector_initialization(self, mock_yolo_class):
        """Test YOLODetector initialization with Ultralytics."""
        from perception_nodes.object_detector import YOLODetector

        # Mock Ultralytics YOLO model
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model

        # Initialize detector
        detector = YOLODetector(
            engine_path="fake_engine.engine",
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )

        self.assertIsNotNone(detector)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.nms_threshold, 0.4)
        mock_yolo_class.assert_called_once_with("fake_engine.engine")

    @patch("perception_nodes.object_detector.YOLO")
    def test_detection_with_ultralytics(self, mock_yolo_class):
        """Test detection using Ultralytics (preprocessing/inference handled internally)."""
        from perception_nodes.object_detector import YOLODetector

        # Mock Ultralytics model and results
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model

        # Mock detection results with proper __len__ support
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=2)  # Fix len() error
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [[100, 100, 200, 200], [300, 300, 400, 400]]
        )
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 2])

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        mock_model.return_value = [mock_result]

        # Initialize detector
        detector = YOLODetector(
            engine_path="fake_engine.engine",
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )

        # Run detection
        boxes, scores, class_ids = detector.detect(self.test_image)

        # Verify results
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(class_ids), 2)
        self.assertEqual(scores[0], 0.9)
        self.assertEqual(scores[1], 0.8)
        self.assertEqual(class_ids[0], 0)
        self.assertEqual(class_ids[1], 2)

        # Verify model was called with correct parameters
        mock_model.assert_called_once_with(self.test_image, conf=0.5, verbose=False)

    @patch("perception_nodes.object_detector.YOLO")
    def test_class_names_loading(self, mock_yolo_class):
        """Test COCO class names are loaded correctly."""
        from perception_nodes.object_detector import YOLODetector

        mock_model = Mock()
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(
            engine_path="fake_engine.engine",
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )
        class_names = detector._load_class_names()

        # COCO has 80 classes
        self.assertEqual(len(class_names), 80)

        # Check some known classes
        self.assertIn("person", class_names)
        self.assertIn("car", class_names)
        self.assertIn("dog", class_names)


class TestObjectDetectorNode(unittest.TestCase):
    """Test cases for ROS2 Object Detector node."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if rclpy.ok():
            rclpy.shutdown()

    @patch("perception_nodes.object_detector.YOLODetector")
    def test_node_initialization(self, mock_detector_class):
        """Test node initialization."""
        from perception_nodes.object_detector import ObjectDetectorNode

        # Mock detector
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Create node
        node = ObjectDetectorNode()

        # Check node was created
        self.assertIsNotNone(node)
        self.assertEqual(node.get_name(), "object_detector")

        # Clean up
        node.destroy_node()

    @patch("perception_nodes.object_detector.YOLODetector")
    def test_image_callback(self, mock_detector_class):
        """Test image callback processes messages correctly."""
        from perception_nodes.object_detector import ObjectDetectorNode

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (
            [[100, 100, 200, 200]],  # boxes
            [0.95],  # scores
            [0],  # class_ids (person)
        )
        mock_detector.class_names = ["person", "car", "dog"]
        mock_detector_class.return_value = mock_detector

        # Create node
        node = ObjectDetectorNode()

        # Create test image message
        bridge = CvBridge()
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_msg = bridge.cv2_to_imgmsg(test_img, encoding="bgr8")

        # Set up subscriber to capture published detections
        received_detections = []

        def detection_callback(msg):
            received_detections.append(msg)

        test_sub = node.create_subscription(
            Detection2DArray, "/perception/objects", detection_callback, 10
        )

        # Process image
        node.image_callback(img_msg)

        # Spin to process callbacks
        rclpy.spin_once(node, timeout_sec=1.0)

        # Check detector was called
        mock_detector.detect.assert_called_once()

        # Clean up
        node.destroy_subscription(test_sub)
        node.destroy_node()

    @patch("perception_nodes.object_detector.YOLODetector")
    def test_visualization(self, mock_detector_class):
        """Test visualization output."""
        from perception_nodes.object_detector import ObjectDetectorNode

        # Mock detector
        mock_detector = MagicMock()
        mock_detector.class_names = ["person", "car", "dog"]
        mock_detector_class.return_value = mock_detector

        # Create node with visualization enabled
        node = ObjectDetectorNode()
        node.visualize = True

        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = [[100, 100, 200, 200]]
        scores = [0.95]
        class_ids = [0]

        # Draw detections
        viz_img = node.draw_detections(test_img, boxes, scores, class_ids)

        # Check image was modified (should have drawn boxes)
        self.assertEqual(viz_img.shape, test_img.shape)
        self.assertFalse(np.array_equal(viz_img, test_img))

        # Clean up
        node.destroy_node()


class TestPerformance(unittest.TestCase):
    """Performance benchmarking tests."""

    @patch("perception_nodes.object_detector.YOLO")
    def test_detection_performance(self, mock_yolo_class):
        """Benchmark end-to-end detection performance (Ultralytics handles preprocessing
        internally)."""
        import time

        from perception_nodes.object_detector import YOLODetector

        # Mock model that returns quickly
        mock_model = Mock()
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=0)
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        detector = YOLODetector(
            engine_path="fake_engine.engine",
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )

        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Benchmark detection (including internal preprocessing)
        num_iterations = 100
        start = time.time()

        for _ in range(num_iterations):
            detector.detect(test_img)

        elapsed = time.time() - start
        avg_time = (elapsed / num_iterations) * 1000  # ms

        # Detection with mocked model should be very fast (<1ms)
        self.assertLess(avg_time, 5.0, f"Mocked detection too slow: {avg_time:.2f}ms")


if __name__ == "__main__":
    unittest.main()
