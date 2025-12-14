import sys
import unittest

# Adjust path to find the module
sys.path.append(
    "/home/imanolgo/repos/local-ai-robot-assistant/src/perception_nodes/perception_nodes"
)

# Mock ROS and msg dependencies before importing object_detector
from unittest.mock import MagicMock  # noqa: E402

# Create hierarchical mock for rclpy
mock_rclpy = MagicMock()
mock_rclpy.node = MagicMock()
mock_rclpy.qos = MagicMock()  # This needs to be explicitly set
sys.modules["rclpy"] = mock_rclpy
sys.modules["rclpy.node"] = mock_rclpy.node
sys.modules["rclpy.qos"] = mock_rclpy.qos

sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["vision_msgs.msg"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["cv_bridge"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()

# Mock the PerceptionEvent message specifically
mock_perception_msg = MagicMock()
mock_perception_msg.PerceptionEvent.ENTERED_FOV = 1
mock_perception_msg.PerceptionEvent.LEFT_FOV = 2
mock_perception_msg.PerceptionEvent.MOVED_SIGNIFICANTLY = 3
sys.modules["robot_interfaces.msg"] = mock_perception_msg

from object_detector import EventGenerator  # noqa: E402


class TestEventGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = EventGenerator(movement_threshold=10.0, missing_frames_threshold=2)

    def test_entered_fov(self):
        # detection: id, class, conf, center_x, center_y
        detections = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 100, "center_y": 100}]

        events = self.generator.update(detections)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], 1)  # ENTERED_FOV
        self.assertEqual(events[0]["id"], 1)

    def test_stationary_update(self):
        # Initial frame
        detections = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 100, "center_y": 100}]
        self.generator.update(detections)

        # Second frame - slight movement below threshold
        detections_2 = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 101, "center_y": 101}]
        events = self.generator.update(detections_2)

        self.assertEqual(len(events), 0)  # No events

    def test_moved_significantly(self):
        # Initial frame
        detections = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 100, "center_y": 100}]
        self.generator.update(detections)

        # Second frame - big movement
        # Dist = 60 pixels > threshold (50 in default, but set 10 in setup)
        detections_2 = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 160, "center_y": 100}]
        events = self.generator.update(detections_2)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], 3)  # MOVED_SIGNIFICANTLY

    def test_left_fov(self):
        # Initial frame
        detections = [{"id": 1, "class": "person", "conf": 0.9, "center_x": 100, "center_y": 100}]
        self.generator.update(detections)

        # Frame 1 missing
        events = self.generator.update([])
        self.assertEqual(len(events), 0)

        # Frame 2 missing (threshold is 2)
        events = self.generator.update([])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], 2)  # LEFT_FOV
        self.assertEqual(events[0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
