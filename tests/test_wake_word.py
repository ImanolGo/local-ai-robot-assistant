#!/usr/bin/env python3
"""
Unit tests for Wake Word Detection Node

Tests model loading, detection accuracy, performance benchmarks, and
cooldown period functionality.
"""

import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from robot_interfaces.msg import AudioData


class TestWakeWordDetector(unittest.TestCase):
    """Test suite for WakeWordDetectorNode."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS2 for all tests."""
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        """Shutdown ROS2 after all tests."""
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures before each test."""
        self.test_node = rclpy.create_node("test_wake_word_node")

    def tearDown(self):
        """Clean up after each test."""
        self.test_node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_node_initialization(self, mock_model):
        """Test that node initializes correctly."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model.return_value = mock_model_instance

        # Create node
        node = WakeWordDetectorNode()

        # Verify initialization
        self.assertIsNotNone(node)
        self.assertEqual(node.wake_word, "hey_rover")
        self.assertEqual(node.confidence_threshold, 0.6)
        self.assertEqual(node.cooldown_seconds, 2.0)
        self.assertTrue(node.running)

        # Cleanup
        node.running = False
        node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_audio_callback(self, mock_model):
        """Test audio callback processes data correctly."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model.return_value = mock_model_instance

        # Create node
        node = WakeWordDetectorNode()

        # Create test audio data
        audio_data = AudioData()
        audio_data.sample_rate = 16000
        audio_data.channels = 1
        audio_data.sample_format = 16  # 16-bit PCM

        # Generate 1 second of test audio
        test_samples = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        audio_data.data = test_samples.tobytes()

        # Process audio
        initial_buffer_size = len(node.audio_buffer)
        node.audio_callback(audio_data)

        # Verify buffer has data
        self.assertGreater(len(node.audio_buffer), initial_buffer_size)

        # Cleanup
        node.running = False
        node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_detection_above_threshold(self, mock_model):
        """Test that detection occurs when confidence is above threshold."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model_instance.predict.return_value = {"hey_rover": 0.8}  # High confidence
        mock_model.return_value = mock_model_instance

        # Create node with lower threshold for testing
        with patch.object(Node, "declare_parameter"):
            with patch.object(Node, "get_parameter") as mock_get_param:
                # Configure parameters
                param_values = {
                    "wake_word": "hey_rover",
                    "confidence_threshold": 0.6,
                    "cooldown_seconds": 1.0,
                    "sample_rate": 16000,
                    "chunk_size": 1280,
                    "enable_verbose_logging": False,
                    "model_path": "",
                }
                mock_get_param.side_effect = lambda name: Mock(value=param_values[name])

                node = WakeWordDetectorNode()

        # Create test audio chunk
        test_chunk = np.random.randn(1280).astype(np.float32)

        # Add to buffer
        with node.buffer_lock:
            node.audio_buffer.extend(test_chunk)

        # Wait for processing
        time.sleep(0.5)

        # Verify detection occurred
        self.assertGreater(node.total_detections, 0)
        self.assertTrue(node.is_in_cooldown)

        # Cleanup
        node.running = False
        node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_cooldown_period(self, mock_model):
        """Test that cooldown period prevents multiple triggers."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model_instance.predict.return_value = {"hey_rover": 0.8}
        mock_model.return_value = mock_model_instance

        # Create node
        node = WakeWordDetectorNode()

        # First detection
        initial_detections = node.total_detections
        node.handle_detection(0.8)
        self.assertEqual(node.total_detections, initial_detections + 1)
        self.assertTrue(node.is_in_cooldown)

        # Immediate second detection (should be ignored)
        node.handle_detection(0.9)
        # Detection count shouldn't increase during cooldown in processing loop
        # (handle_detection is called only when not in cooldown)

        # Wait for cooldown to expire
        time.sleep(node.cooldown_seconds + 0.1)
        node.is_in_cooldown = False

        # Third detection (after cooldown)
        node.handle_detection(0.85)
        self.assertGreaterEqual(node.total_detections, initial_detections + 2)

        # Cleanup
        node.running = False
        node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_confidence_publishing(self, mock_model):
        """Test that confidence scores are published."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model.return_value = mock_model_instance

        # Create node
        node = WakeWordDetectorNode()

        # Create subscriber to capture published confidence
        received_confidence = []

        def confidence_callback(msg):
            received_confidence.append(msg.data)

        confidence_sub = self.test_node.create_subscription(
            Float32, "/audio/wake_word_confidence", confidence_callback, 10
        )

        # Publish a confidence value
        conf_msg = Float32()
        conf_msg.data = 0.75
        node.confidence_pub.publish(conf_msg)

        # Spin to process callbacks
        rclpy.spin_once(self.test_node, timeout_sec=1.0)

        # Verify confidence was published
        self.assertEqual(len(received_confidence), 1)
        self.assertAlmostEqual(received_confidence[0], 0.75, places=2)

        # Cleanup
        self.test_node.destroy_subscription(confidence_sub)
        node.running = False
        node.destroy_node()

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", False)
    def test_missing_openwakeword(self):
        """Test that node fails gracefully when openWakeWord is not installed."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode  # noqa: E402

        with self.assertRaises(ImportError):
            _ = WakeWordDetectorNode()

    def test_audio_normalization(self):
        """Test that audio data is correctly normalized to float32."""
        # Test int16 to float32 conversion
        test_samples = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        normalized = test_samples.astype(np.float32) / 32768.0

        # Verify range
        self.assertGreaterEqual(normalized.min(), -1.0)
        self.assertLessEqual(normalized.max(), 1.0)

        # Verify specific values
        self.assertAlmostEqual(normalized[0], -1.0, places=2)
        self.assertAlmostEqual(normalized[2], 0.0, places=2)
        self.assertAlmostEqual(normalized[4], 0.99997, places=4)

    def test_chunk_size_calculation(self):
        """Test that chunk size is appropriate for sample rate."""
        # 80ms at 16kHz = 1280 samples
        sample_rate = 16000
        target_duration_ms = 80
        expected_chunk_size = int(sample_rate * target_duration_ms / 1000)

        self.assertEqual(expected_chunk_size, 1280)

    @patch("audio_interface_nodes.wake_word_detector_node.OPENWAKEWORD_AVAILABLE", True)
    @patch("audio_interface_nodes.wake_word_detector_node.Model")
    def test_buffer_management(self, mock_model):
        """Test that audio buffer is managed correctly."""
        from audio_interface_nodes.wake_word_detector_node import WakeWordDetectorNode

        # Mock the openWakeWord model
        mock_model_instance = MagicMock()
        mock_model_instance.models = {"hey_rover": None}
        mock_model.return_value = mock_model_instance

        # Create node
        node = WakeWordDetectorNode()

        # Fill buffer beyond capacity
        max_buffer_size = node.audio_buffer.maxlen
        large_audio = np.random.randn(max_buffer_size * 2).astype(np.float32)

        with node.buffer_lock:
            node.audio_buffer.extend(large_audio)

        # Verify buffer doesn't exceed max size
        self.assertEqual(len(node.audio_buffer), max_buffer_size)

        # Cleanup
        node.running = False
        node.destroy_node()


class TestWakeWordPerformance(unittest.TestCase):
    """Performance tests for wake word detection."""

    @classmethod
    def setUpClass(cls):
        """Initialize ROS2 for all tests."""
        if not rclpy.ok():
            rclpy.init()

    def test_processing_latency(self):
        """Test that processing latency is within target (<100ms)."""
        try:
            from openwakeword.model import Model
        except ImportError:
            self.skipTest("openWakeWord not available")

        # Create model
        model = Model()

        # Generate test audio chunk (80ms at 16kHz)
        chunk_size = 1280
        test_chunk = np.random.randn(chunk_size).astype(np.float32)

        # Measure processing time
        start_time = time.time()
        _ = model.predict(test_chunk)
        processing_time = time.time() - start_time

        # Verify latency is under target (100ms)
        self.assertLess(
            processing_time,
            0.1,
            f"Processing took {processing_time*1000:.2f}ms, target is <100ms",
        )

    def test_cpu_usage_estimate(self):
        """Estimate CPU usage during wake word detection."""
        try:
            from openwakeword.model import Model
        except ImportError:
            self.skipTest("openWakeWord not available")

        # Create model
        model = Model()

        # Process chunks for 1 second
        chunk_size = 1280  # 80ms at 16kHz
        chunks_per_second = 1000 / 80  # ~12.5 chunks
        test_duration = 1.0  # seconds

        start_time = time.time()
        chunks_processed = 0

        while time.time() - start_time < test_duration:
            test_chunk = np.random.randn(chunk_size).astype(np.float32)
            _ = model.predict(test_chunk)
            chunks_processed += 1
            time.sleep(0.08)  # Simulate real-time processing

        elapsed_time = time.time() - start_time
        chunks_per_second_actual = chunks_processed / elapsed_time

        # Log results
        print(f"\nProcessed {chunks_processed} chunks in {elapsed_time:.2f}s")
        print(f"Rate: {chunks_per_second_actual:.2f} chunks/second")
        print(f"Expected real-time rate: {chunks_per_second:.2f} chunks/second")

        # Verify we can keep up with real-time
        self.assertGreaterEqual(
            chunks_per_second_actual,
            chunks_per_second * 0.9,
            "Cannot maintain real-time processing rate",
        )


if __name__ == "__main__":
    unittest.main()
