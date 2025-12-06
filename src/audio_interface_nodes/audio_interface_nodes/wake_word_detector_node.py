#!/usr/bin/env python3
"""
Wake Word Detection Node for Local AI Robot Assistant

This node continuously monitors audio from the /audio/raw topic and detects
the wake word "Hey Rover" using openWakeWord. It publishes detection events
with confidence scores to /audio/wake_word_detected.

Features:
- Always-on wake word detection
- Configurable confidence threshold
- Cooldown period to prevent multiple triggers
- Low CPU usage (<5% target)
- Real-time processing with minimal latency (<100ms)

Author: Local AI Robot Assistant Team
License: MIT
"""

import os
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32

from robot_interfaces.msg import AudioData, AudioEvent

try:
    from openwakeword.model import Model

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False


class WakeWordDetectorNode(Node):
    """
    ROS2 node for wake word detection using openWakeWord.

    Continuously monitors audio stream and detects "Hey Rover" wake word.
    Publishes detection events with confidence scores.
    """

    def __init__(self):
        super().__init__("wake_word_detector_node")

        # Check if openWakeWord is available
        if not OPENWAKEWORD_AVAILABLE:
            self.get_logger().error(
                "openWakeWord not available. Install with: pip install openwakeword"
            )
            raise ImportError("openWakeWord library not found")

        # Declare parameters
        self.declare_parameter("wake_word", "hey_rover")
        self.declare_parameter("confidence_threshold", 0.6)
        self.declare_parameter("cooldown_seconds", 2.0)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("chunk_size", 1280)  # 80ms at 16kHz
        self.declare_parameter("enable_verbose_logging", False)
        self.declare_parameter("model_path", "")  # Empty = use default models

        # Get parameters
        self.wake_word = self.get_parameter("wake_word").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.cooldown_seconds = self.get_parameter("cooldown_seconds").value
        self.sample_rate = self.get_parameter("sample_rate").value
        self.chunk_size = self.get_parameter("chunk_size").value
        self.verbose_logging = self.get_parameter("enable_verbose_logging").value
        self.model_path = self.get_parameter("model_path").value

        # State variables
        self.last_detection_time = 0.0
        self.is_in_cooldown = False
        self.running = False

        # Audio buffer for processing
        self.audio_buffer = deque(maxlen=self.chunk_size * 10)  # 10 chunks buffer
        self.buffer_lock = threading.Lock()

        # Statistics
        self.total_chunks_processed = 0
        self.total_detections = 0
        self.false_positive_count = 0
        self.messages_received = 0

        # Initialize openWakeWord model
        self.get_logger().info("Initializing openWakeWord model...")
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Detect inference framework from file extension
                if self.model_path.endswith(".onnx"):
                    self.model = Model(
                        wakeword_models=[self.model_path], inference_framework="onnx"
                    )
                    self.get_logger().info(f"Loaded custom ONNX model from: {self.model_path}")
                else:
                    self.model = Model(wakeword_models=[self.model_path])
                    self.get_logger().info(f"Loaded custom model from: {self.model_path}")
            else:
                # Use default models (will download if not present)
                self.model = Model()
                self.get_logger().info("Loaded default openWakeWord models")

            # Log available models
            available_models = list(self.model.models.keys())
            self.get_logger().info(f"Available wake words: {available_models}")

            # Verify wake word is available
            if self.wake_word not in available_models:
                self.get_logger().warn(
                    f"Wake word '{self.wake_word}' not in available models. "
                    f"Using first available: {available_models[0]}"
                )
                self.wake_word = available_models[0]

        except Exception as e:
            self.get_logger().error(f"Failed to initialize openWakeWord model: {e}")
            raise

        # Create QoS profile for audio subscription
        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Create subscriber
        self.audio_sub = self.create_subscription(
            AudioData, "/audio/raw", self.audio_callback, audio_qos
        )

        # Create publishers
        self.wake_word_pub = self.create_publisher(Bool, "/audio/wake_word_detected", 10)

        self.confidence_pub = self.create_publisher(Float32, "/audio/wake_word_confidence", 10)

        self.event_pub = self.create_publisher(AudioEvent, "/audio/events", 10)

        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Create status timer
        self.status_timer = self.create_timer(10.0, self.publish_status)

        self.get_logger().info(
            f"Wake word detector initialized - "
            f"Wake word: '{self.wake_word}', "
            f"Threshold: {self.confidence_threshold}, "
            f"Cooldown: {self.cooldown_seconds}s"
        )

    def audio_callback(self, msg: AudioData):
        """
        Callback for incoming audio data.

        Args:
            msg: AudioData message with raw audio samples
        """
        self.messages_received += 1

        # STRICT sample rate validation - fail if mismatch
        if msg.sample_rate != self.sample_rate:
            if self.total_chunks_processed == 0:  # Log once
                self.get_logger().error(
                    f"CRITICAL: Audio sample rate mismatch! "
                    f"Expected {self.sample_rate}Hz, got {msg.sample_rate}Hz. "
                    f"Audio capture node must resample correctly!"
                )
            # Skip this message - don't process invalid audio
            return

        # Convert audio data to numpy array (keep as int16 for openWakeWord)
        audio_np = np.frombuffer(bytes(msg.data), dtype=np.int16)

        # Debug: Log audio statistics occasionally
        if self.verbose_logging and self.total_chunks_processed % 100 == 0:
            rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
            self.get_logger().info(
                f"Audio chunk stats - min: {audio_np.min()}, "
                f"max: {audio_np.max()}, rms: {rms:.1f}, samples: {len(audio_np)}"
            )

        # Add to buffer (keep as int16)
        with self.buffer_lock:
            self.audio_buffer.extend(audio_np)

    def processing_loop(self):
        """
        Main processing loop running in separate thread.
        Continuously processes audio chunks for wake word detection.
        """
        self.get_logger().info("Wake word processing thread started")

        while self.running:
            try:
                # Get audio chunk from buffer
                with self.buffer_lock:
                    if len(self.audio_buffer) < self.chunk_size:
                        time.sleep(0.01)  # Wait for more data
                        continue

                    # Get chunk as int16 (openWakeWord expects int16)
                    chunk = np.array(list(self.audio_buffer)[: self.chunk_size], dtype=np.int16)
                    # Remove processed samples
                    for _ in range(self.chunk_size):
                        self.audio_buffer.popleft()

                # Process chunk with openWakeWord (expects int16 audio)
                prediction = self.model.predict(chunk)

                self.total_chunks_processed += 1

                # Debug: Log all predictions if verbose
                if self.verbose_logging and self.total_chunks_processed % 50 == 0:
                    self.get_logger().info(
                        f"Prediction keys: {list(prediction.keys())}, "
                        f"Values: {[(k, f'{v:.4f}') for k, v in prediction.items()]}"
                    )

                # Check if wake word detected
                if self.wake_word in prediction:
                    confidence = prediction[self.wake_word]

                    if self.verbose_logging:
                        self.get_logger().debug(f"Wake word confidence: {confidence:.4f}")

                    # Publish confidence
                    conf_msg = Float32()
                    conf_msg.data = float(confidence)
                    self.confidence_pub.publish(conf_msg)

                    # Check if above threshold and not in cooldown
                    current_time = time.time()
                    if confidence >= self.confidence_threshold:
                        if not self.is_in_cooldown:
                            self.handle_detection(confidence)
                        elif self.verbose_logging:
                            self.get_logger().debug(
                                f"Detection in cooldown period (confidence: {confidence:.4f})"
                            )

                    # Update cooldown status
                    if current_time - self.last_detection_time > self.cooldown_seconds:
                        self.is_in_cooldown = False
                else:
                    # Wake word not in prediction - log this issue
                    if self.total_chunks_processed % 100 == 0:
                        self.get_logger().warn(
                            f"Wake word '{self.wake_word}' not found in predictions. "
                            f"Available: {list(prediction.keys())}"
                        )

            except Exception as e:
                self.get_logger().error(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Prevent tight error loop

        self.get_logger().info("Wake word processing thread stopped")

    def handle_detection(self, confidence: float):
        """
        Handle wake word detection.

        Args:
            confidence: Detection confidence score
        """
        current_time = time.time()

        # Update state
        self.last_detection_time = current_time
        self.is_in_cooldown = True
        self.total_detections += 1

        # Log detection
        self.get_logger().info(
            f"Wake word '{self.wake_word}' detected! Confidence: {confidence:.4f}"
        )

        # Publish detection
        detection_msg = Bool()
        detection_msg.data = True
        self.wake_word_pub.publish(detection_msg)

        # Publish audio event
        event_msg = AudioEvent()
        event_msg.header.stamp = self.get_clock().now().to_msg()
        event_msg.event_type = "wake_word_detected"
        event_msg.confidence = float(confidence)
        event_msg.details = f"Wake word: {self.wake_word}"
        self.event_pub.publish(event_msg)

    def publish_status(self):
        """Publish periodic status information."""
        if self.total_chunks_processed > 0:
            self.get_logger().info(
                f"Status - Messages received: {self.messages_received}, "
                f"Chunks processed: {self.total_chunks_processed}, "
                f"Detections: {self.total_detections}, "
                f"Buffer size: {len(self.audio_buffer)}, "
                f"In cooldown: {self.is_in_cooldown}"
            )

    def destroy_node(self):
        """Cleanup on node shutdown."""
        self.get_logger().info("Shutting down wake word detector...")
        self.running = False

        # Wait for processing thread
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        # Log final statistics
        self.get_logger().info(
            f"Final statistics - "
            f"Total chunks: {self.total_chunks_processed}, "
            f"Total detections: {self.total_detections}"
        )

        super().destroy_node()


def main(args=None):
    """Main entry point for wake word detector node."""
    rclpy.init(args=args)

    try:
        node = WakeWordDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in wake word detector node: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
