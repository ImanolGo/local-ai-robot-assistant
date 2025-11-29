#!/usr/bin/env python3
"""
Piper TTS ROS2 Node.

This node provides text-to-speech synthesis using the Piper TTS system.
Subscribes to text commands and publishes synthesized audio data.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from audio_common_msgs.msg import AudioData
from rclpy.node import Node
from std_msgs.msg import String

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from piper import PiperVoice
except ImportError:
    print("ERROR: piper-tts package not found. Install with: pip install piper-tts")
    sys.exit(1)


class PiperTTSNode(Node):
    """ROS2 node for Piper text-to-speech synthesis."""

    def __init__(self):
        super().__init__("piper_tts_node")

        # Declare parameters
        self.declare_parameter(
            "model_path",
            "/home/imanolgo/repos/local-ai-robot-assistant/models/piper_voice/en_US-lessac-medium.onnx",  # noqa E501
        )
        self.declare_parameter("audio_topic", "synthesized_audio")
        self.declare_parameter("text_topic", "text_to_synthesize")
        self.declare_parameter("synthesis_rate_hz", 10.0)  # Max synthesis rate

        # Get parameters
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        audio_topic = self.get_parameter("audio_topic").get_parameter_value().string_value
        text_topic = self.get_parameter("text_topic").get_parameter_value().string_value
        self.synthesis_rate = (
            self.get_parameter("synthesis_rate_hz").get_parameter_value().double_value
        )

        # Initialize Piper voice model
        self.voice: Optional[PiperVoice] = None
        self.model_path = Path(model_path)
        self._load_model()

        # Initialize ROS2 interfaces
        self.audio_publisher = self.create_publisher(AudioData, audio_topic, 10)
        self.text_subscription = self.create_subscription(
            String, text_topic, self.text_callback, 10
        )

        # Synthesis management
        self.synthesis_lock = threading.Lock()
        self.last_synthesis_time = 0.0
        self.min_synthesis_interval = 1.0 / self.synthesis_rate

        self.get_logger().info("Piper TTS node initialized")
        self.get_logger().info(f"  Model: {self.model_path}")
        self.get_logger().info(f"  Audio topic: {audio_topic}")
        self.get_logger().info(f"  Text topic: {text_topic}")
        self.get_logger().info(f"  Max synthesis rate: {self.synthesis_rate} Hz")

    def _load_model(self):
        """Load the Piper voice model."""
        if not self.model_path.exists():
            self.get_logger().error(f"Model file not found: {self.model_path}")
            return

        config_path = self.model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            self.get_logger().error(f"Config file not found: {config_path}")
            return

        self.get_logger().info(f"Loading Piper model: {self.model_path}")
        start_time = time.time()

        try:
            self.voice = PiperVoice.load(str(self.model_path))
            load_time = time.time() - start_time
            self.get_logger().info(f"Model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            self.get_logger().error(f"Failed to load Piper model: {e}")
            self.voice = None

    def text_callback(self, msg: String):
        """Handle incoming text synthesis requests."""
        text = msg.data.strip()

        if not text:
            self.get_logger().debug("Empty text received, skipping synthesis")
            return

        if not self.voice:
            self.get_logger().error("Voice model not loaded, cannot synthesize")
            return

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_synthesis_time

        if time_since_last < self.min_synthesis_interval:
            self.get_logger().debug(
                f"Rate limiting: skipping synthesis (last synthesis {time_since_last:.3f}s ago)"
            )
            return

        # Perform synthesis in thread to avoid blocking
        synthesis_thread = threading.Thread(target=self._synthesize_text, args=(text,))
        synthesis_thread.start()

    def _synthesize_text(self, text: str):
        """Synthesize text to speech and publish audio data."""
        with self.synthesis_lock:
            try:
                self.get_logger().info(
                    f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'"
                )
                start_time = time.time()

                # Synthesize audio
                audio_chunks = []
                for audio_chunk in self.voice.synthesize(text):
                    audio_chunks.append(audio_chunk.audio_int16_bytes)

                # Concatenate all chunks
                audio_bytes = b"".join(audio_chunks)
                synthesis_time = time.time() - start_time

                # Convert to audio message
                audio_msg = AudioData()
                audio_msg.data = list(audio_bytes)

                # Set audio format information (assuming 16-bit PCM)
                audio_msg.data = audio_bytes

                # Publish synthesized audio
                self.audio_publisher.publish(audio_msg)

                # Update timing
                self.last_synthesis_time = time.time()

                word_count = len(text.split())
                self.get_logger().info(
                    f"Synthesis completed: {word_count} words in {synthesis_time:.3f}s "
                    f"({synthesis_time/word_count:.3f}s/word)"
                )

            except Exception as e:
                self.get_logger().error(f"Synthesis failed: {e}")

    def destroy_node(self):
        """Cleanup on shutdown."""
        self.get_logger().info("Shutting down Piper TTS node")
        super().destroy_node()


def main(args=None):
    """Main function to run the Piper TTS node."""
    rclpy.init(args=args)

    try:
        node = PiperTTSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
