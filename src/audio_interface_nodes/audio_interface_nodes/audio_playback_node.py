#!/usr/bin/env python3
"""
Audio Playback Node for Local AI Robot Assistant.

Receives audio data from ROS2 topics and plays through USB speakers.
Implements priority queue-based playback with volume normalization.
"""

import queue
import threading
import time
from typing import Optional

import numpy as np
import rclpy
import sounddevice as sd
import yaml
from rclpy.node import Node

try:
    from audio_common_msgs.msg import AudioData
except ImportError:
    from robot_interfaces.msg import AudioData

from robot_interfaces.msg import AudioEvent


class AudioPlaybackNode(Node):
    """ROS2 node for playing audio through USB speakers."""

    def __init__(self):
        super().__init__("audio_playback_node")

        # Declare parameters
        self.declare_parameter("config_file", "config/audio_config.yaml")
        self.declare_parameter("audio_topic", "/audio/tts_output")
        self.declare_parameter("event_topic", "/audio/events")
        self.declare_parameter("sample_rate", 48000)  # Speaker native rate
        self.declare_parameter("channels", 2)  # Stereo
        self.declare_parameter("device_name", "UACDemoV1.0")
        self.declare_parameter("volume_normalization", True)
        self.declare_parameter("max_amplitude", 0.95)  # Prevent clipping
        self.declare_parameter("reconnect_interval", 2.0)

        # Load configuration
        self._load_config()

        # Audio parameters
        self.sample_rate = self.get_parameter("sample_rate").get_parameter_value().integer_value
        self.channels = self.get_parameter("channels").get_parameter_value().integer_value
        self.device_name = self.get_parameter("device_name").get_parameter_value().string_value
        self.volume_normalization = (
            self.get_parameter("volume_normalization").get_parameter_value().bool_value
        )
        self.max_amplitude = self.get_parameter("max_amplitude").get_parameter_value().double_value
        self.reconnect_interval = (
            self.get_parameter("reconnect_interval").get_parameter_value().double_value
        )

        # Find audio device
        self.device_index = self._find_audio_device()

        # Priority queue for playback (lower number = higher priority)
        self.playback_queue = queue.PriorityQueue()

        # ROS2 subscribers
        audio_topic = self.get_parameter("audio_topic").get_parameter_value().string_value
        self.audio_subscriber = self.create_subscription(
            AudioData, audio_topic, self.audio_callback, 10
        )

        # Event publisher
        event_topic = self.get_parameter("event_topic").get_parameter_value().string_value
        self.event_publisher = self.create_publisher(AudioEvent, event_topic, 10)

        # Audio stream
        self.stream: Optional[sd.OutputStream] = None

        # State management
        self.is_running = True
        self.is_streaming = False
        self.is_playing = False
        self.reconnect_thread: Optional[threading.Thread] = None

        # Playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

        # Statistics
        self.audio_chunks_played = 0
        self.last_stats_time = time.time()

        # Initialize audio
        self._initialize_audio()

        # Create timer for statistics
        self.create_timer(10.0, self._report_statistics)

        self.get_logger().info("Audio Playback Node initialized")
        self.get_logger().info(f"  Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"  Channels: {self.channels}")
        self.get_logger().info(f"  Volume normalization: {self.volume_normalization}")
        self.get_logger().info(f"  Max amplitude: {self.max_amplitude}")
        self.get_logger().info(f"  Subscribing to: {audio_topic}")
        self.get_logger().info(f"  Publishing events to: {event_topic}")

    def _load_config(self):
        """Load audio configuration from YAML file."""
        config_file = self.get_parameter("config_file").get_parameter_value().string_value

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Extract speaker configuration
            if "speaker" in config:
                speaker_config = config["speaker"]

                # Update parameters from config
                if "native_format" in speaker_config:
                    native_format = speaker_config["native_format"]
                    self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                "sample_rate",
                                rclpy.Parameter.Type.INTEGER,
                                native_format.get("sample_rate", 48000),
                            ),
                            rclpy.parameter.Parameter(
                                "channels",
                                rclpy.Parameter.Type.INTEGER,
                                native_format.get("channels", 2),
                            ),
                        ]
                    )

                # Get device name
                if "name" in speaker_config:
                    self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                "device_name",
                                rclpy.Parameter.Type.STRING,
                                speaker_config["name"],
                            )
                        ]
                    )

            self.get_logger().info(f"Loaded configuration from {config_file}")

        except FileNotFoundError:
            self.get_logger().warn(
                f"Config file not found: {config_file}, using default parameters"
            )
        except Exception as e:
            self.get_logger().error(f"Error loading config: {e}")

    def _find_audio_device(self) -> Optional[int]:
        """Find the audio output device index by  name."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["max_output_channels"] > 0 and self.device_name in device["name"]:
                    self.get_logger().info(f"Found speaker: {device['name']} (index: {i})")

                    # Verify sample rate support
                    try:
                        sd.check_output_settings(
                            device=i,
                            samplerate=self.sample_rate,
                            channels=self.channels,
                        )
                        self.get_logger().info(
                            f"Using {self.sample_rate}Hz, {self.channels}ch for device {i}"
                        )
                        return i
                    except sd.PortAudioError as e:
                        self.get_logger().warn(f"Device {i} doesn't support config: {e}")
                        continue

            # Device not found by name, use default
            self.get_logger().warn(
                f"Device '{self.device_name}' not found, using default output device"
            )
            default_device = sd.query_devices(kind="output")
            if default_device:
                self.get_logger().info(f"Default device: {default_device['name']}")
                return None
            else:
                raise RuntimeError("No output audio device available")

        except Exception as e:
            self.get_logger().error(f"Error finding audio device: {e}")
            raise

    def _initialize_audio(self):
        """Initialize sounddevice output stream."""
        try:
            stream_kwargs = {
                "samplerate": self.sample_rate,
                "channels": self.channels,
                "dtype": np.float32,
                "latency": "low",
            }

            if self.device_index is not None:
                stream_kwargs["device"] = (None, self.device_index)

            self.stream = sd.OutputStream(**stream_kwargs)
            self.stream.start()
            self.is_streaming = True

            self.get_logger().info(
                f"Audio output stream started: {self.sample_rate}Hz, {self.channels}ch"
            )

        except Exception as e:
            self.get_logger().error(f"Failed to initialize audio output: {e}")
            self.is_streaming = False
            self._schedule_reconnect()

    def audio_callback(self, msg: AudioData):
        """Handle incoming audio data messages."""
        try:
            # Default priority for TTS output
            priority = 5

            # Add to playback queue
            self.playback_queue.put((priority, time.time(), msg))

            self.get_logger().debug(
                f"Queued audio chunk (queue size: {self.playback_queue.qsize()})"
            )

        except Exception as e:
            self.get_logger().error(f"Error queueing audio: {e}")

    def _playback_loop(self):
        """Main playback loop running in separate thread."""
        while self.is_running:
            try:
                # Wait for audio data (with timeout to allow checking is_running)
                try:
                    priority, timestamp, msg = self.playback_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Process and play audio
                self._play_audio(msg)

                # Mark task as done
                self.playback_queue.task_done()

            except Exception as e:
                self.get_logger().error(f"Error in playback loop: {e}")
                time.sleep(0.1)  # Avoid tight loop on error

    def _play_audio(self, msg: AudioData):
        """Process and play audio data."""
        try:
            # Publish playback started event
            self._publish_event(AudioEvent.EVENT_PLAYBACK_STARTED, "")

            self.is_playing = True

            # Convert bytes to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16)

            # Convert int16 to float32 [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Get source parameters
            source_rate = msg.sample_rate if hasattr(msg, "sample_rate") else 16000
            source_channels = msg.channels if hasattr(msg, "channels") else 1

            # Resample if needed
            if source_rate != self.sample_rate:
                audio_float = self._resample_audio(audio_float, source_rate, self.sample_rate)

            # Convert mono to stereo if needed
            if source_channels == 1 and self.channels == 2:
                audio_float = np.column_stack([audio_float, audio_float])
            elif source_channels == 2 and self.channels == 1:
                audio_float = np.mean(audio_float.reshape(-1, 2), axis=1)

            # Apply volume normalization
            if self.volume_normalization:
                audio_float = self._normalize_volume(audio_float)

            # Play audio
            if self.stream and self.is_streaming:
                self.stream.write(audio_float)
                self.audio_chunks_played += 1
            else:
                self.get_logger().warn("Stream not available, skipping playback")

            self.is_playing = False

            # Publish playback complete event
            self._publish_event(AudioEvent.EVENT_PLAYBACK_COMPLETE, "")

        except Exception as e:
            self.get_logger().error(f"Error playing audio: {e}")
            self.is_playing = False
            self._publish_event(AudioEvent.EVENT_AUDIO_ERROR, str(e))

    def _resample_audio(
        self, audio_data: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio data."""
        if source_rate == target_rate:
            return audio_data

        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(source_rate, target_rate)
        up = target_rate // g
        down = source_rate // g

        resampled = resample_poly(audio_data, up, down)
        return resampled.astype(np.float32)

    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio volume to prevent clipping."""
        # Find peak amplitude
        peak = np.abs(audio_data).max()

        if peak > self.max_amplitude:
            # Scale down to prevent clipping
            scale_factor = self.max_amplitude / peak
            audio_data = audio_data * scale_factor
            self.get_logger().debug(f"Normalized audio (peak: {peak:.3f} -> {self.max_amplitude})")

        return audio_data

    def _publish_event(self, event_type: str, data: str):
        """Publish audio event."""
        try:
            event = AudioEvent()
            event.header.stamp = self.get_clock().now().to_msg()
            event.event_type = event_type
            event.data = data
            event.device_id = self.device_name if self.device_name else "default"

            self.event_publisher.publish(event)

        except Exception as e:
            self.get_logger().error(f"Error publishing event: {e}")

    def _report_statistics(self):
        """Report playback statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time

        queue_size = self.playback_queue.qsize()

        if elapsed > 0:
            playback_rate = self.audio_chunks_played / elapsed

            self.get_logger().info(
                f"Playback stats: {playback_rate:.1f} chunks/s, "
                f"queue: {queue_size}, "
                f"playing: {self.is_playing}, "
                f"streaming: {self.is_streaming}"
            )

            # Reset counters
            self.audio_chunks_played = 0
            self.last_stats_time = current_time

        if not self.is_streaming:
            self.get_logger().warn("Audio output stream not active")

    def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
            self.reconnect_thread.start()

    def _reconnect_loop(self):
        """Attempt to reconnect to audio output device."""
        self.get_logger().info("Starting audio output device reconnection attempts...")

        attempt = 0
        while self.is_running and not self.is_streaming:
            attempt += 1
            self.get_logger().info(f"Reconnection attempt {attempt}...")

            # Clean up
            self._cleanup_audio()

            # Wait before retry
            wait_time = min(self.reconnect_interval * (2 ** min(attempt - 1, 3)), 10.0)
            time.sleep(wait_time)

            # Try to initialize
            try:
                self.device_index = self._find_audio_device()
                self._initialize_audio()
                if self.is_streaming:
                    self.get_logger().info("Successfully reconnected to audio output device")
                    return
            except Exception as e:
                self.get_logger().error(f"Reconnection attempt {attempt} failed: {e}")

        self.get_logger().info("Reconnection loop terminated")

    def _cleanup_audio(self):
        """Clean up audio resources."""
        try:
            if self.stream is not None:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
                self.stream = None

        except Exception as e:
            self.get_logger().error(f"Error cleaning up audio output: {e}")

    def destroy_node(self):
        """Clean up on shutdown."""
        self.get_logger().info("Shutting down Audio Playback Node...")
        self.is_running = False

        # Wait for playback queue to empty (with timeout)
        timeout = 5.0
        start_time = time.time()
        while not self.playback_queue.empty() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        self._cleanup_audio()
        super().destroy_node()


def main(args=None):
    """Main function to run the audio playback node."""
    rclpy.init(args=args)

    try:
        node = AudioPlaybackNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
