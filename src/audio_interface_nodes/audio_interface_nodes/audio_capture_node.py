#!/usr/bin/env python3
"""
Audio Capture Node for Local AI Robot Assistant.

Captures audio from USB microphone and publishes to /audio/raw topic.
Implements circular buffer management for wake word detection and
automatic USB device reconnection.
"""

import re
import subprocess
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Header

try:
    from audio_common_msgs.msg import AudioData
except ImportError:
    # Fallback to custom message if audio_common_msgs not available
    from robot_interfaces.msg import AudioData


class AudioCaptureNode(Node):
    """ROS2 node for capturing audio from USB microphone."""

    def __init__(self):
        super().__init__("audio_capture_node")

        # Declare parameters
        self.declare_parameter("config_file", "config/audio_config.yaml")
        self.declare_parameter("audio_topic", "/audio/raw")
        self.declare_parameter("target_sample_rate", 16000)
        self.declare_parameter("channels", 1)
        self.declare_parameter("chunk_duration_ms", 80)  # 80ms chunks for openWakeWord
        self.declare_parameter("buffer_duration", 5.0)  # 5 seconds circular buffer
        self.declare_parameter("device_name", "USB PnP Sound Device")
        self.declare_parameter("reconnect_interval", 2.0)  # seconds
        self.declare_parameter("audio_gain", 1.0)  # Default gain (will be dynamic)

        # Load configuration
        self._load_config()

        # Initialize audio parameters
        self.target_sample_rate = (
            self.get_parameter("target_sample_rate").get_parameter_value().integer_value
        )
        self.channels = self.get_parameter("channels").get_parameter_value().integer_value
        chunk_duration_ms = (
            self.get_parameter("chunk_duration_ms").get_parameter_value().integer_value
        )
        buffer_duration = self.get_parameter("buffer_duration").get_parameter_value().double_value
        self.device_name = self.get_parameter("device_name").get_parameter_value().string_value
        self.reconnect_interval = (
            self.get_parameter("reconnect_interval").get_parameter_value().double_value
        )
        self.audio_gain = self.get_parameter("audio_gain").get_parameter_value().double_value

        # Find audio device
        self.device_string = self._find_audio_device()
        self.hardware_sample_rate = self.target_sample_rate  # We trust plughw to resample

        # Calculate chunk size based on target sample rate (since plughw handles resampling)
        self.chunk_size = int(self.target_sample_rate * chunk_duration_ms / 1000)

        # Calculate circular buffer size (number of chunks at target rate)
        self.buffer_max_chunks = int((buffer_duration * self.target_sample_rate) / self.chunk_size)

        # Circular buffer for audio data
        self.audio_buffer = deque(maxlen=self.buffer_max_chunks)
        self.buffer_lock = threading.Lock()

        # ROS2 publisher
        audio_topic = self.get_parameter("audio_topic").get_parameter_value().string_value
        self.audio_publisher = self.create_publisher(AudioData, audio_topic, 10)

        # Audio capture process
        self.capture_process: Optional[subprocess.Popen] = None
        self.capture_thread: Optional[threading.Thread] = None

        # State management
        self.is_running = True
        self.is_streaming = False
        self.reconnect_thread: Optional[threading.Thread] = None

        # Statistics
        self.frames_captured = 0
        self.last_stats_time = time.time()

        # Start audio capture
        self._initialize_audio()

        # Create timer for statistics reporting
        self.create_timer(10.0, self._report_statistics)

        self.get_logger().info("Audio Capture Node initialized")
        self.get_logger().info(f"  Device: {self.device_string}")
        self.get_logger().info(f"  Target sample rate: {self.target_sample_rate} Hz")
        if self.hardware_sample_rate == self.target_sample_rate:
            self.get_logger().info("  ✅ No resampling needed (rates match)")
        else:
            self.get_logger().info(
                f"  ℹ️  Hardware rate: {self.hardware_sample_rate}Hz -> Target:\
                     {self.target_sample_rate}Hz"
            )
            self.get_logger().info(
                "  Note: 'plughw' device handles hardware resampling automatically"
            )
            self.get_logger().info(
                "  Note: 'plughw' device handles hardware resampling automatically"
            )
        self.get_logger().info(f"  Channels: {self.channels}")
        self.get_logger().info(f"  Chunk size: {self.chunk_size} samples ({chunk_duration_ms}ms)")
        self.get_logger().info(
            f"  Circular buffer: {buffer_duration}s ({self.buffer_max_chunks} chunks)"
        )
        self.get_logger().info(f"  Publishing to: {audio_topic}")

    def _load_config(self):
        """Load audio configuration from YAML file."""
        config_file = self.get_parameter("config_file").get_parameter_value().string_value

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Extract microphone configuration
            if "microphone" in config:
                mic_config = config["microphone"]

                # Update parameters from config
                if "speech_recognition" in mic_config:
                    sr_config = mic_config["speech_recognition"]
                    self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                "target_sample_rate",
                                rclpy.Parameter.Type.INTEGER,
                                sr_config.get("sample_rate", 16000),
                            ),
                            rclpy.parameter.Parameter(
                                "channels",
                                rclpy.Parameter.Type.INTEGER,
                                sr_config.get("channels", 1),
                            ),
                        ]
                    )

                # Get device name
                if "name" in mic_config:
                    self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                "device_name",
                                rclpy.Parameter.Type.STRING,
                                mic_config["name"],
                            )
                        ]
                    )

            # Extract pipeline configuration for chunk duration
            if "pipeline" in config and "wake_word" in config["pipeline"]:
                ww_config = config["pipeline"]["wake_word"]
                if "chunk_duration_ms" in ww_config:
                    self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                "chunk_duration_ms",
                                rclpy.Parameter.Type.INTEGER,
                                ww_config["chunk_duration_ms"],
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

    def _find_audio_device(self) -> str:
        """
        Detects the first USB microphone available via 'arecord -l'.
        Returns the ALSA device string (e.g., 'plughw:1,0') or 'default'.
        Using 'plughw' ensures automatic sample rate conversion.
        """
        try:
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
            output = result.stdout

            # Look for lines like: card 1: Device [USB PnP Sound Device],
            # device 0: USB Audio [USB Audio]
            # Regex to capture card number and device number for USB devices
            match = re.search(r"card (\d+):.*USB.*device (\d+):", output, re.IGNORECASE)

            if match:
                card_num = match.group(1)
                dev_num = match.group(2)
                device = f"plughw:{card_num},{dev_num}"
                self.get_logger().info(f"Found USB microphone: {device}")
                return device

        except Exception as e:
            self.get_logger().error(f"Error detecting USB microphone: {e}")

        self.get_logger().warn("USB microphone not found, using 'default'")
        return "default"

    def _normalize_audio(
        self, audio_data: np.ndarray, target_peak: int = 20000, max_gain: float = 15.0
    ) -> np.ndarray:
        """Normalize audio volume to a target peak amplitude.

        Args:
            audio_data: Input audio array (int16)
            target_peak: Target peak amplitude (default 20000 out of 32767)
            max_gain: Maximum allowed gain factor

        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            gain = min(target_peak / max_val, max_gain)
            # Only apply gain if signal is weak but not silent (noise floor check)
            if max_val > 100 and gain > 1.0:
                return (audio_data * gain).astype(np.int16)
        return audio_data

    def _initialize_audio(self):
        """Initialize arecord subprocess."""
        try:
            # Construct arecord command
            # arecord -D <device> -r <rate> -c <channels> -f S16_LE -t raw
            cmd = [
                "arecord",
                "-D",
                self.device_string,
                "-r",
                str(self.target_sample_rate),
                "-c",
                str(self.channels),
                "-f",
                "S16_LE",
                "-t",
                "raw",
                "--buffer-size=8192",
            ]

            self.get_logger().info(f"Starting audio capture: {' '.join(cmd)}")

            self.capture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            self.is_streaming = True

            # Start reading thread
            self.capture_thread = threading.Thread(target=self._read_audio_stream)
            self.capture_thread.daemon = True
            self.capture_thread.start()

            self.get_logger().info("Audio stream started")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize audio: {e}")
            self.is_streaming = False
            self._schedule_reconnect()

    def _read_audio_stream(self):
        """Read raw audio data from arecord stdout."""
        bytes_per_sample = 2  # S16_LE
        chunk_bytes = self.chunk_size * self.channels * bytes_per_sample
        buffer = b""

        while self.is_streaming and self.capture_process:
            # Check if process is still alive
            if self.capture_process.poll() is not None:
                self.get_logger().error(
                    f"arecord process exited with code {self.capture_process.returncode}"
                )
                if self.capture_process.stderr:
                    err = self.capture_process.stderr.read().decode()
                    self.get_logger().error(f"arecord stderr: {err}")
                break

            try:
                # Read raw bytes
                needed = chunk_bytes - len(buffer)
                data = self.capture_process.stdout.read(needed)

                if not data:
                    # EOF
                    break

                buffer += data

                if len(buffer) >= chunk_bytes:
                    # We have a full chunk
                    process_data = buffer[:chunk_bytes]
                    buffer = buffer[chunk_bytes:]

                    # Convert to numpy array
                    # S16_LE -> int16
                    audio_int16 = np.frombuffer(process_data, dtype=np.int16)

                    self._process_audio_chunk(audio_int16)

            except Exception as e:
                self.get_logger().error(f"Error reading audio stream: {e}")
                break

        self.get_logger().warn("Audio capture thread stopped")
        self.is_streaming = False
        if self.is_running:
            self._schedule_reconnect()

    def _process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process incoming audio data."""
        try:
            # Normalize audio (Dynamic Gain Control)
            audio_normalized = self._normalize_audio(audio_chunk)

            # Add to circular buffer
            with self.buffer_lock:
                self.audio_buffer.append(audio_normalized.copy())

            # Publish to ROS2 topic
            self._publish_audio(audio_normalized)

            self.frames_captured += 1

        except Exception as e:
            self.get_logger().error(f"Error in audio processing: {e}")

    def _publish_audio(self, audio_array: np.ndarray):
        """Publish audio data to ROS2 topic."""
        try:
            # audio_array is already int16 and normalized
            audio_bytes = audio_array.tobytes()

            msg = AudioData()

            # Set header
            if hasattr(msg, "header"):
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "microphone"

            # Set audio data
            msg.data = audio_bytes

            # Set metadata if available
            if hasattr(msg, "sample_rate"):
                msg.sample_rate = self.target_sample_rate
            if hasattr(msg, "channels"):
                msg.channels = self.channels
            if hasattr(msg, "encoding"):
                msg.encoding = "S16_LE"
            if hasattr(msg, "chunk_size"):
                msg.chunk_size = len(audio_array)

            self.audio_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing audio: {e}")

    def _report_statistics(self):
        """Report audio capture statistics."""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time

        if elapsed > 0 and self.is_streaming:
            capture_rate = self.frames_captured / elapsed
            expected_rate = self.target_sample_rate / self.chunk_size

            self.get_logger().info(
                f"Audio stats: {capture_rate:.1f} chunks/s "
                f"(expected: {expected_rate:.1f}), "
                f"buffer: {len(self.audio_buffer)}/{self.buffer_max_chunks} chunks"
            )

            # Reset counters
            self.frames_captured = 0
            self.last_stats_time = current_time
        elif not self.is_streaming:
            self.get_logger().warn("Audio stream not active, attempting to reconnect...")

    def _schedule_reconnect(self):
        """Schedule reconnection attempt."""
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=self._reconnect_loop)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()

    def _reconnect_loop(self):
        """Attempt to reconnect to audio device."""
        self.get_logger().info("Starting audio device reconnection attempts...")

        attempt = 0
        while self.is_running and not self.is_streaming:
            attempt += 1
            self.get_logger().info(f"Reconnection attempt {attempt}...")

            # Clean up existing resources
            self._cleanup_audio()

            # Wait before retry (exponential backoff, max 10 seconds)
            wait_time = min(self.reconnect_interval * (2 ** min(attempt - 1, 3)), 10.0)
            time.sleep(wait_time)

            # Try to initialize
            try:
                # Re-detect device in case it changed
                self.device_string = self._find_audio_device()
                self._initialize_audio()
                if self.is_streaming:
                    self.get_logger().info("Successfully reconnected to audio device")
                    return
            except Exception as e:
                self.get_logger().error(f"Reconnection attempt {attempt} failed: {e}")

        self.get_logger().info("Reconnection loop terminated")

    def _cleanup_audio(self):
        """Clean up audio resources."""
        try:
            self.is_streaming = False

            if self.capture_process:
                self.capture_process.terminate()
                try:
                    self.capture_process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.capture_process.kill()
                self.capture_process = None

            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)

        except Exception as e:
            self.get_logger().error(f"Error cleaning up audio: {e}")

    def get_buffer_data(self) -> np.ndarray:
        """Get all data from circular buffer as concatenated array."""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self.audio_buffer))

    def destroy_node(self):
        """Clean up on shutdown."""
        self.get_logger().info("Shutting down Audio Capture Node...")
        self.is_running = False
        self._cleanup_audio()
        super().destroy_node()


def main(args=None):
    """Main function to run the audio capture node."""
    rclpy.init(args=args)

    try:
        node = AudioCaptureNode()
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
