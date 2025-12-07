#!/usr/bin/env python3
"""
Streamlined Audio Playback Node for Local AI Robot Assistant.

This node provides unified audio output with:
1. Integrated Piper TTS synthesis (no separate tts_node needed)
2. Event-driven notification sounds (wake word, speech end)
3. Priority-based playback queue with interruption support
4. No audio data transmission over ROS2 (text messages only)
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import sounddevice as sd
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import AudioEvent

try:
    from piper import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class AudioType:
    """Audio type constants for queue management."""

    NOTIFICATION = "notification"
    TTS = "tts"
    OTHER = "other"


class AudioPlaybackNode(Node):
    """Streamlined audio playback node with integrated TTS and notifications."""

    def __init__(self):
        super().__init__("audio_playback_node")

        # Declare base parameters
        self.declare_parameter("config_file", "config/audio_config.yaml")
        self.declare_parameter("sample_rate", 48000)  # Speaker native rate
        self.declare_parameter("channels", 2)  # Stereo
        self.declare_parameter("device_name", "UACDemoV1.0")
        self.declare_parameter("volume_normalization", True)
        self.declare_parameter("max_amplitude", 0.95)
        self.declare_parameter("reconnect_interval", 2.0)

        # TTS parameters
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_model_path", "models/piper_voice/en_US-lessac-medium.onnx")
        self.declare_parameter("tts_priority", 5)
        self.declare_parameter("tts_lazy_loading", True)

        # Notification parameters
        self.declare_parameter("notifications_enabled", True)
        self.declare_parameter("notification_priority", 1)
        self.declare_parameter("allow_interruption", True)  # Notifications can interrupt TTS

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

        # TTS configuration
        self.tts_enabled = self.get_parameter("tts_enabled").get_parameter_value().bool_value
        self.tts_model_path = Path(
            self.get_parameter("tts_model_path").get_parameter_value().string_value
        )
        self.tts_priority = self.get_parameter("tts_priority").get_parameter_value().integer_value
        self.tts_lazy_loading = (
            self.get_parameter("tts_lazy_loading").get_parameter_value().bool_value
        )

        # Notification configuration
        self.notifications_enabled = (
            self.get_parameter("notifications_enabled").get_parameter_value().bool_value
        )
        self.notification_priority = (
            self.get_parameter("notification_priority").get_parameter_value().integer_value
        )
        self.allow_interruption = (
            self.get_parameter("allow_interruption").get_parameter_value().bool_value
        )

        # Find audio device
        self.device_index = self._find_audio_device()

        # Priority queue for playback (lower number = higher priority)
        self.playback_queue = queue.PriorityQueue()

        # TTS State
        self.tts_voice: Optional[PiperVoice] = None
        self.tts_lock = threading.Lock()

        # Notification sounds (preloaded)
        self.notification_sounds = {}
        self.notification_event_map = {}

        # ROS2 Subscribers
        if self.tts_enabled:
            self.tts_subscriber = self.create_subscription(
                String, "/audio/tts_request", self.tts_callback, 10
            )

        if self.notifications_enabled:
            self.event_subscriber = self.create_subscription(
                AudioEvent, "/audio/events", self.event_callback, 10
            )

        # Event publisher
        self.event_publisher = self.create_publisher(AudioEvent, "/audio/events", 10)

        # Audio stream
        self.stream: Optional[sd.OutputStream] = None

        # State management
        self.is_running = True
        self.is_streaming = False
        self.is_playing = False
        self.current_audio_type = None
        self.playback_lock = threading.Lock()
        self.reconnect_thread: Optional[threading.Thread] = None

        # Playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

        # Statistics
        self.audio_chunks_played = 0
        self.tts_requests_processed = 0
        self.notifications_played = 0
        self.last_stats_time = time.time()

        # Initialize TTS (if not lazy loading)
        if self.tts_enabled and not self.tts_lazy_loading:
            self._initialize_tts()

        # Load notification sounds
        if self.notifications_enabled:
            self._load_notification_sounds()

        # Initialize audio
        self._initialize_audio()

        # Create timer for statistics
        self.create_timer(10.0, self._report_statistics)

        self.get_logger().info("Streamlined Audio Playback Node initialized")
        self.get_logger().info(f"  Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"  Channels: {self.channels}")
        self.get_logger().info(f"  TTS enabled: {self.tts_enabled}")
        self.get_logger().info(f"  Notifications enabled: {self.notifications_enabled}")
        self.get_logger().info(f"  Interruption allowed: {self.allow_interruption}")

    def _load_config(self):
        """Load audio configuration from YAML file."""
        config_file = self.get_parameter("config_file").get_parameter_value().string_value

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Extract speaker configuration
            if "speaker" in config:
                speaker_config = config["speaker"]

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

            # Extract playback configuration (NEW)
            if "playback" in config:
                playback_config = config["playback"]

                # TTS configuration
                if "tts" in playback_config:
                    tts_config = playback_config["tts"]
                    params = []

                    if "enabled" in tts_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "tts_enabled",
                                rclpy.Parameter.Type.BOOL,
                                tts_config["enabled"],
                            )
                        )
                    if "model_path" in tts_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "tts_model_path",
                                rclpy.Parameter.Type.STRING,
                                tts_config["model_path"],
                            )
                        )
                    if "priority" in tts_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "tts_priority",
                                rclpy.Parameter.Type.INTEGER,
                                tts_config["priority"],
                            )
                        )
                    if "lazy_loading" in tts_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "tts_lazy_loading",
                                rclpy.Parameter.Type.BOOL,
                                tts_config["lazy_loading"],
                            )
                        )

                    if params:
                        self.set_parameters(params)

                # Notification configuration
                if "notifications" in playback_config:
                    notif_config = playback_config["notifications"]
                    params = []

                    if "enabled" in notif_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "notifications_enabled",
                                rclpy.Parameter.Type.BOOL,
                                notif_config["enabled"],
                            )
                        )
                    if "priority" in notif_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "notification_priority",
                                rclpy.Parameter.Type.INTEGER,
                                notif_config["priority"],
                            )
                        )
                    if "allow_interruption" in notif_config:
                        params.append(
                            rclpy.parameter.Parameter(
                                "allow_interruption",
                                rclpy.Parameter.Type.BOOL,
                                notif_config["allow_interruption"],
                            )
                        )

                    # Store sound mapping
                    if "sounds" in notif_config:
                        self.notification_event_map = notif_config["sounds"]

                    if params:
                        self.set_parameters(params)

            self.get_logger().info(f"Loaded configuration from {config_file}")

        except FileNotFoundError:
            self.get_logger().warn(
                f"Config file not found: {config_file}, using default parameters"
            )
        except Exception as e:
            self.get_logger().error(f"Error loading config: {e}")

    def _find_audio_device(self) -> Optional[int]:
        """Find the audio output device index by name."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["max_output_channels"] > 0 and self.device_name in device["name"]:
                    self.get_logger().info(f"Found speaker: {device['name']} (index: {i})")

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

    def _initialize_tts(self):
        """Initialize Piper TTS model."""
        if not PIPER_AVAILABLE:
            self.get_logger().error("Piper TTS not available. Install with: pip install piper-tts")
            self.tts_enabled = False
            return

        if self.tts_voice is not None:
            return  # Already loaded

        try:
            if not self.tts_model_path.exists():
                self.get_logger().error(f"TTS model not found: {self.tts_model_path}")
                self.tts_enabled = False
                return

            config_path = self.tts_model_path.with_suffix(".onnx.json")
            if not config_path.exists():
                self.get_logger().error(f"TTS config not found: {config_path}")
                self.tts_enabled = False
                return

            self.get_logger().info(f"Loading Piper TTS model: {self.tts_model_path}")
            start_time = time.time()

            self.tts_voice = PiperVoice.load(str(self.tts_model_path))

            load_time = time.time() - start_time
            self.get_logger().info(f"Piper TTS model loaded in {load_time:.2f}s")

        except Exception as e:
            self.get_logger().error(f"Failed to load Piper TTS: {e}")
            self.tts_voice = None
            self.tts_enabled = False

    def _load_notification_sounds(self):
        """Load notification sound files."""
        if not SOUNDFILE_AVAILABLE:
            self.get_logger().error("soundfile not available. Install with: pip install soundfile")
            self.notifications_enabled = False
            return

        if not self.notification_event_map:
            self.get_logger().warn("No notification sound mapping configured")
            return

        for event_type, sound_path in self.notification_event_map.items():
            try:
                if not os.path.exists(sound_path):
                    self.get_logger().warn(f"Notification sound not found: {sound_path}")
                    continue

                # Load audio file
                audio_data, source_sr = sf.read(sound_path)
                self.notification_sounds[event_type] = (audio_data, source_sr)

                self.get_logger().info(
                    f"Loaded notification: {event_type} -> {sound_path} "
                    f"({len(audio_data)} samples, {source_sr}Hz)"
                )

            except Exception as e:
                self.get_logger().error(
                    f"Failed to load notification {event_type} from {sound_path}: {e}"
                )

    def tts_callback(self, msg: String):
        """Handle TTS text requests."""
        if not self.tts_enabled:
            self.get_logger().warn("TTS is disabled")
            return

        text = msg.data.strip()
        if not text:
            self.get_logger().debug("Empty text received, skipping TTS")
            return

        # Lazy load TTS model if needed
        if self.tts_voice is None:
            self._initialize_tts()

        if self.tts_voice is None:
            self.get_logger().error("TTS voice not available")
            return

        # Synthesize in background thread
        synthesis_thread = threading.Thread(
            target=self._synthesize_and_queue, args=(text,), daemon=True
        )
        synthesis_thread.start()

    def _synthesize_and_queue(self, text: str):
        """Synthesize text to audio and queue for playback."""
        with self.tts_lock:
            try:
                self.get_logger().info(
                    f"Synthesizing TTS: '{text[:50]}{'...' if len(text) > 50 else ''}'"
                )
                start_time = time.time()

                # Synthesize audio chunks
                audio_chunks = []
                for audio_chunk in self.tts_voice.synthesize(text):
                    audio_chunks.append(audio_chunk.audio_int16_bytes)

                # Concatenate all chunks
                audio_bytes = b"".join(audio_chunks)

                # Convert to numpy array (int16 -> float32)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0

                synthesis_time = time.time() - start_time

                # Get source sample rate
                source_rate = self.tts_voice.config.sample_rate

                # Queue for playback with normal priority
                self.playback_queue.put(
                    (
                        self.tts_priority,
                        time.time(),
                        AudioType.TTS,
                        (audio_float, source_rate),
                    )
                )

                word_count = len(text.split())
                self.get_logger().info(
                    f"TTS synthesized: {word_count} words in {synthesis_time:.3f}s "
                    f"({synthesis_time/word_count:.3f}s/word)"
                )

                self.tts_requests_processed += 1

            except Exception as e:
                self.get_logger().error(f"TTS synthesis failed: {e}")

    def event_callback(self, msg: AudioEvent):
        """Handle audio events for notification triggers."""
        if not self.notifications_enabled:
            return

        event_type = msg.event_type

        if event_type in self.notification_sounds:
            audio_data, source_sr = self.notification_sounds[event_type]

            # Queue notification with high priority
            self.playback_queue.put(
                (
                    self.notification_priority,
                    time.time(),
                    AudioType.NOTIFICATION,
                    (audio_data.copy(), source_sr),
                )
            )

            self.get_logger().info(f"Queued notification for event: {event_type}")

    def _playback_loop(self):
        """Main playback loop running in separate thread."""
        while self.is_running:
            try:
                # Wait for audio data (with timeout to allow checking is_running)
                try:
                    priority, timestamp, audio_type, audio_tuple = self.playback_queue.get(
                        timeout=0.5
                    )
                except queue.Empty:
                    continue

                # Check for interruption
                if self.allow_interruption and audio_type == AudioType.NOTIFICATION:
                    # Notification can interrupt current playback
                    # (stream.write is blocking, so we can't truly interrupt mid-chunk,
                    # but next item will be the notification)
                    pass

                # Process and play audio
                audio_data, source_rate = audio_tuple
                self._play_audio(audio_data, source_rate, audio_type)

                # Mark task as done
                self.playback_queue.task_done()

            except Exception as e:
                self.get_logger().error(f"Error in playback loop: {e}")
                time.sleep(0.1)  # Avoid tight loop on error

    def _play_audio(self, audio_data: np.ndarray, source_rate: int, audio_type: str):
        """Process and play audio data."""
        try:
            with self.playback_lock:
                # Publish playback started event
                self._publish_event("playback_started", f"type: {audio_type}")

                self.is_playing = True
                self.current_audio_type = audio_type

                # Process audio for playback
                processed_audio = self._process_audio_for_playback(audio_data, source_rate)

                # Play audio
                if self.stream and self.is_streaming:
                    self.stream.write(processed_audio)
                    self.audio_chunks_played += 1

                    if audio_type == AudioType.NOTIFICATION:
                        self.notifications_played += 1
                else:
                    self.get_logger().warn("Stream not available, skipping playback")

                self.is_playing = False
                self.current_audio_type = None

                # Publish playback complete event
                self._publish_event("playback_complete", f"type: {audio_type}")

        except Exception as e:
            self.get_logger().error(f"Error playing audio: {e}")
            self.is_playing = False
            self.current_audio_type = None
            self._publish_event("audio_error", str(e))

    def _process_audio_for_playback(self, audio_data: np.ndarray, source_rate: int) -> np.ndarray:
        """Convert audio to speaker format (resampling, channel conversion, normalization)."""
        # Ensure float32
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Resample if needed
        if source_rate != self.sample_rate:
            audio_data = self._resample_audio(audio_data, source_rate, self.sample_rate)

        # Convert mono to stereo if needed
        if audio_data.ndim == 1 and self.channels == 2:
            audio_data = np.column_stack([audio_data, audio_data])
        elif audio_data.ndim == 2 and audio_data.shape[1] == 2 and self.channels == 1:
            audio_data = np.mean(audio_data, axis=1)

        # Apply volume normalization
        if self.volume_normalization:
            audio_data = self._normalize_volume(audio_data)

        return audio_data

    def _resample_audio(
        self, audio_data: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio data using linear interpolation."""
        if source_rate == target_rate:
            return audio_data

        # Handle stereo
        if audio_data.ndim == 2:
            # Resample each channel
            resampled_channels = []
            for ch in range(audio_data.shape[1]):
                resampled_channels.append(
                    self._resample_audio(audio_data[:, ch], source_rate, target_rate)
                )
            return np.column_stack(resampled_channels)

        # Simple linear interpolation for mono
        original_length = len(audio_data)
        resample_ratio = target_rate / source_rate
        target_length = int(original_length * resample_ratio)

        resampled = np.interp(
            np.linspace(0, original_length - 1, target_length),
            np.arange(original_length),
            audio_data,
        )

        return resampled.astype(np.float32)

    def _normalize_volume(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio volume to prevent clipping."""
        peak = np.abs(audio_data).max()

        if peak > self.max_amplitude:
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
                f"TTS: {self.tts_requests_processed}, "
                f"notifications: {self.notifications_played}, "
                f"playing: {self.is_playing} ({self.current_audio_type}), "
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
        self.get_logger().info("Shutting down Streamlined Audio Playback Node...")
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
