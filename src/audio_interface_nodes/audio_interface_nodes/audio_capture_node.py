#!/usr/bin/env python3
"""
Self-Contained Audio Processing Pipeline for Local AI Robot Assistant.

This node implements a complete audio processing pipeline:
1. Captures audio from USB microphone (no ROS2 streaming)
2. Runs wake word detection continuously (openWakeWord)
3. Activates VAD after wake word detection (Silero VAD)
4. Transcribes speech using faster-whisper
5. Publishes only lightweight control messages

State Machine:
  IDLE -> WAKE_WORD_DETECTED -> RECORDING -> TRANSCRIBING -> IDLE
"""

import os
import re
import subprocess
import threading
import time
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node

try:
    from robot_interfaces.msg import AudioEvent, TranscriptionResult
except ImportError:
    AudioEvent = None
    TranscriptionResult = None

try:
    from openwakeword.model import Model as WakeWordModel

    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from silero_vad import VADIterator, load_silero_vad

    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False


class PipelineState(Enum):
    """Audio pipeline state machine states."""

    IDLE = "idle"  # Listening for wake word
    WAKE_WORD_DETECTED = "wake_word_detected"  # Wake word triggered
    RECORDING = "recording"  # VAD active, capturing speech
    TRANSCRIBING = "transcribing"  # Running Whisper
    ERROR = "error"  # Error state


class AudioCaptureNode(Node):
    """Self-contained audio processing pipeline node."""

    def __init__(self):
        super().__init__("audio_capture_node")

        # Declare parameters
        self.declare_parameter("config_file", "config/audio_config.yaml")
        self.declare_parameter("target_sample_rate", 16000)
        self.declare_parameter("channels", 1)
        self.declare_parameter("chunk_duration_ms", 80)  # 80ms chunks for openWakeWord
        self.declare_parameter("buffer_duration", 5.0)  # 5 seconds circular buffer
        self.declare_parameter("device_name", "USB PnP Sound Device")
        self.declare_parameter("reconnect_interval", 2.0)  # seconds

        # Wake Word Parameters
        self.declare_parameter("wake_word", "hey_rover")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("cooldown_seconds", 2.0)
        self.declare_parameter("model_path", "")
        self.declare_parameter("enable_verbose_logging", False)

        # VAD Parameters
        self.declare_parameter("vad_threshold", 0.5)
        self.declare_parameter("min_speech_duration_ms", 250)
        self.declare_parameter("min_silence_duration_ms", 500)

        # Whisper Parameters
        self.declare_parameter("whisper_model_size", "tiny.en")
        self.declare_parameter("whisper_compute_type", "int8")
        self.declare_parameter("whisper_device", "cpu")
        self.declare_parameter("max_recording_duration", 15)

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

        # Wake Word Config
        self.wake_word = self.get_parameter("wake_word").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.cooldown_seconds = self.get_parameter("cooldown_seconds").value
        self.model_path = self.get_parameter("model_path").value
        self.verbose_logging = self.get_parameter("enable_verbose_logging").value

        # VAD Config
        self.vad_threshold = self.get_parameter("vad_threshold").value
        self.min_speech_duration_ms = self.get_parameter("min_speech_duration_ms").value
        self.min_silence_duration_ms = self.get_parameter("min_silence_duration_ms").value

        # Whisper Config
        self.whisper_model_size = self.get_parameter("whisper_model_size").value
        self.whisper_compute_type = self.get_parameter("whisper_compute_type").value
        self.whisper_device = self.get_parameter("whisper_device").value
        self.max_recording_duration = self.get_parameter("max_recording_duration").value

        # Pipeline State Machine
        self.state = PipelineState.IDLE
        self.state_lock = threading.Lock()

        # Wake Word State
        self.last_detection_time = 0.0
        self.is_in_cooldown = False
        self.wake_word_model = None

        # VAD State
        self.vad_model = None
        self.vad_iterator = None
        self.speech_start_time = None
        self.recording_start_time = None

        # Whisper State
        self.whisper_model = None
        self.transcription_thread: Optional[threading.Thread] = None

        # Recording buffer (for VAD-captured speech)
        self.recording_buffer = []

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

        # ROS2 Publishers (NO audio streaming, only control messages)
        if AudioEvent:
            self.event_pub = self.create_publisher(AudioEvent, "/audio/events", 10)
        if TranscriptionResult:
            self.transcription_pub = self.create_publisher(
                TranscriptionResult, "/audio/transcription", 10
            )

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

        # Initialize Models
        self._initialize_wake_word_model()
        self._initialize_vad_model()
        self._initialize_whisper_model()

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
        self.get_logger().info(f"  Channels: {self.channels}")
        self.get_logger().info(f"  Chunk size: {self.chunk_size} samples ({chunk_duration_ms}ms)")
        self.get_logger().info(
            f"  Circular buffer: {buffer_duration}s ({self.buffer_max_chunks} chunks)"
        )

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

            # Extract pipeline configuration for wake word
            if "pipeline" in config and "wake_word" in config["pipeline"]:
                ww_config = config["pipeline"]["wake_word"]

                # List of parameters to update from config
                params_to_update = [
                    ("chunk_duration_ms", rclpy.Parameter.Type.INTEGER),
                    (
                        "wake_word",
                        rclpy.Parameter.Type.STRING,
                        "wake_word_name",
                    ),  # Map config name to param name
                    ("confidence_threshold", rclpy.Parameter.Type.DOUBLE),
                    ("cooldown_seconds", rclpy.Parameter.Type.DOUBLE),
                    ("model_path", rclpy.Parameter.Type.STRING),
                    ("enable_verbose_logging", rclpy.Parameter.Type.BOOL),
                ]

                new_params = []
                for param_def in params_to_update:
                    param_name = param_def[0]
                    param_type = param_def[1]
                    config_key = param_def[2] if len(param_def) > 2 else param_name

                    if config_key in ww_config:
                        new_params.append(
                            rclpy.parameter.Parameter(
                                param_name,
                                param_type,
                                ww_config[config_key],
                            )
                        )

                if new_params:
                    self.set_parameters(new_params)

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
        """Process incoming audio data based on current state."""
        try:
            # Normalize audio (Dynamic Gain Control)
            audio_normalized = self._normalize_audio(audio_chunk)

            # Add to circular buffer (for pre-roll)
            with self.buffer_lock:
                self.audio_buffer.append(audio_normalized.copy())

            # NO AUDIO PUBLISHING - process locally based on state
            self.frames_captured += 1

            # State machine processing
            with self.state_lock:
                current_state = self.state

            if current_state == PipelineState.IDLE:
                # Only run wake word detection in IDLE state
                self._detect_wake_word(audio_normalized)

            elif current_state == PipelineState.WAKE_WORD_DETECTED:
                # Transition to RECORDING and start VAD
                with self.state_lock:
                    self.state = PipelineState.RECORDING
                    self.speech_start_time = None
                self.get_logger().info("Starting VAD...")

            elif current_state == PipelineState.RECORDING:
                # Run VAD and accumulate audio
                self._process_vad(audio_normalized)

            elif current_state == PipelineState.TRANSCRIBING:
                # Transcription running in background, skip processing
                pass

        except Exception as e:
            self.get_logger().error(f"Error in audio processing: {e}")

    def _initialize_wake_word_model(self):
        """Initialize openWakeWord model."""
        if not OPENWAKEWORD_AVAILABLE:
            self.get_logger().error("openWakeWord library not found. Wake word detection disabled.")
            return

        self.get_logger().info("Initializing openWakeWord model...")
        try:
            if self.model_path and os.path.exists(self.model_path):
                if self.model_path.endswith(".onnx"):
                    self.wake_word_model = WakeWordModel(
                        wakeword_models=[self.model_path], inference_framework="onnx"
                    )
                    self.get_logger().info(f"Loaded custom ONNX model: {self.model_path}")
                else:
                    self.wake_word_model = WakeWordModel(wakeword_models=[self.model_path])
                    self.get_logger().info(f"Loaded custom model: {self.model_path}")
            else:
                self.wake_word_model = WakeWordModel()
                self.get_logger().info("Loaded default openWakeWord models")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize wake word model: {e}")
            self.wake_word_model = None

    def _initialize_vad_model(self):
        """Initialize Silero VAD model."""
        if not SILERO_VAD_AVAILABLE:
            self.get_logger().warn("Silero VAD library not found. VAD disabled.")
            return

        self.get_logger().info("Initializing Silero VAD model...")
        try:
            self.vad_model = load_silero_vad(onnx=True)
            self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.target_sample_rate)
            self.get_logger().info("Silero VAD model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize VAD model: {e}")
            self.vad_model = None
            self.vad_iterator = None

    def _initialize_whisper_model(self):
        """Initialize faster-whisper model."""
        if not WHISPER_AVAILABLE:
            self.get_logger().warn("faster-whisper library not found. Transcription disabled.")
            return

        self.get_logger().info(f"Initializing Whisper model ({self.whisper_model_size})...")
        try:
            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            self.get_logger().info("Whisper model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Whisper model: {e}")
            self.whisper_model = None

    def _detect_wake_word(self, audio_chunk: np.ndarray):
        """Run wake word detection on audio chunk (only in IDLE state)."""
        if self.wake_word_model is None:
            return

        # Only detect wake word in IDLE state
        with self.state_lock:
            if self.state != PipelineState.IDLE:
                return

        try:
            # Measure inference time
            start_time = time.time()

            # Predict
            prediction = self.wake_word_model.predict(audio_chunk)

            # Check inference time
            inference_time = (time.time() - start_time) * 1000
            if inference_time > 80:  # Warn if slower than real-time (80ms chunk)
                self.get_logger().warn(f"Wake word inference slow: {inference_time:.1f}ms")

            # Check for wake word
            if self.wake_word in prediction:
                confidence = prediction[self.wake_word]

                current_time = time.time()

                if confidence >= self.confidence_threshold:
                    # Check cooldown
                    if not self.is_in_cooldown:
                        self.get_logger().info(f"🎤 WAKE WORD DETECTED! ({confidence:.3f})")

                        # Publish event
                        if AudioEvent:
                            event = AudioEvent()
                            event.header.stamp = self.get_clock().now().to_msg()
                            event.event_type = "wake_word_detected"
                            event.confidence = float(confidence)
                            event.data = f"Wake word: {self.wake_word}"
                            self.event_pub.publish(event)

                        # Transition to WAKE_WORD_DETECTED state
                        with self.state_lock:
                            self.state = PipelineState.WAKE_WORD_DETECTED
                            self.recording_start_time = current_time
                            self.recording_buffer = []

                        self.last_detection_time = current_time
                        self.is_in_cooldown = True

                # Reset cooldown if time passed
                if self.is_in_cooldown and (
                    current_time - self.last_detection_time > self.cooldown_seconds
                ):
                    self.is_in_cooldown = False

        except Exception as e:
            self.get_logger().error(f"Error in wake word detection: {e}")

    def _process_vad(self, audio_chunk: np.ndarray):
        """Process audio with VAD to detect speech start/end."""
        if self.vad_model is None or self.vad_iterator is None:
            self.get_logger().warn("VAD not available, skipping")
            return

        try:
            # Accumulate audio first
            self.recording_buffer.append(audio_chunk.copy())

            # Silero VAD expects chunks of 512 samples for 16kHz
            # Our chunks are 1280 samples (80ms), so we need to split them
            VAD_CHUNK_SIZE = 512

            # Convert to float32 for VAD
            audio_float32 = audio_chunk.astype(np.float32) / 32768.0

            # Process in 512-sample chunks
            for i in range(0, len(audio_float32), VAD_CHUNK_SIZE):
                chunk_segment = audio_float32[i : i + VAD_CHUNK_SIZE]

                # Skip if segment is too small
                if len(chunk_segment) < VAD_CHUNK_SIZE:
                    continue

                audio_tensor = torch.tensor(chunk_segment)

                # Run VAD on this segment
                speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)

                # Check for speech events
                if speech_dict:
                    if "start" in speech_dict and self.speech_start_time is None:
                        self.speech_start_time = time.time()
                        self.get_logger().info("🎤 Speech started")

                        # Publish event
                        if AudioEvent:
                            event = AudioEvent()
                            event.header.stamp = self.get_clock().now().to_msg()
                            event.event_type = "speech_started"
                            self.event_pub.publish(event)

                    elif "end" in speech_dict and self.speech_start_time is not None:
                        self.get_logger().info("🎤 Speech ended")

                        # Publish event
                        if AudioEvent:
                            event = AudioEvent()
                            event.header.stamp = self.get_clock().now().to_msg()
                            event.event_type = "speech_ended"
                            duration = time.time() - self.speech_start_time
                            event.duration = float(duration)
                            self.event_pub.publish(event)

                        # Transition to TRANSCRIBING state
                        with self.state_lock:
                            self.state = PipelineState.TRANSCRIBING

                        # Start transcription in background thread
                        self._start_transcription()
                        return  # Exit early since we're transitioning

            # Check for timeout
            if self.recording_start_time is not None:
                recording_duration = time.time() - self.recording_start_time
                if recording_duration > self.max_recording_duration:
                    self.get_logger().warn(f"Recording timeout ({self.max_recording_duration}s)")

                    # Force end recording
                    with self.state_lock:
                        self.state = PipelineState.TRANSCRIBING
                    self._start_transcription()

        except Exception as e:
            self.get_logger().error(f"Error in VAD processing: {e}")

    def _start_transcription(self):
        """Start transcription in background thread."""
        if self.whisper_model is None:
            self.get_logger().error("Whisper model not available")
            with self.state_lock:
                self.state = PipelineState.IDLE
            return

        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcribe_audio)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()

    def _transcribe_audio(self):
        """Transcribe recorded audio using Whisper (runs in background thread)."""
        try:
            self.get_logger().info("Starting transcription...")

            # Concatenate recording buffer
            if len(self.recording_buffer) == 0:
                self.get_logger().warn("No audio to transcribe")
                with self.state_lock:
                    self.state = PipelineState.IDLE
                return

            audio_data = np.concatenate(self.recording_buffer)
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Run Whisper transcription
            start_time = time.time()
            segments, info = self.whisper_model.transcribe(
                audio_float,
                beam_size=5,
                language="en",
                vad_filter=True,
            )

            transcription = " ".join([segment.text.strip() for segment in segments])
            inference_time = time.time() - start_time

            self.get_logger().info(f"Transcription: '{transcription}' ({inference_time:.2f}s)")

            # Publish transcription result
            if TranscriptionResult:
                result = TranscriptionResult()
                result.header.stamp = self.get_clock().now().to_msg()
                result.text = transcription
                result.confidence = 1.0  # Whisper doesn't provide confidence
                result.duration = float(len(audio_data) / self.target_sample_rate)
                result.language = info.language if hasattr(info, "language") else "en"
                self.transcription_pub.publish(result)

            # Publish event
            if AudioEvent:
                event = AudioEvent()
                event.header.stamp = self.get_clock().now().to_msg()
                event.event_type = "asr_complete"
                event.data = transcription
                event.duration = float(inference_time)
                self.event_pub.publish(event)

            # Return to IDLE state
            with self.state_lock:
                self.state = PipelineState.IDLE
                self.recording_buffer = []
                self.speech_start_time = None
                self.recording_start_time = None

            self.get_logger().info("Returned to IDLE state")

        except Exception as e:
            self.get_logger().error(f"Error in transcription: {e}")
            with self.state_lock:
                self.state = PipelineState.IDLE

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
