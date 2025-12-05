#!/usr/bin/env python3
"""
Audio Model Setup and Testing Script

This script tests and benchmarks the audio models for the Local AI Robot Assistant:
1. openWakeWord for wake word detection
2. faster-whisper for speech recognition

Features:
- Wake word detection accuracy testing
- Speech recognition performance benchmarking
- CPU/memory usage monitoring
- Real-time performance validation
- Audio file testing support

Usage:
    python scripts/test_audio_models.py [--test-wake-word] [--test-whisper] [--benchmark]
    python scripts/test_audio_models.py --test-audio-files
"""

import argparse
import logging
import os
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf
import yaml
from faster_whisper import WhisperModel
from openwakeword import Model as WakeWordModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets" / "audio"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioConfig:
    """Load and manage audio configuration."""

    def __init__(self, config_file: str = "audio_config.yaml"):
        """Initialize audio configuration.

        Args:
            config_file: Name of the config file in the config directory
        """
        config_path = CONFIG_DIR / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Audio config not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    @property
    def microphone_device(self) -> str:
        """Get the microphone device string."""
        return self.config["microphone"]["device"]

    @property
    def speech_sample_rate(self) -> int:
        """Get sample rate for speech recognition."""
        return self.config["microphone"]["speech_recognition"]["sample_rate"]

    @property
    def speech_channels(self) -> int:
        """Get number of channels for speech recognition."""
        return self.config["microphone"]["speech_recognition"]["channels"]

    @property
    def wake_word_sample_rate(self) -> int:
        """Get sample rate for wake word detection."""
        return self.config["pipeline"]["wake_word"]["sample_rate"]

    @property
    def wake_word_channels(self) -> int:
        """Get number of channels for wake word detection."""
        return self.config["pipeline"]["wake_word"]["channels"]

    @property
    def wake_word_chunk_duration_ms(self) -> int:
        """Get chunk duration in ms for wake word detection."""
        return self.config["pipeline"]["wake_word"]["chunk_duration_ms"]

    @property
    def microphone_chunk_size(self) -> int:
        """Get microphone chunk size."""
        return self.config["microphone"]["chunk_size"]

    @property
    def microphone_buffer_size(self) -> int:
        """Get microphone buffer size."""
        return self.config["microphone"]["buffer_size"]

    @property
    def silence_threshold_db(self) -> float:
        """Get silence threshold in dB."""
        return self.config["pipeline"]["speech_to_text"]["silence_threshold_db"]

    @property
    def max_recording_duration(self) -> int:
        """Get maximum recording duration in seconds."""
        return self.config["pipeline"]["speech_to_text"]["max_recording_duration"]


class AudioModelTester:
    """Test and benchmark audio models for the robot assistant."""

    def __init__(self, config_file: str = "audio_config.yaml"):
        """Initialize the audio model tester.

        Args:
            config_file: Audio configuration file name
        """
        self.audio_config = AudioConfig(config_file)
        self.wake_word_model = None
        self.whisper_model = None
        self.audio_buffer = []

        # Use configuration values
        self.sample_rate = self.audio_config.speech_sample_rate
        self.wake_word_sample_rate = self.audio_config.wake_word_sample_rate
        self.wake_word_chunk_ms = self.audio_config.wake_word_chunk_duration_ms
        self.chunk_size = int(self.wake_word_sample_rate * self.wake_word_chunk_ms / 1000)
        self.microphone_device = self.audio_config.microphone_device

        logger.info("Audio configuration loaded:")
        logger.info(f"  Microphone device: {self.microphone_device}")
        logger.info(f"  Speech sample rate: {self.sample_rate} Hz")
        logger.info(f"  Wake word sample rate: {self.wake_word_sample_rate} Hz")
        logger.info(
            f"  Wake word chunk size: {self.chunk_size} samples ({self.wake_word_chunk_ms}ms)"
        )

    def _get_audio_device_info(self) -> Tuple[Optional[int], int]:
        """Get the audio device index and best sample rate for the configured microphone.

        Returns:
            Tuple of (device_index, sample_rate) where sample_rate is either:
            - The target rate (16000Hz) if supported by the device/ALSA
            - The hardware native rate (44100Hz) for manual resampling
        """
        try:
            devices = sd.query_devices()
            target_rate = self.wake_word_sample_rate  # 16000Hz for speech

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0 and "USB PnP Sound Device" in device["name"]:
                    logger.info(f"Found microphone device: {device['name']} (index: {i})")

                    # Try target rate first (16000Hz)
                    # Note: sounddevice accesses hw device directly, doesn't use ALSA's plughw
                    # So we need to use hardware-supported rates and resample manually
                    try:
                        sd.check_input_settings(device=i, samplerate=target_rate, channels=1)
                        logger.info(f"Device supports target rate {target_rate}Hz directly")
                        return i, target_rate
                    except sd.PortAudioError:
                        # Device doesn't support 16000Hz natively
                        # Use 44100Hz (common USB mic rate) and we'll resample manually
                        logger.info(f"Device doesn't support {target_rate}Hz natively")
                        logger.info(
                            f"Using hardware rate 44100Hz with manual resampling to {target_rate}Hz"
                        )
                        return i, 44100

            logger.warning("USB PnP Sound Device not found, using default input device")
            return None, 44100

        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
            return None, 44100

    def _resample_audio_fast(
        self, audio_data: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Fast resample audio data using scipy or fallback to decimation.

        Args:
            audio_data: Input audio data (float32, range -1 to 1)
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio data
        """
        if source_rate == target_rate:
            return audio_data

        try:
            # Calculate number of output samples
            _ = int(len(audio_data) * target_rate / source_rate)

            # Use resample_poly for efficient, high-quality resampling
            # This is much better than simple decimation
            # Find GCD to optimize resampling
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(source_rate, target_rate)
            up = target_rate // g
            down = source_rate // g

            logger.debug(f"Resampling {source_rate}Hz -> {target_rate}Hz (up={up}, down={down})")

            resampled = resample_poly(audio_data, up, down)

            return resampled.astype(np.float32)

        except ImportError:
            logger.warning("scipy not available, using simple decimation (lower quality)")
            # Fallback to simple decimation
            step = source_rate / target_rate
            indices = np.arange(0, len(audio_data), step)
            indices = indices[indices < len(audio_data)].astype(int)
            return audio_data[indices]

    def load_audio_file(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Try with soundfile first
            audio_data, sample_rate = sf.read(file_path, dtype="float32")

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                logger.info("Converting stereo to mono...")
                audio_data = np.mean(audio_data, axis=1)

            # Validate audio data
            if len(audio_data) == 0:
                raise ValueError("Audio file is empty")

            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-6:
                logger.warning(f"⚠️  Audio appears silent (RMS={rms:.6f})")

            logger.info(f"Loaded audio file: {file_path}")
            logger.info(f"  Sample rate: {sample_rate} Hz")
            logger.info(f"  Duration: {len(audio_data)/sample_rate:.2f}s")
            logger.info(f"  Samples: {len(audio_data)}")
            logger.info(f"  RMS level: {rms:.6f}")
            logger.info(f"  Range: [{audio_data.min():.6f}, {audio_data.max():.6f}]")

            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            # Try with wave module as fallback
            try:
                with wave.open(str(file_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    sample_width = wf.getsampwidth()
                    audio_bytes = wf.readframes(n_frames)

                    logger.info(f"Wave file info: {n_channels}ch, {sample_width}B, {sample_rate}Hz")

                    # Convert bytes to numpy array based on sample width
                    if sample_width == 2:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                        # Convert to float32 [-1, 1]
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sample_width}")

                    # Convert stereo to mono if needed
                    if n_channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                    rms = np.sqrt(np.mean(audio_data**2))

                    logger.info(f"Loaded audio file (wave): {file_path}")
                    logger.info(f"  Sample rate: {sample_rate} Hz")
                    logger.info(f"  Duration: {len(audio_data)/sample_rate:.2f}s")
                    logger.info(f"  RMS level: {rms:.6f}")

                    return audio_data, sample_rate
            except Exception as e2:
                logger.error(f"Failed to load with wave module: {e2}")
                raise

    def test_wake_word_from_file(self, audio_file: Path) -> Dict:
        """Test wake word detection on an audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Dict with test results
        """
        if not self.wake_word_model:
            logger.error("Wake word model not loaded")
            return {}

        logger.info(f"Testing wake word detection on file: {audio_file}")

        # Load audio file
        audio_data, file_sample_rate = self.load_audio_file(audio_file)

        # Resample if needed
        if file_sample_rate != self.wake_word_sample_rate:
            logger.info(f"Resampling from {file_sample_rate}Hz to {self.wake_word_sample_rate}Hz")
            audio_data = self._resample_audio_fast(
                audio_data, file_sample_rate, self.wake_word_sample_rate
            )

        # Convert to int16 format expected by openWakeWord
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Process in chunks
        chunk_samples = int(self.wake_word_sample_rate * self.wake_word_chunk_ms / 1000)
        detections = []
        scores = []

        start_time = time.perf_counter()

        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]

            # Skip if chunk is too small
            if len(chunk) < 160:
                continue

            # Get predictions
            predictions = self.wake_word_model.predict(chunk)

            # Check for detections
            for model_name, score in predictions.items():
                scores.append(score)
                if score > 0.5:
                    timestamp = i / self.wake_word_sample_rate
                    detections.append({"time": timestamp, "score": score, "model": model_name})
                    logger.info(f"Wake word detected at {timestamp:.2f}s with score {score:.3f}")

        inference_time = time.perf_counter() - start_time

        results = {
            "file": str(audio_file),
            "detections": len(detections),
            "detection_details": detections,
            "max_score": max(scores) if scores else 0.0,
            "avg_score": np.mean(scores) if scores else 0.0,
            "inference_time_ms": inference_time * 1000,
            "audio_duration_s": len(audio_data) / self.wake_word_sample_rate,
        }

        logger.info("Wake word test results:")
        logger.info(f"  Detections: {results['detections']}")
        logger.info(f"  Max score: {results['max_score']:.3f}")
        logger.info(f"  Avg score: {results['avg_score']:.3f}")
        logger.info(f"  Processing time: {results['inference_time_ms']:.1f}ms")

        return results

    def test_speech_recognition_from_file(
        self, audio_file: Path, expected_text: Optional[str] = None
    ) -> Dict:
        """Test speech recognition on an audio file.

        Args:
            audio_file: Path to audio file
            expected_text: Expected transcription (optional)

        Returns:
            Dict with test results
        """
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return {}

        logger.info(f"Testing speech recognition on file: {audio_file}")

        # For faster-whisper, we can pass the file path directly!
        # It handles all the audio loading internally
        logger.info(f"Passing file path directly to faster-whisper: {audio_file}")

        # Transcribe - pass file path as string
        start_time = time.perf_counter()

        # Collect all segments with details
        segments_list = []
        segments, info = self.whisper_model.transcribe(
            str(audio_file),  # Pass file path as string!
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # IMPORTANT: segments is a generator, must consume it
        for segment in segments:
            segments_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            logger.info(f"  Segment [{segment.start:.2f}s - {segment.end:.2f}s]: '{segment.text}'")

        # Collect transcription
        transcription = " ".join([seg["text"].strip() for seg in segments_list])
        inference_time = time.perf_counter() - start_time

        # Get audio duration from info or load file to calculate
        try:
            audio_data, sample_rate = self.load_audio_file(audio_file)
            audio_duration = len(audio_data) / sample_rate
        except Exception:
            # Fallback if we can't load the file
            audio_duration = info.duration if hasattr(info, "duration") else 1.0

        # Calculate real-time factor
        rtf = inference_time / audio_duration

        results = {
            "file": str(audio_file),
            "transcription": transcription,
            "expected": expected_text,
            "inference_time_ms": inference_time * 1000,
            "audio_duration_s": audio_duration,
            "real_time_factor": rtf,
            "language": info.language if hasattr(info, "language") else "unknown",
            "language_probability": (
                info.language_probability if hasattr(info, "language_probability") else 0.0
            ),
            "num_segments": len(segments_list),
            "segments": segments_list,
        }

        logger.info("Speech recognition results:")
        logger.info(
            f"  Detected language: {results['language']} (probability: \
                {results['language_probability']:.2f})"
        )
        logger.info(f"  Transcription: '{transcription}'")
        logger.info(f"  Number of segments: {len(segments_list)}")
        if expected_text:
            logger.info(f"  Expected: '{expected_text}'")
            # Simple word-level accuracy
            trans_words = set(transcription.lower().split())
            expected_words = set(expected_text.lower().split())

            # Calculate overlap
            if len(expected_words) > 0:
                matches = len(trans_words & expected_words)
                accuracy = matches / len(expected_words) * 100
                results["accuracy_percent"] = accuracy
                logger.info(f"  Word overlap: {matches}/{len(expected_words)} words")
                logger.info(f"  Accuracy: {accuracy:.1f}%")

            # Also check sequence accuracy
            trans_lower = transcription.lower()
            expected_lower = expected_text.lower()
            if trans_lower == expected_lower:
                logger.info("  ✅ Exact match!")
            elif trans_lower in expected_lower or expected_lower in trans_lower:
                logger.info("  ⚠️  Partial match")
            else:
                logger.info("  ❌ Different content")

        logger.info(f"  Processing time: {results['inference_time_ms']:.1f}ms")
        logger.info(f"  Real-time factor: {rtf:.2f}x")

        return results

    def test_audio_files(self) -> Dict:
        """Test models on pre-recorded audio files.

        Returns:
            Dict with all test results
        """
        logger.info("\n=== Testing with Audio Files ===")

        results = {"wake_word_tests": [], "speech_recognition_tests": []}

        # Test wake word on HeyRover.wav
        hey_rover_file = ASSETS_DIR / "HeyRover.wav"
        if hey_rover_file.exists() and self.wake_word_model:
            logger.info("\n--- Testing Wake Word Detection ---")
            ww_result = self.test_wake_word_from_file(hey_rover_file)
            results["wake_word_tests"].append(ww_result)

            # Validate
            if ww_result.get("detections", 0) > 0:
                logger.info("✅ Wake word detected successfully!")
            else:
                logger.warning("❌ Wake word NOT detected (expected at least 1 detection)")
        else:
            if not hey_rover_file.exists():
                logger.warning(f"Wake word test file not found: {hey_rover_file}")
            if not self.wake_word_model:
                logger.warning("Wake word model not loaded")

        # Test speech recognition on TheRainInSpain.wav
        rain_file = ASSETS_DIR / "TheRainInSpain.wav"
        expected_text = "The rain in Spain stays mainly in the plane."

        if rain_file.exists() and self.whisper_model:
            logger.info("\n--- Testing Speech Recognition ---")
            sr_result = self.test_speech_recognition_from_file(rain_file, expected_text)
            results["speech_recognition_tests"].append(sr_result)

            # Validate
            if sr_result.get("accuracy_percent", 0) > 70:
                logger.info("✅ Speech recognition accuracy good!")
            else:
                logger.warning(
                    f"⚠️  Speech recognition accuracy low: \
                        {sr_result.get('accuracy_percent', 0):.1f}%"
                )
        else:
            if not rain_file.exists():
                logger.warning(f"Speech recognition test file not found: {rain_file}")
            if not self.whisper_model:
                logger.warning("Whisper model not loaded")

        return results

    def setup_wake_word_model(self) -> bool:
        """Set up the openWakeWord model.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            model_path = MODEL_DIR / "wake_word" / "hey_roe_ver.onnx"

            logger.info("Loading openWakeWord model...")
            start_time = time.time()

            if model_path.exists():
                logger.info(f"Using custom wake word model: {model_path}")
                self.wake_word_model = WakeWordModel(
                    wakeword_models=[str(model_path)], inference_framework="onnx"
                )
            else:
                logger.info("Using default 'hey rover' model")
                self.wake_word_model = WakeWordModel(
                    wakeword_models=["hey_roe_ver"], inference_framework="onnx"
                )

            load_time = time.time() - start_time
            logger.info(f"Wake word model loaded in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            return False

    def setup_whisper_model(self) -> bool:
        """Set up the faster-whisper model.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Loading faster-whisper model...")
            start_time = time.time()

            self.whisper_model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=1,
            )

            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            return False

    def test_wake_word_detection(self, test_duration: float = 10.0) -> Dict:
        """Test wake word detection performance.

        Args:
            test_duration: Duration to test in seconds

        Returns:
            Dict: Test results including CPU usage, latency, etc.
        """
        if not self.wake_word_model:
            logger.error("Wake word model not loaded")
            return {}

        logger.info(f"Testing wake word detection for {test_duration}s...")
        logger.info("Say 'Hey Rover' to test detection")

        mic_device, hardware_rate = self._get_audio_device_info()

        results = {
            "detections": 0,
            "avg_cpu_usage": 0.0,
            "avg_inference_time": 0.0,
            "max_cpu_usage": 0.0,
            "samples_processed": 0,
            "device_used": mic_device,
            "hardware_sample_rate": hardware_rate,
            "target_sample_rate": self.wake_word_sample_rate,
        }

        cpu_readings = []
        inference_times = []

        def audio_callback(indata, frames, time_info, status):
            """Process audio chunks for wake word detection."""
            try:
                if status:
                    logger.debug(f"Audio status: {status}")

                if results["samples_processed"] % 10 == 0:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    cpu_readings.append(cpu_percent)

                start_inference = time.perf_counter()

                if hardware_rate != self.wake_word_sample_rate:
                    audio_chunk = self._resample_audio_fast(
                        indata[:, 0], hardware_rate, self.wake_word_sample_rate
                    )
                else:
                    audio_chunk = indata[:, 0]

                audio_data = (audio_chunk * 32767).astype(np.int16)

                if len(audio_data) >= 160:
                    predictions = self.wake_word_model.predict(audio_data)

                    inference_time = time.perf_counter() - start_inference
                    inference_times.append(inference_time * 1000)

                    for model_name, score in predictions.items():
                        if score > 0.5:
                            results["detections"] += 1
                            logger.info(f"Wake word detected! Score: {score:.3f}")

                results["samples_processed"] += 1

            except Exception as e:
                logger.error(f"Callback error: {e}")

        try:
            hardware_chunk_size = int(hardware_rate * self.wake_word_chunk_ms / 1000)

            stream_kwargs = {
                "callback": audio_callback,
                "channels": self.audio_config.wake_word_channels,
                "samplerate": hardware_rate,
                "blocksize": hardware_chunk_size,
                "dtype": np.float32,
                "latency": "low",
            }

            if mic_device is not None:
                stream_kwargs["device"] = (mic_device, None)

            logger.info(
                f"Starting audio stream: {hardware_rate}Hz, {hardware_chunk_size} samples/chunk"
            )
            logger.info("🎤 Recording active - say 'Hey Rover' now!")

            with sd.InputStream(**stream_kwargs):
                time.sleep(0.1)
                time.sleep(test_duration)

        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            return {}

        if cpu_readings:
            results["avg_cpu_usage"] = np.mean(cpu_readings)
            results["max_cpu_usage"] = np.max(cpu_readings)

        if inference_times:
            results["avg_inference_time"] = np.mean(inference_times)
            results["max_inference_time"] = np.max(inference_times)

        results["test_duration"] = test_duration

        return results

    def test_speech_recognition(self, test_phrases: Optional[List[str]] = None) -> Dict:
        """Test speech recognition performance.

        Args:
            test_phrases: Optional list of phrases to test

        Returns:
            Dict: Test results including recognition accuracy and timing
        """
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return {}

        if test_phrases is None:
            test_phrases = [
                "Hello robot, how are you?",
                "Move forward slowly",
                "Turn left and stop",
            ]

        logger.info("Testing speech recognition...")
        mic_device, hardware_rate = self._get_audio_device_info()

        results = {
            "tests_completed": 0,
            "transcriptions": [],
            "avg_inference_time": 0.0,
            "avg_cpu_usage": 0.0,
            "memory_usage_mb": 0.0,
            "device_used": mic_device,
            "hardware_sample_rate": hardware_rate,
            "target_sample_rate": self.sample_rate,
        }

        inference_times = []
        cpu_readings = []

        for i, phrase in enumerate(test_phrases):
            logger.info(f"\nTest {i+1}/{len(test_phrases)}")
            logger.info(f"Please say: '{phrase}'")
            input("Press Enter when ready to record...")

            max_duration = self.audio_config.max_recording_duration
            logger.info(f"Recording... ({max_duration} seconds)")

            start_cpu = psutil.cpu_percent()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024

            recording_kwargs = {
                "frames": int(max_duration * hardware_rate),
                "samplerate": hardware_rate,
                "channels": self.audio_config.speech_channels,
                "dtype": np.float32,
            }

            if mic_device is not None:
                recording_kwargs["device"] = (mic_device, None)

            recording = sd.rec(**recording_kwargs)
            sd.wait()

            # Resample if needed
            if hardware_rate != self.sample_rate:
                audio_data = self._resample_audio_fast(
                    recording[:, 0], hardware_rate, self.sample_rate
                )
            else:
                audio_data = recording[:, 0]

            # For numpy arrays, faster-whisper expects float32 in [-1, 1]
            # NOT int16! Keep as float32
            logger.info(
                f"Audio data: dtype={audio_data.dtype}, \
                    range=[{audio_data.min():.4f}, {audio_data.max():.4f}]"
            )

            # Transcribe with float32 array
            start_inference = time.perf_counter()
            segments, info = self.whisper_model.transcribe(
                audio_data,  # Pass float32 array for live audio
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Consume the generator
            transcription = " ".join([segment.text.strip() for segment in segments])
            inference_time = time.perf_counter() - start_inference
            inference_times.append(inference_time * 1000)

            end_cpu = psutil.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024

            cpu_readings.append((start_cpu + end_cpu) / 2)
            memory_usage = end_memory - start_memory

            results["transcriptions"].append(
                {
                    "expected": phrase,
                    "transcribed": transcription,
                    "inference_time_ms": inference_time * 1000,
                    "memory_delta_mb": memory_usage,
                }
            )

            logger.info(f"Expected: {phrase}")
            logger.info(f"Got: {transcription}")
            logger.info(f"Inference time: {inference_time*1000:.1f}ms")

            results["tests_completed"] += 1

        if inference_times:
            results["avg_inference_time"] = np.mean(inference_times)
            results["max_inference_time"] = np.max(inference_times)

        if cpu_readings:
            results["avg_cpu_usage"] = np.mean(cpu_readings)

        results["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024

        return results

    def benchmark_models(self) -> Dict:
        """Run comprehensive benchmarks on both models.

        Returns:
            Dict: Comprehensive benchmark results
        """
        logger.info("Running comprehensive audio model benchmarks...")

        benchmark_results = {
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "platform": os.uname().machine,
            },
            "wake_word": {},
            "whisper": {},
        }

        if self.wake_word_model:
            logger.info("\n=== Wake Word Benchmark ===")
            ww_results = self.test_wake_word_detection(test_duration=5.0)
            benchmark_results["wake_word"] = ww_results

            target_cpu = 5.0
            if ww_results.get("avg_cpu_usage", 100) < target_cpu:
                logger.info(
                    f"✅ Wake word CPU usage: {ww_results.get('avg_cpu_usage', 0):.1f}% "
                    f"(target: <{target_cpu}%)"
                )
            else:
                logger.warning(
                    f"❌ Wake word CPU usage: {ww_results.get('avg_cpu_usage', 0):.1f}% "
                    f"(target: <{target_cpu}%)"
                )

        if self.whisper_model:
            logger.info("\n=== Whisper Benchmark ===")
            test_audio = np.sin(2 * np.pi * 440 * np.arange(0, 2, 1 / self.sample_rate))
            test_audio = (test_audio * 32767).astype(np.int16)

            start_time = time.perf_counter()
            segments, info = self.whisper_model.transcribe(test_audio, beam_size=1)
            inference_time = time.perf_counter() - start_time

            audio_duration = len(test_audio) / self.sample_rate
            rtf = inference_time / audio_duration

            benchmark_results["whisper"] = {
                "inference_time_ms": inference_time * 1000,
                "audio_duration_s": audio_duration,
                "real_time_factor": rtf,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            }

            target_rtf = 0.3
            target_memory = 300

            if rtf < target_rtf:
                logger.info(f"✅ Whisper real-time factor: {rtf:.2f}x (target: <{target_rtf}x)")
            else:
                logger.warning(f"❌ Whisper real-time factor: {rtf:.2f}x (target: <{target_rtf}x)")

            if benchmark_results["whisper"]["memory_usage_mb"] < target_memory:
                logger.info(
                    f"✅ Whisper memory usage: "
                    f"{benchmark_results['whisper']['memory_usage_mb']:.1f}MB "
                    f"(target: <{target_memory}MB)"
                )
            else:
                logger.warning(
                    f"❌ Whisper memory usage: "
                    f"{benchmark_results['whisper']['memory_usage_mb']:.1f}MB "
                    f"(target: <{target_memory}MB)"
                )

        return benchmark_results

    def validate_audio_setup(self) -> bool:
        """Validate that audio hardware is properly configured.

        Returns:
            bool: True if audio setup is valid, False otherwise
        """
        logger.info("Validating audio setup...")

        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    logger.info(
                        f"  Input {i}: {device['name']} ({device['max_input_channels']} channels)"
                    )

            mic_device, hardware_rate = self._get_audio_device_info()
            if mic_device is None:
                logger.warning("Configured microphone device not found, will use default")

            logger.info("Testing basic recording capability...")
            test_duration = 1

            recording_kwargs = {
                "frames": int(test_duration * hardware_rate),
                "samplerate": hardware_rate,
                "channels": self.audio_config.speech_channels,
                "dtype": np.float32,
            }

            if mic_device is not None:
                recording_kwargs["device"] = (mic_device, None)

            logger.info(
                f"Testing with: {hardware_rate}Hz, {self.audio_config.speech_channels} channel(s)"
            )

            test_recording = sd.rec(**recording_kwargs)
            sd.wait()

            if test_recording is not None and len(test_recording) > 0:
                avg_amplitude = np.mean(np.abs(test_recording))
                logger.info(f"✅ Recording test successful. Average amplitude: {avg_amplitude:.6f}")

                if avg_amplitude < 1e-6:
                    logger.warning(
                        "⚠️  Very low audio signal detected. Check microphone connection and volume"
                    )

                if hardware_rate != self.sample_rate:
                    logger.info(
                        f"Testing resampling from {hardware_rate}Hz to {self.sample_rate}Hz..."
                    )
                    resampled = self._resample_audio_fast(
                        test_recording[:, 0], hardware_rate, self.sample_rate
                    )
                    logger.info(
                        f"✅ Resampling successful: {len(resampled)} samples at {self.sample_rate}Hz"
                    )

                return True
            else:
                logger.error("❌ Recording test failed - no audio data captured")
                return False

        except Exception as e:
            logger.error(f"❌ Audio validation failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main function to run audio model tests."""
    parser = argparse.ArgumentParser(description="Test and benchmark audio models")
    parser.add_argument("--test-wake-word", action="store_true", help="Test wake word detection")
    parser.add_argument("--test-whisper", action="store_true", help="Test speech recognition")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmarks")
    parser.add_argument("--test-audio-files", action="store_true", help="Test with audio files")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--validate-audio", action="store_true", help="Validate audio setup only")
    parser.add_argument("--config", default="audio_config.yaml", help="Audio config file name")

    args = parser.parse_args()

    if not any(
        [
            args.test_wake_word,
            args.test_whisper,
            args.benchmark,
            args.test_audio_files,
            args.all,
            args.validate_audio,
        ]
    ):
        args.test_audio_files = True  # Default to testing audio files

    try:
        tester = AudioModelTester(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize audio tester: {e}")
        return 1

    # Always validate audio setup first (unless only testing files)
    if not args.test_audio_files or args.validate_audio:
        if not tester.validate_audio_setup():
            logger.error("Audio validation failed. Please check your audio configuration.")
            if not args.validate_audio and not args.test_audio_files:
                return 1

    if args.validate_audio:
        logger.info("✅ Audio validation completed successfully!")
        return 0

    # Setup models based on what tests we're running
    need_wake_word = args.test_wake_word or args.benchmark or args.all or args.test_audio_files
    need_whisper = args.test_whisper or args.benchmark or args.all or args.test_audio_files

    if need_wake_word:
        if not tester.setup_wake_word_model():
            logger.error("Failed to setup wake word model")
            return 1

    if need_whisper:
        if not tester.setup_whisper_model():
            logger.error("Failed to setup whisper model")
            return 1

    # Run tests
    try:
        if args.test_audio_files or args.all:
            logger.info("\n" + "=" * 60)
            logger.info("TESTING WITH AUDIO FILES")
            logger.info("=" * 60)
            file_results = tester.test_audio_files()

            # Print summary
            print("\n" + "=" * 60)
            print("AUDIO FILE TEST SUMMARY")
            print("=" * 60)

            if file_results.get("wake_word_tests"):
                print("\nWake Word Detection:")
                for test in file_results["wake_word_tests"]:
                    print(f"  File: {Path(test['file']).name}")
                    print(f"  Detections: {test['detections']}")
                    print(f"  Max Score: {test['max_score']:.3f}")
                    if test["detections"] > 0:
                        print("  Status: ✅ PASS")
                    else:
                        print("  Status: ❌ FAIL")

            if file_results.get("speech_recognition_tests"):
                print("\nSpeech Recognition:")
                for test in file_results["speech_recognition_tests"]:
                    print(f"  File: {Path(test['file']).name}")
                    print(f"  Transcription: '{test['transcription']}'")
                    print(f"  Expected: '{test['expected']}'")
                    if "accuracy_percent" in test:
                        print(f"  Accuracy: {test['accuracy_percent']:.1f}%")
                        if test["accuracy_percent"] > 70:
                            print("  Status: ✅ PASS")
                        else:
                            print("  Status: ⚠️  PARTIAL")
                    print(f"  RTF: {test['real_time_factor']:.2f}x")

        if args.benchmark:
            results = tester.benchmark_models()
            logger.info("\n=== Benchmark Results ===")

            if "wake_word" in results and results["wake_word"]:
                ww = results["wake_word"]
                print("Wake Word Detection:")
                print(f"  Average CPU: {ww.get('avg_cpu_usage', 0):.1f}%")
                print(f"  Average inference: {ww.get('avg_inference_time', 0):.1f}ms")
                print(f"  Detections: {ww.get('detections', 0)}")

            if "whisper" in results and results["whisper"]:
                whisper = results["whisper"]
                print("\nSpeech Recognition:")
                print(f"  Real-time factor: {whisper.get('real_time_factor', 0):.2f}x")
                print(f"  Memory usage: {whisper.get('memory_usage_mb', 0):.1f}MB")
                print(f"  Inference time: {whisper.get('inference_time_ms', 0):.1f}ms")

        elif args.test_wake_word:
            results = tester.test_wake_word_detection()
            logger.info(f"Wake word test completed: {results}")

        elif args.test_whisper:
            results = tester.test_speech_recognition()
            logger.info(f"Speech recognition test completed: {results}")

        logger.info("\n✅ Audio model testing completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("\nTesting interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
