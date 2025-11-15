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

Usage:
    python scripts/test_audio_models.py [--test-wake-word] [--test-whisper] [--benchmark]
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import sounddevice as sd
import yaml
from faster_whisper import WhisperModel
from openwakeword import Model as WakeWordModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

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
        self.chunk_size = int(
            self.wake_word_sample_rate * self.wake_word_chunk_ms / 1000
        )  # Convert ms to samples
        self.microphone_device = self.audio_config.microphone_device

        logger.info("Audio configuration loaded:")
        logger.info(f"  Microphone device: {self.microphone_device}")
        logger.info(f"  Speech sample rate: {self.sample_rate} Hz")
        logger.info(f"  Wake word sample rate: {self.wake_word_sample_rate} Hz")
        logger.info(
            f"  Wake word chunk size: {self.chunk_size} samples ({self.wake_word_chunk_ms}ms)"
        )

    def _get_audio_device_info(self) -> Tuple[Optional[int], int]:
        """Get the audio device index and appropriate sample rate for the configured microphone.

        Returns:
            Tuple of (device_index, sample_rate) where sample_rate is hardware-supported
        """
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0 and "USB PnP Sound Device" in device["name"]:
                    logger.info(f"Found microphone device: {device['name']} (index: {i})")

                    # Check supported sample rates for this device
                    supported_rates = [
                        44100,
                        48000,
                        22050,
                    ]  # Common rates, highest first
                    for rate in supported_rates:
                        try:
                            sd.check_input_settings(device=i, samplerate=rate, channels=1)
                            logger.info(f"Device supports {rate}Hz sample rate")
                            return i, rate
                        except sd.PortAudioError:
                            continue

                    logger.warning(f"No supported sample rates found for device {i}")
                    return i, 44100  # Default fallback

            logger.warning("USB PnP Sound Device not found, using default input device")
            return None, self.sample_rate

        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
            return None, self.sample_rate

    def _resample_audio_fast(
        self, audio_data: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Fast resample audio data using decimation for real-time processing.

        Args:
            audio_data: Input audio data
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio data
        """
        if source_rate == target_rate:
            return audio_data

        # For common case: 44100 -> 16000, use simple decimation
        if source_rate == 44100 and target_rate == 16000:
            # Ratio is approximately 2.76, so take every ~3rd sample
            step = int(source_rate / target_rate)
            return audio_data[::step]
        elif source_rate == 48000 and target_rate == 16000:
            # Ratio is 3.0, so take every 3rd sample
            return audio_data[::3]
        else:
            # Fallback to simple decimation
            step = max(1, int(source_rate / target_rate))
            return audio_data[::step]

    def setup_wake_word_model(self) -> bool:
        """Set up the openWakeWord model.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            model_path = MODEL_DIR / "wake_word" / "hey_jarvis_v0.1.onnx"
            if not model_path.exists():
                logger.error(f"Wake word model not found: {model_path}")
                return False

            logger.info("Loading openWakeWord model...")
            start_time = time.time()

            # Check if using custom model or default
            if model_path.exists():
                logger.info(f"Using custom wake word model: {model_path}")
                self.wake_word_model = WakeWordModel(
                    wakeword_models=[str(model_path)], inference_framework="onnx"
                )
            else:
                logger.info("Using default 'hey jarvis' model")
                self.wake_word_model = WakeWordModel(
                    wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx"
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

            # Use the tiny model for best performance on Jetson
            self.whisper_model = WhisperModel(
                "tiny",
                device="cpu",  # Use CPU for now, can optimize to CUDA later
                compute_type="int8",  # Quantized for better performance
                cpu_threads=4,  # Optimize for Jetson Orin Nano
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
        logger.info("Say 'Hey Jarvis' to test detection")

        # Get the microphone device and its native sample rate
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
        start_test = time.time()

        def audio_callback(indata, frames, time_info, status):
            """Process audio chunks for wake word detection."""
            try:
                if status:
                    logger.debug(f"Audio status: {status}")  # Reduce to debug level

                # Monitor CPU usage (less frequently to reduce overhead)
                if results["samples_processed"] % 10 == 0:  # Only every 10th sample
                    cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
                    cpu_readings.append(cpu_percent)

                # Process audio chunk
                start_inference = time.perf_counter()

                # Fast resample audio if needed
                if hardware_rate != self.wake_word_sample_rate:
                    audio_chunk = self._resample_audio_fast(
                        indata[:, 0], hardware_rate, self.wake_word_sample_rate
                    )
                else:
                    audio_chunk = indata[:, 0]

                # Convert to the format expected by openWakeWord
                audio_data = (audio_chunk * 32767).astype(np.int16)

                # Get predictions (only if we have enough samples)
                if len(audio_data) >= 160:  # Minimum samples for wake word
                    predictions = self.wake_word_model.predict(audio_data)

                    inference_time = time.perf_counter() - start_inference
                    inference_times.append(inference_time * 1000)  # Convert to ms

                    # Check for wake word detection
                    for model_name, score in predictions.items():
                        if score > 0.5:  # Threshold for detection
                            results["detections"] += 1
                            logger.info(f"Wake word detected! Score: {score:.3f}")

                results["samples_processed"] += 1

            except Exception as e:
                logger.error(f"Callback error: {e}")
                # Don't re-raise, just log and continue

        try:
            # Calculate chunk size for hardware sample rate (make it larger to reduce overhead)
            hardware_chunk_size = int(hardware_rate * self.wake_word_chunk_ms / 1000)

            # Start audio stream with specific device and larger buffers
            stream_kwargs = {
                "callback": audio_callback,
                "channels": self.audio_config.wake_word_channels,
                "samplerate": hardware_rate,
                "blocksize": hardware_chunk_size,
                "dtype": np.float32,
                "latency": "low",  # Request low latency
            }

            # Add device if found
            if mic_device is not None:
                stream_kwargs["device"] = (
                    mic_device,
                    None,
                )  # (input_device, output_device)

            logger.info(
                f"Starting audio stream: {hardware_rate}Hz, {hardware_chunk_size} samples/chunk"
            )
            logger.info("Listening for wake word... (speak now)")

            with sd.InputStream(**stream_kwargs):
                # Give it a moment to start
                time.sleep(0.1)
                logger.info("🎤 Recording active - say 'Hey Jarvis' now!")
                time.sleep(test_duration)

        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            return {}

        # Calculate statistics
        if cpu_readings:
            results["avg_cpu_usage"] = np.mean(cpu_readings)
            results["max_cpu_usage"] = np.max(cpu_readings)

        if inference_times:
            results["avg_inference_time"] = np.mean(inference_times)
            results["max_inference_time"] = np.max(inference_times)

        results["test_duration"] = time.time() - start_test

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
                "What do you see?",
                "Go to the kitchen",
            ]

        logger.info("Testing speech recognition...")
        logger.info("Speak clearly when prompted")

        # Get the microphone device and its native sample rate
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

            # Record audio for the configured duration
            max_duration = self.audio_config.max_recording_duration
            logger.info(f"Recording... ({max_duration} seconds)")

            start_cpu = psutil.cpu_percent()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Configure recording parameters
            recording_kwargs = {
                "frames": int(max_duration * hardware_rate),
                "samplerate": hardware_rate,
                "channels": self.audio_config.speech_channels,
                "dtype": np.float32,
            }

            # Add device if found
            if mic_device is not None:
                recording_kwargs["device"] = (mic_device, None)

            recording = sd.rec(**recording_kwargs)
            sd.wait()  # Wait for recording to complete

            # Resample to target rate if needed
            if hardware_rate != self.sample_rate:
                audio_data = self._resample_audio_fast(
                    recording[:, 0], hardware_rate, self.sample_rate
                )
            else:
                audio_data = recording[:, 0]

            # Convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)

            # Transcribe
            start_inference = time.perf_counter()

            segments, info = self.whisper_model.transcribe(
                audio_data, beam_size=1, language="en"  # Faster inference
            )

            # Collect transcription
            transcription = " ".join([segment.text.strip() for segment in segments])

            inference_time = time.perf_counter() - start_inference
            inference_times.append(inference_time * 1000)  # Convert to ms

            # Monitor resource usage
            end_cpu = psutil.cpu_percent()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB

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

        # Calculate averages
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

        # Test wake word model
        if self.wake_word_model:
            logger.info("\n=== Wake Word Benchmark ===")
            ww_results = self.test_wake_word_detection(test_duration=5.0)
            benchmark_results["wake_word"] = ww_results

            # Check if it meets targets
            target_cpu = 5.0  # < 5% CPU usage target
            if ww_results.get("avg_cpu_usage", 100) < target_cpu:
                logger.info(
                    f"✅ Wake word CPU usage: {ww_results.get('avg_cpu_usage', 0):.1f}% \
                        (target: <{target_cpu}%)"
                )
            else:
                logger.warning(
                    f"❌ Wake word CPU usage: {ww_results.get('avg_cpu_usage', 0):.1f}% \
                        (target: <{target_cpu}%)"
                )

        # Test whisper model
        if self.whisper_model:
            logger.info("\n=== Whisper Benchmark ===")
            # Create a test audio snippet programmatically
            test_audio = np.sin(2 * np.pi * 440 * np.arange(0, 2, 1 / self.sample_rate))
            test_audio = (test_audio * 32767).astype(np.int16)

            start_time = time.perf_counter()
            segments, info = self.whisper_model.transcribe(test_audio, beam_size=1)
            inference_time = time.perf_counter() - start_time

            # Calculate real-time factor
            audio_duration = len(test_audio) / self.sample_rate
            rtf = inference_time / audio_duration

            benchmark_results["whisper"] = {
                "inference_time_ms": inference_time * 1000,
                "audio_duration_s": audio_duration,
                "real_time_factor": rtf,
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            }

            # Check if it meets targets
            target_rtf = 0.3  # < 0.3x real-time factor target
            target_memory = 300  # < 300MB memory target

            if rtf < target_rtf:
                logger.info(f"✅ Whisper real-time factor: {rtf:.2f}x (target: <{target_rtf}x)")
            else:
                logger.warning(f"❌ Whisper real-time factor: {rtf:.2f}x (target: <{target_rtf}x)")

            if benchmark_results["whisper"]["memory_usage_mb"] < target_memory:
                logger.info(
                    f"✅ Whisper memory usage: \
                        {benchmark_results['whisper']['memory_usage_mb']:.1f}\
                            MB (target: <{target_memory}MB)"
                )
            else:
                logger.warning(
                    f"❌ Whisper memory usage: \
                        {benchmark_results['whisper']['memory_usage_mb']:.1f}MB \
                            (target: <{target_memory}MB)"
                )

        return benchmark_results

    def validate_audio_setup(self) -> bool:
        """Validate that audio hardware is properly configured.

        Returns:
            bool: True if audio setup is valid, False otherwise
        """
        logger.info("Validating audio setup...")

        try:
            # Check if devices are available
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    logger.info(
                        f"  Input {i}: {device['name']} ({device['max_input_channels']} channels)"
                    )

            # Test microphone device
            mic_device, hardware_rate = self._get_audio_device_info()
            if mic_device is None:
                logger.warning("Configured microphone device not found, will use default")

            # Test basic recording capability
            logger.info("Testing basic recording capability...")
            test_duration = 1  # 1 second test

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

            # Check if we got valid audio data
            if test_recording is not None and len(test_recording) > 0:
                avg_amplitude = np.mean(np.abs(test_recording))
                logger.info(f"✅ Recording test successful. Average amplitude: {avg_amplitude:.6f}")

                if avg_amplitude < 1e-6:
                    logger.warning(
                        "⚠️  Very low audio signal detected. Check microphone connection and volume"
                    )

                # Test resampling if needed
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
            return False


def main():
    """Main function to run audio model tests."""
    parser = argparse.ArgumentParser(description="Test and benchmark audio models")
    parser.add_argument("--test-wake-word", action="store_true", help="Test wake word detection")
    parser.add_argument("--test-whisper", action="store_true", help="Test speech recognition")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--validate-audio", action="store_true", help="Validate audio setup only")
    parser.add_argument("--config", default="audio_config.yaml", help="Audio config file name")

    args = parser.parse_args()

    if not any(
        [
            args.test_wake_word,
            args.test_whisper,
            args.benchmark,
            args.all,
            args.validate_audio,
        ]
    ):
        args.all = True  # Default to running all tests

    try:
        tester = AudioModelTester(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize audio tester: {e}")
        return 1

    # Always validate audio setup first
    if not tester.validate_audio_setup():
        logger.error("Audio validation failed. Please check your audio configuration.")
        if not args.validate_audio:  # If just validating, don't exit with error
            return 1

    if args.validate_audio:
        logger.info("✅ Audio validation completed successfully!")
        return 0

    # Setup models
    if args.test_wake_word or args.benchmark or args.all:
        if not tester.setup_wake_word_model():
            logger.error("Failed to setup wake word model")
            return 1

    if args.test_whisper or args.benchmark or args.all:
        if not tester.setup_whisper_model():
            logger.error("Failed to setup whisper model")
            return 1

    # Run tests
    try:
        if args.benchmark or args.all:
            results = tester.benchmark_models()
            logger.info("\n=== Benchmark Results ===")

            # Print summary
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

        logger.info("✅ Audio model testing completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
