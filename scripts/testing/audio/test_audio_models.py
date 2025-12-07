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
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import psutil
import soundfile as sf
import torch
import yaml
from faster_whisper import WhisperModel
from openwakeword import Model as WakeWordModel
from silero_vad import VADIterator, load_silero_vad

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
        self.vad_model = None
        self.vad_iterator = None

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

    def get_usb_microphone(self) -> Optional[str]:
        """
        Detects the first USB microphone available via 'arecord -l'.
        Returns the ALSA device string (e.g., 'plughw:1,0') or None if not found.
        Using 'plughw' ensures automatic sample rate conversion if needed.
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
                return f"plughw:{card_num},{dev_num}"

        except Exception as e:
            logger.error(f"Error detecting USB microphone: {e}")

        return None

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

    def _capture_audio_stream(
        self, duration: float, sample_rate: int, chunk_size: int = 1024
    ) -> Generator[bytes, None, None]:
        """Capture audio from microphone using arecord.

        Yields raw bytes chunks.
        """
        usb_mic = self.get_usb_microphone()
        device = usb_mic if usb_mic else "pulse"

        logger.info(f"Using audio device: {device}")
        if usb_mic:
            logger.info("  (Auto-detected USB Microphone)")

        arecord_cmd = [
            "arecord",
            "-D",
            device,
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(sample_rate),
            "-t",
            "raw",
            "--buffer-size=8192",
        ]

        logger.info(f"Starting capture: {' '.join(arecord_cmd)}")

        process = subprocess.Popen(
            arecord_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        start_time = time.time()

        try:
            while (time.time() - start_time) < duration:
                if process.poll() is not None:
                    stderr = process.stderr.read().decode("utf-8", errors="ignore")
                    logger.error(f"arecord failed: {stderr}")
                    break

                # For streaming, we try to read chunk_size bytes
                # But we handle partial reads by yielding whatever we get
                chunk = process.stdout.read(chunk_size * 2)  # 2 bytes per sample
                if chunk:
                    yield chunk
                else:
                    time.sleep(0.01)

        finally:
            process.terminate()
            process.wait()

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
            # Use resample_poly for efficient, high-quality resampling
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(source_rate, target_rate)
            up = target_rate // g
            down = source_rate // g

            return resample_poly(audio_data, up, down).astype(np.float32)

        except ImportError:
            logger.warning("scipy not available, using simple decimation (lower quality)")
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
            audio_data, sample_rate = sf.read(file_path, dtype="float32")

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            if len(audio_data) == 0:
                raise ValueError("Audio file is empty")

            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Failed to load audio file {file_path}: {e}")
            # Fallback to wave module
            with wave.open(str(file_path), "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                audio_bytes = wf.readframes(wf.getnframes())

                if wf.getsampwidth() == 2:
                    audio_data = (
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                elif wf.getsampwidth() == 4:
                    audio_data = (
                        np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                    )
                else:
                    raise ValueError(f"Unsupported sample width: {wf.getsampwidth()}")

                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                return audio_data, sample_rate

    def test_wake_word_from_file(self, audio_file: Path) -> Dict:
        """Test wake word detection on an audio file."""
        if not self.wake_word_model:
            logger.error("Wake word model not loaded")
            return {}

        logger.info(f"Testing wake word detection on file: {audio_file}")
        audio_data, file_sample_rate = self.load_audio_file(audio_file)

        if file_sample_rate != self.wake_word_sample_rate:
            audio_data = self._resample_audio_fast(
                audio_data, file_sample_rate, self.wake_word_sample_rate
            )

        audio_int16 = (audio_data * 32767).astype(np.int16)
        chunk_samples = int(self.wake_word_sample_rate * self.wake_word_chunk_ms / 1000)
        detections = []
        scores = []

        start_time = time.perf_counter()

        for i in range(0, len(audio_int16), chunk_samples):
            chunk = audio_int16[i : i + chunk_samples]
            if len(chunk) < 160:
                continue

            predictions = self.wake_word_model.predict(chunk)

            for model_name, score in predictions.items():
                scores.append(score)
                if score > 0.5:
                    timestamp = i / self.wake_word_sample_rate
                    detections.append({"time": timestamp, "score": score, "model": model_name})
                    logger.info(f"Wake word detected at {timestamp:.2f}s with score {score:.3f}")

        inference_time = time.perf_counter() - start_time

        return {
            "file": str(audio_file),
            "detections": len(detections),
            "max_score": max(scores) if scores else 0.0,
            "inference_time_ms": inference_time * 1000,
        }

    def test_speech_recognition_from_file(
        self, audio_file: Path, expected_text: Optional[str] = None
    ) -> Dict:
        """Test speech recognition on an audio file."""
        if not self.whisper_model:
            logger.error("Whisper model not loaded")
            return {}

        logger.info(f"Testing speech recognition on file: {audio_file}")
        start_time = time.perf_counter()

        segments, info = self.whisper_model.transcribe(
            str(audio_file),
            beam_size=5,
            language="en",
            vad_filter=True,
        )

        transcription = " ".join([segment.text.strip() for segment in segments])
        inference_time = time.perf_counter() - start_time

        logger.info(f"Got: '{transcription}'")
        if expected_text:
            logger.info(f"Expected: '{expected_text}'")

        results = {
            "file": str(audio_file),
            "transcription": transcription,
            "expected": expected_text,
            "inference_time_ms": inference_time * 1000,
            "real_time_factor": inference_time / info.duration,
        }

        if expected_text:
            trans_words = set(transcription.lower().split())
            expected_words = set(expected_text.lower().split())
            if len(expected_words) > 0:
                matches = len(trans_words & expected_words)
                results["accuracy_percent"] = matches / len(expected_words) * 100

        return results

    def test_audio_files(self) -> Dict:
        """Test models on pre-recorded audio files."""
        logger.info("\n=== Testing with Audio Files ===")
        results = {"wake_word_tests": [], "speech_recognition_tests": []}

        hey_rover_file = ASSETS_DIR / "HeyRover.wav"
        if hey_rover_file.exists() and self.wake_word_model:
            ww_result = self.test_wake_word_from_file(hey_rover_file)
            results["wake_word_tests"].append(ww_result)

            if ww_result["detections"] > 0:
                logger.info(f"✅ Wake Word Test Passed: {ww_result['detections']} detections")
            else:
                logger.error("❌ Wake Word Test Failed: No detections")

        rain_file = ASSETS_DIR / "TheRainInSpain.wav"
        if rain_file.exists() and self.whisper_model:
            sr_result = self.test_speech_recognition_from_file(
                rain_file, "The rain in Spain stays mainly in the plane."
            )
            results["speech_recognition_tests"].append(sr_result)

            acc = sr_result.get("accuracy_percent", 0)
            if acc > 70:
                logger.info(f"✅ Speech Recognition Test Passed: {acc:.1f}% accuracy")
            else:
                logger.error(f"❌ Speech Recognition Test Failed: {acc:.1f}% accuracy")

        return results

    def setup_wake_word_model(self) -> bool:
        """Set up the openWakeWord model."""
        try:
            model_path = MODEL_DIR / "wake_word" / "hey_roe_ver.onnx"
            logger.info("Loading openWakeWord model...")

            models = [str(model_path)] if model_path.exists() else ["hey_roe_ver"]
            self.wake_word_model = WakeWordModel(wakeword_models=models, inference_framework="onnx")

            return True
        except Exception as e:
            logger.error(f"Failed to load wake word model: {e}")
            return False

    def setup_whisper_model(self) -> bool:
        """Set up the faster-whisper model."""
        try:
            logger.info("Loading faster-whisper model...")
            self.whisper_model = WhisperModel(
                "tiny", device="cpu", compute_type="int8", cpu_threads=4, num_workers=1
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load whisper model: {e}")
            return False

    def setup_silero_vad_model(self) -> bool:
        """Set up the Silero VAD model using silero-vad package."""
        try:
            logger.info("Loading Silero VAD model (package)...")
            # Load model with force_onnx_cpu=True (implicit in onnx=True for this package version)
            self.vad_model = load_silero_vad(onnx=True)
            self.vad_iterator = VADIterator(self.vad_model, sampling_rate=16000)
            return True
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            return False

    def test_silero_vad_benchmark(self) -> Dict:
        """Benchmark Silero VAD performance using package."""
        if not self.vad_model:
            return {}

        logger.info("\n=== Silero VAD Benchmark ===")

        batch_size = 1
        sequence_length = 512  # 32ms at 16k

        # Inputs: torch tensor [batch, time]
        dummy_input = torch.zeros((batch_size, sequence_length), dtype=torch.float32)

        # Warmup
        self.vad_model(dummy_input, 16000)

        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(iterations):
                self.vad_model(dummy_input, 16000)

        total_time = time.perf_counter() - start_time
        avg_time_ms = (total_time / iterations) * 1000

        # Real-time factor calculation
        chunk_duration_ms = (sequence_length / 16000) * 1000
        rtf = avg_time_ms / chunk_duration_ms

        logger.info(f"Average inference time: {avg_time_ms:.3f} ms / chunk")
        logger.info(f"Real-time factor: {rtf:.4f}x (lower is better)")

        if rtf < 0.1:
            logger.info("✅ Silero VAD Performance: Excellent")
        else:
            logger.info("⚠️ Silero VAD Performance: Acceptable but could be better")

        return {"avg_inference_time_ms": avg_time_ms, "real_time_factor": rtf}

    def test_silero_vad_from_file(self, audio_file: Path) -> Dict:
        """Test Silero VAD on audio file."""
        if not self.vad_model:
            return {}

        logger.info(f"Testing VAD on file: {audio_file}")

        # Load audio using soundfile (returns float32)
        audio_data, sr = sf.read(str(audio_file), dtype="float32")

        # Resample if needed
        if sr != 16000:
            logger.info(f"Resampling from {sr} to 16000 Hz...")
            # Simple resampling or use existing method
            from scipy.signal import resample_poly

            audio_data = resample_poly(audio_data, 16000, sr)
            sr = 16000

        # Create iterator fresh
        vad_iterator = VADIterator(self.vad_model, sampling_rate=16000)

        # Process in chunks of 512 samples (32ms)
        chunk_size = 512
        speech_chunks = []

        logger.info("Processing...")
        start_time = time.perf_counter()

        # Convert to torch tensor
        wav = torch.tensor(audio_data)

        for i in range(0, len(wav), chunk_size):
            chunk = wav[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break

            speech_dict = vad_iterator(chunk, return_seconds=True)
            if speech_dict:
                speech_chunks.append(speech_dict)
                logger.info(f"Activity detected: {speech_dict}")

        inference_time = time.perf_counter() - start_time
        logger.info(f"Found {len(speech_chunks)} speech segments")

        return {
            "file": str(audio_file),
            "segments": len(speech_chunks),
            "inference_time": inference_time,
        }

    def test_silero_vad_realtime(self):
        """Test Silero VAD with Microphone."""
        if not self.vad_model:
            return

        logger.info("\n=== Real-time VAD Test ===")
        logger.info("Speak into the microphone... (Ctrl+C to stop)")

        vad_iterator = VADIterator(self.vad_model, sampling_rate=16000)
        chunk_size = 512  # 32ms

        try:
            # Reusing capture logic but asking for specific chunk size
            # Note: _capture_audio_stream yields bytes
            for raw_chunk in self._capture_audio_stream(
                duration=30.0,  # 30 seconds test
                sample_rate=16000,
                chunk_size=chunk_size * 2,  # bytes
            ):
                # Convert bytes to float32 tensor
                audio_int16 = np.frombuffer(raw_chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                tensor_chunk = torch.tensor(audio_float32)

                # Check size match
                if len(tensor_chunk) >= chunk_size:
                    # If we got more, just take first chunk for simplicity or iterate
                    # Ideally we buffer, but for quick test:
                    process_chunk = tensor_chunk[:chunk_size]

                    speech_dict = vad_iterator(process_chunk, return_seconds=True)
                    if speech_dict:
                        logger.info(f"🎤 VAD Event: {speech_dict}")

                        # Visualization
                        print(
                            (
                                "🔴 SPEECH DETECTED"
                                if "start" in speech_dict or "end" not in speech_dict
                                else "⚪ SILENCE"
                            ),
                            end="\r",
                        )

        except KeyboardInterrupt:
            pass
        print("\nTest finished.")

    def test_wake_word_detection(self, test_duration: float = 10.0) -> Dict:
        """Test wake word detection performance using arecord."""
        if not self.wake_word_model:
            return {}

        logger.info(f"Testing wake word detection for {test_duration}s...")
        logger.info("Say 'Hey Rover' to test detection")

        results = {
            "detections": 0,
            "avg_cpu_usage": 0.0,
            "avg_inference_time": 0.0,
            "samples_processed": 0,
        }

        cpu_readings = []
        inference_times = []

        # Buffer for accumulating partial chunks
        buffer = b""
        bytes_per_chunk = self.chunk_size * 2  # 2 bytes per sample

        for raw_chunk in self._capture_audio_stream(test_duration, self.wake_word_sample_rate):
            buffer += raw_chunk

            while len(buffer) >= bytes_per_chunk:
                chunk_bytes = buffer[:bytes_per_chunk]
                buffer = buffer[bytes_per_chunk:]

                audio_data = np.frombuffer(chunk_bytes, dtype=np.int16)
                audio_data = self._normalize_audio(audio_data, target_peak=15000)

                # CPU check
                if results["samples_processed"] % 10 == 0:
                    cpu_readings.append(psutil.cpu_percent(interval=None))

                # Inference
                start = time.perf_counter()
                predictions = self.wake_word_model.predict(audio_data)
                inference_times.append((time.perf_counter() - start) * 1000)

                for score in predictions.values():
                    if score > 0.5:
                        results["detections"] += 1
                        logger.info(f"Wake word detected! Score: {score:.3f}")

                results["samples_processed"] += 1

        if cpu_readings:
            results["avg_cpu_usage"] = np.mean(cpu_readings)
        if inference_times:
            results["avg_inference_time"] = np.mean(inference_times)

        return results

    def test_speech_recognition(self, test_phrases: Optional[List[str]] = None) -> Dict:
        """Test speech recognition performance using arecord."""
        if not self.whisper_model:
            return {}

        test_phrases = test_phrases or [
            "Hello robot, how are you?",
            "Move forward slowly",
            "Turn left and stop",
        ]

        logger.info("Testing speech recognition...")
        results = {"transcriptions": []}

        for i, phrase in enumerate(test_phrases):
            logger.info(f"\nTest {i+1}/{len(test_phrases)}")
            logger.info(f"Please say: '{phrase}'")
            input("Press Enter when ready to record...")

            duration = self.audio_config.max_recording_duration
            logger.info(f"Recording... ({duration} seconds)")

            # Capture all audio first
            audio_buffer = b""
            for chunk in self._capture_audio_stream(duration, self.sample_rate):
                audio_buffer += chunk

            if not audio_buffer:
                logger.error("No audio recorded!")
                continue

            # Process
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_int16 = self._normalize_audio(audio_int16, target_peak=20000)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            start = time.perf_counter()
            segments, _ = self.whisper_model.transcribe(
                audio_float, beam_size=5, language="en", vad_filter=True
            )
            transcription = " ".join([s.text.strip() for s in segments])
            inference_time = (time.perf_counter() - start) * 1000

            logger.info(f"Got: {transcription}")
            results["transcriptions"].append(
                {
                    "expected": phrase,
                    "transcribed": transcription,
                    "inference_time_ms": inference_time,
                }
            )

        return results

    def benchmark_models(self) -> Dict:
        """Run comprehensive benchmarks."""
        results = {"wake_word": {}, "whisper": {}}

        if self.wake_word_model:
            logger.info("\n=== Wake Word Benchmark ===")
            results["wake_word"] = self.test_wake_word_detection(test_duration=5.0)

        if self.whisper_model:
            logger.info("\n=== Whisper Benchmark ===")
            # Synthetic benchmark
            test_audio = np.sin(2 * np.pi * 440 * np.arange(0, 2, 1 / self.sample_rate))
            test_audio = (test_audio * 32767).astype(np.int16).astype(np.float32) / 32768.0

            start = time.perf_counter()
            self.whisper_model.transcribe(test_audio, beam_size=1)
            duration = time.perf_counter() - start

            results["whisper"] = {
                "inference_time_ms": duration * 1000,
                "real_time_factor": duration / 2.0,
            }
            logger.info(f"Real-time factor: {results['whisper']['real_time_factor']:.2f}x")

        if self.vad_model:
            results["silero_vad"] = self.test_silero_vad_benchmark()

        return results

    def validate_audio_setup(self) -> bool:
        """Validate audio hardware configuration."""
        logger.info("Validating audio setup...")

        # Check if we can capture 1 second of audio
        try:
            chunks = list(self._capture_audio_stream(1.0, 16000))
            total_bytes = sum(len(c) for c in chunks)
            if total_bytes > 0:
                logger.info(f"✅ Audio capture successful ({total_bytes} bytes)")
                return True
            else:
                logger.error("❌ Audio capture failed: No data received")
                return False
        except Exception as e:
            logger.error(f"❌ Audio validation failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark audio models")
    parser.add_argument("--test-wake-word", action="store_true", help="Test wake word detection")
    parser.add_argument("--test-whisper", action="store_true", help="Test speech recognition")
    parser.add_argument("--test-silero", action="store_true", help="Test Silero VAD")
    parser.add_argument("--test-silero-file", action="store_true", help="Test VAD on audio file")
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
            args.test_silero,
            args.test_silero_file,
            args.benchmark,
            args.test_audio_files,
            args.all,
            args.validate_audio,
        ]
    ):
        args.test_audio_files = True

    try:
        tester = AudioModelTester(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize audio tester: {e}")
        return 1

    if args.validate_audio:
        return 0 if tester.validate_audio_setup() else 1

    # Setup models
    if args.test_wake_word or args.benchmark or args.all or args.test_audio_files:
        if not tester.setup_wake_word_model():
            return 1

    if args.test_whisper or args.benchmark or args.all or args.test_audio_files:
        if not tester.setup_whisper_model():
            return 1

    if args.test_silero or args.test_silero_file or args.benchmark or args.all:
        if not tester.setup_silero_vad_model():
            return 1

    # Run tests
    if args.test_audio_files or args.all:
        tester.test_audio_files()

    if args.test_silero_file or args.all:
        rain_file = ASSETS_DIR / "TheRainInSpain.wav"
        if rain_file.exists():
            tester.test_silero_vad_from_file(rain_file)
        else:
            logger.error("Audio file not found for VAD test")

    if args.benchmark:
        tester.benchmark_models()
    elif args.test_wake_word:
        tester.test_wake_word_detection()
    elif args.test_whisper:
        tester.test_speech_recognition()
    elif args.test_silero:
        tester.test_silero_vad_realtime()

    return 0


if __name__ == "__main__":
    exit(main())
