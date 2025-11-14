#!/usr/bin/env python3
"""
Test script for Piper TTS integration.

This script tests the piper-tts Python package for text-to-speech synthesis,
including quality assessment and latency benchmarking.
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from piper import PiperVoice
except ImportError:
    print("ERROR: piper-tts package not found. Install with: pip install piper-tts")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("WARNING: soundfile and librosa not available. Audio analysis will be limited.")
    sf = None
    librosa = None

try:
    import sounddevice as sd
except ImportError:
    print("WARNING: sounddevice not available. Audio streaming will be disabled.")
    sd = None


class PiperTTSHandler:
    """Handler for Piper TTS operations with performance monitoring."""

    def __init__(self, model_path: str):
        """
        Initialize PiperTTS handler.

        Args:
            model_path: Path to the .onnx model file
        """
        self.model_path = Path(model_path)
        self.voice: Optional[PiperVoice] = None
        self._load_model()

    def _load_model(self):
        """Load the Piper voice model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        config_path = self.model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"Loading Piper model: {self.model_path}")
        start_time = time.time()

        self.voice = PiperVoice.load(str(self.model_path))

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f}s")

    def synthesize(self, text: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, float]:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file

        Returns:
            Tuple of (audio_data, synthesis_time)
        """
        if not self.voice:
            raise RuntimeError("Voice model not loaded")

        print(f"Synthesizing: '{text}'")
        start_time = time.time()

        # Synthesize audio - collect all audio chunks
        audio_chunks = []
        for audio_chunk in self.voice.synthesize(text):
            # Use the audio_int16_bytes attribute for raw bytes data
            audio_chunks.append(audio_chunk.audio_int16_bytes)

        synthesis_time = time.time() - start_time

        # Concatenate all chunks
        audio_bytes = b"".join(audio_chunks)

        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Save to file if requested
        if output_path:
            self._save_audio(audio_data, output_path)
            print(f"Audio saved to: {output_path}")

        print(f"Synthesis completed in {synthesis_time:.3f}s")
        return audio_data, synthesis_time

    def synthesize_stream(self, text: str) -> float:
        """
        Synthesize text to speech with real-time audio streaming playback.

        Args:
            text: Text to synthesize

        Returns:
            float: Total synthesis and playback time
        """
        if not self.voice:
            raise RuntimeError("Voice model not loaded")

        if not sd:
            raise RuntimeError("sounddevice not available. Install with: pip install sounddevice")

        print(f"Streaming synthesis: '{text}'")
        start_time = time.time()

        # Audio configuration based on hardware requirements
        # Piper outputs mono 22050Hz, USB speaker requires stereo 48000Hz
        piper_sample_rate = self.voice.config.sample_rate  # 22050
        speaker_sample_rate = 48000  # USB speaker only supports 48kHz

        # USB Speaker configuration (UACDemoV1.0 - device 0)
        speaker_device = 0  # UACDemoV1.0: USB Audio (hw:0,0) with 2 output channels
        speaker_channels = 2  # Stereo as required

        # Setup a sounddevice OutputStream for the USB speakers
        print(
            f"Configuring audio: Piper {piper_sample_rate}Hz mono -> USB speaker \
                {speaker_sample_rate}Hz stereo"
        )
        stream = sd.OutputStream(
            samplerate=speaker_sample_rate,  # 48000 Hz as required by USB speaker
            channels=speaker_channels,  # Stereo as required by UACDemoV1.0
            dtype="int16",
            device=speaker_device,  # Explicitly use USB speaker device
        )
        stream.start()

        try:
            # Use the regular synthesize method and stream chunks as they're generated
            for audio_chunk in self.voice.synthesize(text):
                # Get the raw audio bytes from the AudioChunk
                audio_bytes = audio_chunk.audio_int16_bytes
                mono_data = np.frombuffer(audio_bytes, dtype=np.int16)

                # Resample mono data first (before converting to stereo)
                if piper_sample_rate != speaker_sample_rate:
                    # Simple linear interpolation upsampling
                    original_length = len(mono_data)
                    resample_ratio = speaker_sample_rate / piper_sample_rate  # ~2.18
                    target_length = int(original_length * resample_ratio)

                    # Use numpy linear interpolation for resampling
                    mono_data = np.interp(
                        np.linspace(0, original_length - 1, target_length),
                        np.arange(original_length),
                        mono_data,
                    ).astype(np.int16)

                # Convert mono to stereo by creating (frames, 2) array
                # Each frame contains [left, right] where both channels have the same value
                _ = len(mono_data)
                stereo_data = np.column_stack((mono_data, mono_data))

                stream.write(stereo_data)
        finally:
            stream.stop()
            stream.close()

        total_time = time.time() - start_time
        print(f"Streaming synthesis and playback completed in {total_time:.3f}s")
        return total_time

    def _save_audio(self, audio_data: np.ndarray, output_path: str):
        """Save audio data to WAV file."""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sample_rate = self.voice.config.sample_rate if self.voice else 22050

        # Convert float32 to int16 for WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def analyze_audio(self, audio_data: np.ndarray) -> dict:
        """
        Analyze synthesized audio quality.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Dictionary with audio metrics
        """
        sample_rate = self.voice.config.sample_rate if self.voice else 22050
        duration = len(audio_data) / sample_rate

        metrics = {
            "duration": duration,
            "sample_rate": sample_rate,
            "samples": len(audio_data),
            "peak_amplitude": np.max(np.abs(audio_data)),
            "rms_amplitude": np.sqrt(np.mean(audio_data**2)),
        }

        # Additional analysis if librosa is available
        if librosa:
            # Fundamental frequency estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            f0_clean = f0[voiced_flag]
            if len(f0_clean) > 0:
                metrics["mean_f0"] = np.mean(f0_clean)
                metrics["f0_std"] = np.std(f0_clean)
                metrics["voiced_ratio"] = np.mean(voiced_flag)

            # Spectral characteristics
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            metrics["spectral_centroid"] = np.mean(
                librosa.feature.spectral_centroid(S=magnitude)[0]
            )
            metrics["zero_crossing_rate"] = np.mean(
                librosa.feature.zero_crossing_rate(audio_data)[0]
            )

        return metrics


def benchmark_latency(handler: PiperTTSHandler, test_phrases: list, num_runs: int = 5) -> dict:
    """
    Benchmark synthesis latency with various text lengths.

    Args:
        handler: PiperTTS handler
        test_phrases: List of phrases to test
        num_runs: Number of runs per phrase for averaging

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    for phrase in test_phrases:
        word_count = len(phrase.split())
        times = []

        print(f"\nBenchmarking: '{phrase}' ({word_count} words)")

        # Warm up
        handler.synthesize(phrase)

        # Actual timing runs
        for run in range(num_runs):
            _, synthesis_time = handler.synthesize(phrase)
            times.append(synthesis_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        time_per_word = avg_time / word_count if word_count > 0 else 0

        results[phrase] = {
            "word_count": word_count,
            "avg_time": avg_time,
            "std_time": std_time,
            "time_per_word": time_per_word,
            "all_times": times,
        }

        print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Per word: {time_per_word:.3f}s/word")

    return results


def benchmark_streaming(handler: PiperTTSHandler, test_phrases: list, num_runs: int = 3) -> dict:
    """
    Benchmark streaming synthesis latency with various text lengths.

    Args:
        handler: PiperTTS handler
        test_phrases: List of phrases to test
        num_runs: Number of runs per phrase for averaging (fewer for streaming)

    Returns:
        Dictionary with streaming benchmark results
    """
    if not sd:
        print("sounddevice not available - skipping streaming benchmark")
        return {}

    print("Note: Streaming benchmark includes both synthesis and playback time")
    print("Ensure speakers are connected for audio output testing")

    results = {}

    for phrase in test_phrases:
        word_count = len(phrase.split())
        times = []

        print(f"\nStreaming benchmark: '{phrase}' ({word_count} words)")

        # Warm up
        try:
            handler.synthesize_stream(phrase)
        except Exception as e:
            print(f"Warm-up failed: {e}")
            continue

        # Actual timing runs
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            try:
                stream_time = handler.synthesize_stream(phrase)
                times.append(stream_time)
            except Exception as e:
                print(f"  Run {run + 1} failed: {e}")
                continue

        if not times:
            print(f"  All runs failed for phrase: {phrase}")
            continue

        avg_time = np.mean(times)
        std_time = np.std(times)
        time_per_word = avg_time / word_count if word_count > 0 else 0

        results[phrase] = {
            "word_count": word_count,
            "avg_time": avg_time,
            "std_time": std_time,
            "time_per_word": time_per_word,
            "all_times": times,
        }

        print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Per word: {time_per_word:.3f}s/word")

    return results


def quality_test(handler: PiperTTSHandler, test_phrases: list, output_dir: str):
    """
    Test synthesis quality with various phrases.

    Args:
        handler: PiperTTS handler
        test_phrases: List of phrases to test
        output_dir: Directory to save audio files
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, phrase in enumerate(test_phrases):
        print(f"\n--- Quality test {i+1}/{len(test_phrases)} ---")

        output_file = os.path.join(output_dir, f"quality_test_{i+1}.wav")
        audio_data, synthesis_time = handler.synthesize(phrase, output_file)

        # Analyze audio quality
        metrics = handler.analyze_audio(audio_data)

        print(f"Text: '{phrase}'")
        print(f"Duration: {metrics['duration']:.2f}s")
        print(f"Peak amplitude: {metrics['peak_amplitude']:.3f}")
        print(f"RMS amplitude: {metrics['rms_amplitude']:.3f}")

        if "mean_f0" in metrics:
            print(f"Mean F0: {metrics['mean_f0']:.1f} Hz")
            print(f"Voiced ratio: {metrics['voiced_ratio']:.2f}")


def main():
    """Main function to run Piper TTS tests."""
    parser = argparse.ArgumentParser(description="Test Piper TTS performance and quality")
    parser.add_argument(
        "--model",
        default="/home/imanolgo/repos/local-ai-robot-assistant/models/piper_voice/en_US-lessac-medium.onnx",  # noqa E501
        help="Path to Piper ONNX model file",
    )
    parser.add_argument(
        "--output-dir",
        default="./piper_test_output",
        help="Directory to save test audio files",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--quality", action="store_true", help="Run quality tests")
    parser.add_argument(
        "--stream", action="store_true", help="Stream audio to speakers in real-time"
    )
    parser.add_argument(
        "--stream-benchmark", action="store_true", help="Run streaming benchmark tests"
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the Piper text-to-speech system.",
        help="Custom text to synthesize",
    )

    args = parser.parse_args()

    try:
        # Initialize handler
        handler = PiperTTSHandler(args.model)

        # Test phrases for different scenarios
        test_phrases = [
            "Hello world!",  # Short
            "The quick brown fox jumps over the lazy dog.",  # Medium
            "This is a test of the Piper text-to-speech synthesis system running on \
                NVIDIA Jetson.",  # Long
            "Robot, please navigate to the kitchen and bring me a glass of water.",  # Command-like
            "I understand your request and will proceed with the task immediately.",  # Responselike
        ]

        # Custom text synthesis
        print("=== Custom Text Synthesis ===")

        if args.stream:
            # Stream audio to speakers
            print("Streaming audio to speakers...")
            stream_time = handler.synthesize_stream(args.text)

            word_count = len(args.text.split())
            print(f"Streamed {word_count} words in {stream_time:.3f}s")
            print(f"Streaming performance: {stream_time/word_count:.3f}s per word")
            print("Note: Streaming includes both synthesis and playback time")
        else:
            # Regular synthesis with file output
            audio_data, synthesis_time = handler.synthesize(
                args.text, os.path.join(args.output_dir, "custom_synthesis.wav")
            )
            _ = handler.analyze_audio(audio_data)

            word_count = len(args.text.split())
            print(f"Synthesized {word_count} words in {synthesis_time:.3f}s")
            print(f"Performance: {synthesis_time/word_count:.3f}s per word")
            print(
                f"Target check: \
                    {'✓ PASS' if synthesis_time < 0.5 and word_count <= 20 else '✗ FAIL'} \
                    (<500ms for ≤20 words)"
            )

        # Latency benchmark
        if args.benchmark:
            print("\n=== Latency Benchmark ===")
            benchmark_results = benchmark_latency(handler, test_phrases)

            # Summary
            print("\n--- Benchmark Summary ---")
            for phrase, result in benchmark_results.items():
                status = (
                    "✓ PASS"
                    if result["avg_time"] < 0.5 and result["word_count"] <= 20
                    else "✗ FAIL"
                )
                print(
                    f"{result['word_count']:2d} words: {result['avg_time']:.3f}s \
                        ({result['time_per_word']:.3f}s/word) {status}"
                )

        # Streaming benchmark
        if args.stream_benchmark:
            print("\n=== Streaming Benchmark ===")
            streaming_results = benchmark_streaming(handler, test_phrases)

            if streaming_results:
                # Summary
                print("\n--- Streaming Benchmark Summary ---")
                print("Note: Times include both synthesis and real-time playback")
                for phrase, result in streaming_results.items():
                    print(
                        f"{result['word_count']:2d} words: {result['avg_time']:.3f}s \
                            ({result['time_per_word']:.3f}s/word) [STREAMING]"
                    )

        # Quality test
        if args.quality:
            print("\n=== Quality Test ===")
            quality_test(handler, test_phrases, args.output_dir)

        print(f"\nTest completed. Audio files saved to: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
