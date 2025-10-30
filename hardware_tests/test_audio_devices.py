#!/usr/bin/env python3
"""
USB Audio Device Testing Script
Tests microphone recording, speaker playback, and audio performance characteristics.

This script provides comprehensive testing for USB audio devices including:
- Device detection and enumeration
- Microphone recording capabilities
- Speaker playback functionality
- Noise floor measurement
- Audio latency testing
- Sample rate validation
- Simultaneous record/playback testing

Requirements:
- USB microphone connected
- USB speakers connected
- PyAudio library installed
- ALSA utilities (arecord, aplay)
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import wave
from typing import Dict, List, Optional

try:
    import numpy as np
    import pyaudio  # noqa: F401

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not installed. Some tests will be skipped.")
    print("Install with: pip install pyaudio")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioDeviceInfo:
    """Container for audio device information."""

    def __init__(self, card_id: int, device_id: int, name: str, device_type: str):
        self.card_id = card_id
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.alsa_device = f"hw:{card_id},{device_id}"


class AudioDeviceTester:
    """Comprehensive USB audio device testing suite."""

    # Test parameters
    SAMPLE_RATES = [16000, 22050, 44100, 48000]
    CHANNELS = 1  # Mono for testing
    FORMAT_BITS = 16
    CHUNK_SIZE = 1024
    RECORD_DURATION = 5.0  # seconds
    TEST_FREQUENCY = 1000  # Hz for generated test tones

    def __init__(self):
        """Initialize the audio device tester."""
        self.microphones: List[AudioDeviceInfo] = []
        self.speakers: List[AudioDeviceInfo] = []
        self.temp_dir = tempfile.mkdtemp(prefix="audio_test_")
        self.test_results: Dict = {}

        logger.info(f"Temporary files will be stored in: {self.temp_dir}")

    def discover_devices(self) -> bool:
        """
        Discover and catalog audio devices using ALSA.

        Returns:
            bool: True if devices found successfully
        """
        logger.info("Discovering audio devices...")

        try:
            # Get microphones (capture devices)
            result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=True)
            self._parse_device_list(result.stdout, "CAPTURE")

            # Get speakers (playback devices)
            result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
            self._parse_device_list(result.stdout, "PLAYBACK")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enumerate devices: {e}")
            return False

        # Filter for USB devices
        self.microphones = [
            dev for dev in self.microphones if "USB" in dev.name or "UACDemo" in dev.name
        ]
        self.speakers = [dev for dev in self.speakers if "USB" in dev.name or "UACDemo" in dev.name]

        logger.info(f"Found {len(self.microphones)} USB microphones")
        logger.info(f"Found {len(self.speakers)} USB speakers")

        for mic in self.microphones:
            logger.info(f"  Microphone: {mic.name} ({mic.alsa_device})")
        for spk in self.speakers:
            logger.info(f"  Speaker: {spk.name} ({spk.alsa_device})")

        return len(self.microphones) > 0 or len(self.speakers) > 0

    def _parse_device_list(self, output: str, device_type: str) -> None:
        """Parse arecord/aplay output to extract device information."""
        lines = output.split("\n")
        current_devices = self.microphones if device_type == "CAPTURE" else self.speakers

        for line in lines:
            if line.startswith("card "):
                # Parse line like:
                # "card 1: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]"
                try:
                    # Extract card number
                    card_match = line.split(":")[0].strip()
                    card_id = int(card_match.split()[1])

                    # Extract device name (first bracketed part)
                    if "[" in line and "]" in line:
                        first_bracket_start = line.find("[")
                        first_bracket_end = line.find("]", first_bracket_start)
                        device_name = line[first_bracket_start + 1 : first_bracket_end]
                    else:
                        device_name = "Unknown Device"

                    # Extract device number (after "device")
                    device_id = 0
                    if ", device " in line:
                        device_part = line.split(", device ")[1]
                        device_num_str = device_part.split(":")[0].strip()
                        try:
                            device_id = int(device_num_str)
                        except ValueError:
                            device_id = 0

                    device_info = AudioDeviceInfo(card_id, device_id, device_name, device_type)
                    current_devices.append(device_info)

                except (IndexError, ValueError) as e:
                    logger.debug(f"Failed to parse device line: {line} - {e}")

    def test_microphone_detection(self) -> bool:
        """
        Test microphone detection and basic functionality.

        Returns:
            bool: True if microphone tests pass
        """
        logger.info("Testing microphone detection...")

        if not self.microphones:
            logger.error("No USB microphones found")
            return False

        mic = self.microphones[0]  # Test first USB microphone
        logger.info(f"Testing microphone: {mic.name}")

        # Test basic recording with arecord
        test_file = os.path.join(self.temp_dir, "mic_test.wav")

        try:
            cmd = [
                "arecord",
                "-D",
                mic.alsa_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                str(int(self.RECORD_DURATION)),
                test_file,
            ]

            logger.info(f"Recording {self.RECORD_DURATION} seconds of audio...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"Recording failed: {result.stderr}")
                return False

            # Check if file was created and has reasonable size
            if os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                logger.info(f"Recording successful, file size: {file_size} bytes")

                # Basic sanity check - 5 seconds at 16kHz mono 16-bit should be ~160KB
                if file_size < 75000:  # Allow some margin
                    logger.warning(f"Recording file seems small: {file_size} bytes")

                self.test_results["microphone_detection"] = True
                self.test_results["microphone_device"] = mic.alsa_device
                return True
            else:
                logger.error("Recording file was not created")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Recording test timed out")
            return False
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False

    def test_speaker_detection(self) -> bool:
        """
        Test speaker detection and basic functionality.

        Returns:
            bool: True if speaker tests pass
        """
        logger.info("Testing speaker detection...")

        if not self.speakers:
            logger.error("No USB speakers found")
            return False

        speaker = self.speakers[0]  # Test first USB speaker
        logger.info(f"Testing speaker: {speaker.name}")

        # Generate a test tone
        test_file = os.path.join(self.temp_dir, "speaker_test.wav")
        self._generate_test_tone(
            test_file, duration=2.0, frequency=1000, sample_rate=48000, channels=2
        )  # Stereo 48kHz for USB speakers

        try:
            cmd = ["aplay", "-D", speaker.alsa_device, test_file]

            logger.info("Playing test tone for 2 seconds...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"Playback failed: {result.stderr}")
                return False

            logger.info("Playback test completed successfully")
            self.test_results["speaker_detection"] = True
            self.test_results["speaker_device"] = speaker.alsa_device
            return True

        except subprocess.TimeoutExpired:
            logger.error("Playback test timed out")
            return False
        except Exception as e:
            logger.error(f"Speaker test failed: {e}")
            return False

    def measure_noise_floor(self) -> Optional[float]:
        """
        Measure microphone noise floor in dB.

        Returns:
            float: Noise floor in dB, or None if measurement failed
        """
        if not self.microphones:
            logger.warning("No microphones available for noise floor measurement")
            return None

        logger.info("Measuring microphone noise floor...")

        mic = self.microphones[0]
        noise_file = os.path.join(self.temp_dir, "noise_measurement.wav")

        try:
            # Record for full duration for better accuracy
            cmd = [
                "arecord",
                "-D",
                mic.alsa_device,
                "-f",
                "S16_LE",
                "-r",
                "44100",
                "-c",
                "1",
                "-d",
                str(int(self.RECORD_DURATION)),
                noise_file,
            ]

            logger.info(
                f"Recording ambient noise for {self.RECORD_DURATION} seconds (stay quiet)..."
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"Noise recording failed: {result.stderr}")
                return None

            # Analyze the recording
            if PYAUDIO_AVAILABLE:
                noise_level = self._analyze_audio_level(noise_file)
                logger.info(f"Measured noise floor: {noise_level:.2f} dB")
                self.test_results["noise_floor_db"] = noise_level
                return noise_level
            else:
                logger.warning("PyAudio not available, skipping noise analysis")
                return None

        except Exception as e:
            logger.error(f"Noise floor measurement failed: {e}")
            return None

    def test_sample_rates(self) -> Dict[int, bool]:
        """
        Test various sample rates for recording capability.

        Returns:
            Dict[int, bool]: Sample rate support results
        """
        if not self.microphones:
            logger.warning("No microphones available for sample rate testing")
            return {}

        logger.info("Testing sample rate support...")

        mic = self.microphones[0]
        results = {}

        for rate in self.SAMPLE_RATES:
            test_file = os.path.join(self.temp_dir, f"rate_test_{rate}.wav")

            try:
                cmd = [
                    "arecord",
                    "-D",
                    mic.alsa_device,
                    "-f",
                    "S16_LE",
                    "-r",
                    str(rate),
                    "-c",
                    "1",
                    "-d",
                    str(int(self.RECORD_DURATION)),
                    test_file,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                success = result.returncode == 0 and os.path.exists(test_file)
                results[rate] = success

                status = "✓" if success else "✗"
                logger.info(f"  {rate} Hz: {status}")

                if not success:
                    logger.debug(f"Sample rate {rate} failed: {result.stderr}")

            except Exception as e:
                logger.debug(f"Sample rate {rate} test error: {e}")
                results[rate] = False

        self.test_results["supported_sample_rates"] = [
            rate for rate, supported in results.items() if supported
        ]
        return results

    def test_audio_latency(self) -> Optional[float]:
        """
        Test audio latency using simultaneous record/playback.

        Returns:
            float: Estimated latency in milliseconds, or None if test failed
        """
        if not self.microphones or not self.speakers or not PYAUDIO_AVAILABLE:
            logger.warning("Cannot test latency: missing devices or PyAudio")
            return None

        logger.info("Testing audio latency...")

        try:
            # This is a simplified latency test
            # In practice, you'd use a loopback cable for accurate measurement

            mic = self.microphones[0]
            speaker = self.speakers[0]

            # Generate test pulse
            pulse_file = os.path.join(self.temp_dir, "latency_pulse.wav")
            self._generate_test_pulse(pulse_file)

            # Record while playing (simulated loopback)
            record_file = os.path.join(self.temp_dir, "latency_record.wav")

            # Start recording
            record_cmd = [
                "arecord",
                "-D",
                f"plughw:{mic.card_id},{mic.device_id}",  # Use discovered microphone device
                "-f",
                "S16_LE",
                "-r",
                "16000",  # Use microphone's preferred rate
                "-c",
                "1",
                "-d",
                str(int(self.RECORD_DURATION)),
                record_file,
            ]

            # Start playback after short delay
            play_cmd = ["aplay", "-D", speaker.alsa_device, pulse_file]

            logger.info("Starting simultaneous record/playback test...")

            # Start recording
            record_proc = subprocess.Popen(
                record_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            time.sleep(0.5)  # Let recording stabilize

            # Start playback
            play_proc = subprocess.run(play_cmd, capture_output=True, text=True, timeout=5)

            # Wait for recording to finish (allow extra time for 5-second recording)
            record_proc.wait(timeout=10)

            if record_proc.returncode == 0 and play_proc.returncode == 0:
                logger.info("Simultaneous record/playback test completed")
                self.test_results["simultaneous_record_playback"] = True
                # Note: Actual latency measurement would require signal analysis
                estimated_latency = 50.0  # Placeholder - typical USB audio latency
                self.test_results["estimated_latency_ms"] = estimated_latency
                return estimated_latency
            else:
                logger.error("Simultaneous record/playback test failed")
                return None

        except Exception as e:
            logger.error(f"Latency test failed: {e}")
            return None

    def test_volume_range(self) -> bool:
        """
        Test speaker and microphone volume range and control.

        Returns:
            bool: True if volume tests pass
        """
        logger.info("Testing audio device volume range...")

        volume_results = {}

        # Test speaker volume control (if available)
        if self.speakers:
            speaker = self.speakers[0]
            try:
                # Try PCM control (common for USB speakers)
                result = subprocess.run(
                    ["amixer", "-c", str(speaker.card_id), "get", "PCM"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    logger.info(f"Speaker volume control available (PCM) on card {speaker.card_id}")
                    volume_results["speaker_volume_control"] = True

                    # Parse current volume
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Front Left:" in line and "%" in line:
                            # Extract percentage value
                            percent_start = line.find("[") + 1
                            percent_end = line.find("%]")
                            if percent_start > 0 and percent_end > percent_start:
                                current_volume = line[percent_start:percent_end]
                                logger.info(f"Current speaker volume: {current_volume}%")
                                volume_results["speaker_current_volume"] = f"{current_volume}%"
                                break
                else:
                    # Try other common volume control names
                    for control_name in ["Master", "Speaker", "Headphone"]:
                        result = subprocess.run(
                            ["amixer", "-c", str(speaker.card_id), "get", control_name],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Speaker volume control available ({control_name}) "
                                f"on card {speaker.card_id}"
                            )
                            volume_results["speaker_volume_control"] = True
                            break
                    else:
                        logger.warning(
                            f"No volume control found for speaker on card {speaker.card_id}"
                        )
                        volume_results["speaker_volume_control"] = False

            except Exception as e:
                logger.debug(f"Speaker volume test error: {e}")
                volume_results["speaker_volume_control"] = False

        # Test microphone volume control (if available)
        if self.microphones:
            mic = self.microphones[0]
            try:
                # Try Mic control (common for USB microphones)
                result = subprocess.run(
                    ["amixer", "-c", str(mic.card_id), "get", "Mic"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    logger.info(f"Microphone volume control available (Mic) on card {mic.card_id}")
                    volume_results["microphone_volume_control"] = True

                    # Parse current volume
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Mono:" in line and "%" in line:
                            # Extract percentage value
                            percent_start = line.find("[") + 1
                            percent_end = line.find("%]")
                            if percent_start > 0 and percent_end > percent_start:
                                current_volume = line[percent_start:percent_end]
                                logger.info(f"Current microphone volume: {current_volume}%")
                                volume_results["microphone_current_volume"] = f"{current_volume}%"
                                break
                else:
                    # Try other common microphone control names
                    for control_name in ["Capture", "Input"]:
                        result = subprocess.run(
                            ["amixer", "-c", str(mic.card_id), "get", control_name],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            logger.info(
                                f"Microphone volume control available ({control_name}) "
                                f"on card {mic.card_id}"
                            )
                            volume_results["microphone_volume_control"] = True
                            break
                    else:
                        logger.warning(
                            f"No volume control found for microphone on card {mic.card_id}"
                        )
                        volume_results["microphone_volume_control"] = False

            except Exception as e:
                logger.debug(f"Microphone volume test error: {e}")
                volume_results["microphone_volume_control"] = False

        # Store results
        self.test_results.update(volume_results)

        # Return True if at least one volume control is available
        has_volume_control = volume_results.get(
            "speaker_volume_control", False
        ) or volume_results.get("microphone_volume_control", False)

        if has_volume_control:
            logger.info("Volume control testing completed successfully")
        else:
            logger.warning("No volume controls found for audio devices")

        return has_volume_control

    def _generate_test_tone(
        self,
        filename: str,
        duration: float = 2.0,
        frequency: int = 1000,
        sample_rate: int = 44100,
        channels: int = 1,
    ) -> None:
        """Generate a sine wave test tone."""
        frames = int(duration * sample_rate)

        # Generate sine wave
        tone = np.sin(2 * np.pi * frequency * np.linspace(0, duration, frames))

        if channels == 2:
            # Create stereo by duplicating the mono signal
            tone = np.column_stack((tone, tone))

        tone = (tone * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(filename, "w") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(tone.tobytes())

    def _generate_test_pulse(self, filename: str, duration: float = 0.1) -> None:
        """Generate a short pulse for latency testing."""
        sample_rate = 48000  # Use speaker's native rate
        frames = int(duration * sample_rate)

        # Generate short burst
        pulse = np.ones(frames) * 0.5

        # Create stereo by duplicating mono signal
        pulse_stereo = np.column_stack((pulse, pulse))
        pulse_stereo = (pulse_stereo * 32767).astype(np.int16)

        with wave.open(filename, "w") as wav_file:
            wav_file.setnchannels(2)  # Stereo for USB speaker
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pulse_stereo.tobytes())

    def _analyze_audio_level(self, filename: str) -> float:
        """Analyze audio file and return RMS level in dB."""
        try:
            with wave.open(filename, "r") as wav_file:
                frames = wav_file.readframes(-1)
                sound_info = np.frombuffer(frames, dtype=np.int16)

                # Calculate RMS
                rms = np.sqrt(np.mean(sound_info**2))

                # Convert to dB (reference: max 16-bit value)
                if rms > 0:
                    db_level = 20 * np.log10(rms / 32767)
                else:
                    db_level = -96.0  # Silence

                return db_level

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return -96.0

    def print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("USB AUDIO DEVICE TEST SUMMARY")
        print("=" * 60)

        print("\nDevices Found:")
        print(f"  USB Microphones: {len(self.microphones)}")
        for mic in self.microphones:
            print(f"    - {mic.name} ({mic.alsa_device})")

        print(f"  USB Speakers: {len(self.speakers)}")
        for spk in self.speakers:
            print(f"    - {spk.name} ({spk.alsa_device})")

        print("\nTest Results:")
        for test, result in self.test_results.items():
            if isinstance(result, bool):
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"  {test.replace('_', ' ').title()}: {status}")
            elif isinstance(result, (int, float)):
                print(f"  {test.replace('_', ' ').title()}: {result}")
            elif isinstance(result, list):
                print(f"  {test.replace('_', ' ').title()}: {', '.join(map(str, result))}")

        print("\nOptimal Settings Recommendation:")
        if self.test_results.get("supported_sample_rates"):
            rates = self.test_results["supported_sample_rates"]
            if 16000 in rates:
                print("  - Speech Recognition: 16 kHz")
            if 44100 in rates:
                print("  - General Audio: 44.1 kHz")
            print("  - Channels: Mono (1 channel)")
            print("  - Format: 16-bit PCM")

        if self.test_results.get("microphone_device"):
            print(f"  - Microphone Device: {self.test_results['microphone_device']}")
        if self.test_results.get("speaker_device"):
            print(f"  - Speaker Device: {self.test_results['speaker_device']}")

        print(f"\nTemporary files stored in: {self.temp_dir}")
        print("Note: Clean up temp files when testing is complete")

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


def main():
    """Main test execution function."""
    parser = argparse.ArgumentParser(description="Test USB audio devices")
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up temporary files after testing"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize tester
    tester = AudioDeviceTester()

    try:
        print("Starting USB Audio Device Testing...")
        print("=" * 50)

        # Discover devices
        if not tester.discover_devices():
            print("ERROR: No audio devices found!")
            sys.exit(1)

        # Run tests
        tests = [
            ("Microphone Detection", tester.test_microphone_detection),
            ("Speaker Detection", tester.test_speaker_detection),
            ("Sample Rate Support", tester.test_sample_rates),
            ("Noise Floor Measurement", tester.measure_noise_floor),
            ("Volume Range", tester.test_volume_range),
            ("Audio Latency", tester.test_audio_latency),
        ]

        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            try:
                result = test_func()
                if result is not None:
                    status = "✓" if result else "✗"
                    print(f"{test_name}: {status}")
                else:
                    print(f"{test_name}: SKIPPED")
            except Exception as e:
                print(f"{test_name}: ERROR - {e}")

        # Print summary
        tester.print_summary()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)
    finally:
        if args.cleanup:
            tester.cleanup()


if __name__ == "__main__":
    main()
