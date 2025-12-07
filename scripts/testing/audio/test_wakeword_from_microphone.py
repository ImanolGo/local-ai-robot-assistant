# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re
import subprocess
import sys

# Imports
import numpy as np
from openwakeword.model import Model


def get_usb_microphone():
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
        print(f"Error detecting USB microphone: {e}")

    return None


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False,
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="models/wake_word/hey_roe_ver.onnx",
    required=False,
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default="onnx",
    required=False,
)

# Detect default device
detected_device = get_usb_microphone()
default_device = detected_device if detected_device else "pulse"

parser.add_argument(
    "--device_hw",
    help="ALSA device (e.g., 'pulse' for PulseAudio, 'plughw:1,0' for direct USB device)",
    type=str,
    default=default_device,
    required=False,
)

args = parser.parse_args()

# Audio configuration
FORMAT = "S16_LE"  # 16-bit signed little-endian (matches pyaudio.paInt16)
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
BYTES_PER_SAMPLE = 2  # S16_LE is 2 bytes per sample

print("\nConfiguring audio capture:")
print(f"  Device: {args.device_hw}")
if detected_device:
    print(f"  (Auto-detected USB Mic: {detected_device})")
else:
    print(f"  (No USB Mic auto-detected, using default/provided: {args.device_hw})")

print(f"  Sample rate: {RATE} Hz")
print(f"  Channels: {CHANNELS}")
print(f"  Chunk size: {CHUNK} samples")
print(f"  Format: {FORMAT}")

# Start arecord subprocess
arecord_cmd = [
    "arecord",
    "-D",
    args.device_hw,
    "-f",
    FORMAT,
    "-c",
    str(CHANNELS),
    "-r",
    str(RATE),
    "-t",
    "raw",
    "--buffer-size=8192",
]

print(f"\nStarting audio capture: {' '.join(arecord_cmd)}")

try:
    audio_process = subprocess.Popen(
        arecord_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,  # Unbuffered for immediate data
    )

    # Give it a moment to start or fail
    import time

    time.sleep(0.5)

    # Check if process is still running
    if audio_process.poll() is not None:
        stderr_output = audio_process.stderr.read().decode("utf-8", errors="ignore")
        print("arecord failed to start!")
        print(f"Error: {stderr_output}")
        sys.exit(1)

    print(f"Successfully started audio capture from {args.device_hw}\n")
except Exception as e:
    print(f"Error starting arecord: {e}")
    print("Make sure the device exists. Check with: arecord -l")
    sys.exit(1)

# Load pre-trained openwakeword models
print("Loading wake word model...")
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path], inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())
print(f"Loaded {n_models} wake word model(s)")

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#" * 100)
    print("Listening for wakewords...")
    print("#" * 100)
    print("\n" * (n_models * 3))

    chunk_count = 0
    try:
        while True:
            # Get audio from arecord subprocess
            # Read exactly CHUNK samples * BYTES_PER_SAMPLE (S16_LE format)
            bytes_to_read = CHUNK * BYTES_PER_SAMPLE
            raw_data = b""
            while len(raw_data) < bytes_to_read:
                chunk = audio_process.stdout.read(bytes_to_read - len(raw_data))
                if not chunk:
                    # End of file or process died
                    break
                raw_data += chunk

            if len(raw_data) != bytes_to_read:
                print(f"\nWarning: Expected {bytes_to_read} bytes, got {len(raw_data)} bytes")
                if len(raw_data) == 0:
                    print("No data from microphone - check device connection")
                    break
                continue

            # Convert raw bytes to numpy int16 array
            audio = np.frombuffer(raw_data, dtype=np.int16)

            # --- Audio Normalization ---
            # Normalize audio to boost volume if it's too low
            # This helps with the USB mic having low gain
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                # Target peak amplitude (e.g., 50% of int16 max)
                # int16 max is 32767. 50% is ~16000.
                # We limit the gain to avoid amplifying noise too much (e.g., max 10x gain)
                target_peak = 15000
                gain = min(target_peak / max_val, 10.0)

                # Only apply gain if signal is weak but not silent (noise floor check)
                if max_val > 100 and gain > 1.0:
                    audio = (audio * gain).astype(np.int16)
            # ---------------------------

            # Feed to openWakeWord model
            prediction = owwModel.predict(audio)

            chunk_count += 1
            if chunk_count % 100 == 0:
                print(f"Processed {chunk_count} chunks", end="\r")

            # Column titles
            n_spaces = 16
            output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

            for mdl in owwModel.prediction_buffer.keys():
                # Add scores in formatted table
                scores = list(owwModel.prediction_buffer[mdl])
                curr_score = format(scores[-1], ".20f").replace("-", "")

                output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | \
                    {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

            # Print results table
            print("\033[F" * (4 * n_models + 1))
            print(output_string_header, "                             ", end="\r")

    except KeyboardInterrupt:
        print("\n\nStopping wake word detection...")
    except Exception as e:
        print(f"\nError in capture loop: {e}")
    finally:
        audio_process.terminate()
        audio_process.wait()
        print("Audio process terminated.")
