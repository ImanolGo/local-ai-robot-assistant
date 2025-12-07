import time

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

# Configuration matching the node
DEVICE_INDEX = 1  # From previous logs: USB PnP Sound Device
HW_RATE = 44100
TARGET_RATE = 16000
CHUNK_MS = 80
HW_CHUNK = int(HW_RATE * CHUNK_MS / 1000)
DURATION = 5

print(f"Recording for {DURATION} seconds...")
print(f"Device: {DEVICE_INDEX}")
print(f"HW Rate: {HW_RATE}, Target Rate: {TARGET_RATE}")
print(f"HW Chunk: {HW_CHUNK}")

audio_buffer = []


def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    # Simulate the node's processing
    data = indata[:, 0]

    # Linear interpolation resampling
    duration = len(data) / HW_RATE
    target_len = int(duration * TARGET_RATE)
    x_old = np.linspace(0, duration, len(data))
    x_new = np.linspace(0, duration, target_len)
    resampled = np.interp(x_new, x_old, data)

    audio_buffer.append(resampled)


try:
    with sd.InputStream(
        device=DEVICE_INDEX,
        channels=1,
        samplerate=HW_RATE,
        blocksize=HW_CHUNK,
        callback=callback,
    ):
        time.sleep(DURATION)

    print("Recording finished.")

    # Concatenate and save
    full_audio = np.concatenate(audio_buffer)
    print(f"Captured {len(full_audio)} samples")
    print(f"Expected {DURATION * TARGET_RATE} samples")

    rms = np.sqrt(np.mean(full_audio**2))
    print(f"RMS: {rms:.6f}")

    # Save to file
    wav.write("/tmp/test_mic.wav", TARGET_RATE, (full_audio * 32767).astype(np.int16))
    print("Saved to /tmp/test_mic.wav")

except Exception as e:
    print(f"Error: {e}")
