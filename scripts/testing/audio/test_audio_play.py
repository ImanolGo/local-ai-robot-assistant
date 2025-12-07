import subprocess
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

print("=" * 70)
print("SOUNDDEVICE / PORTAUDIO DIAGNOSTIC")
print("=" * 70)

# List all devices
print("\nAvailable audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device["max_output_channels"] > 0:
        print(f"\n  Device {i}: {device['name']}")
        print(f"    Max outputs: {device['max_output_channels']}")
        print(f"    Default SR: {device['default_samplerate']}")
        print(f"    Host API: {sd.query_hostapis(device['hostapi'])['name']}")

        # Check if this is our target speaker
        if "UACDemo" in device["name"] or device["max_output_channels"] == 2:
            print("    ⭐ POTENTIAL TARGET DEVICE")

            # Try to check if we can use it
            try:
                sd.check_output_settings(device=i, samplerate=48000, channels=2)
                print("    ✓ Can configure at 48kHz stereo")
            except Exception as e:
                print(f"    ✗ Configuration check failed: {e}")

# Show default device
print("\n" + "-" * 70)
print("Default output device:")
try:
    default = sd.query_devices(kind="output")
    print(f"  {default['name']} (index: {sd.default.device[1]})")
except Exception as e:
    print(f"  Error: {e}")

# Load test audio
print("\n" + "=" * 70)
print("LOADING TEST AUDIO")
print("=" * 70)
audio, sr = sf.read("assets/audio/notify_asc.wav")
print("Audio loaded:")
print(f"  Shape: {audio.shape}")
print(f"  Sample rate: {sr} Hz")
print(f"  Dtype: {audio.dtype}")
print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
print(f"  Duration: {len(audio) / sr:.2f} seconds")

# Convert mono to stereo if needed
if audio.ndim == 1:
    print("\n  Converting mono to stereo...")
    audio = np.column_stack([audio, audio])
    print(f"  New shape: {audio.shape}")

# Ensure correct sample rate for speaker (48kHz)
if sr != 48000:
    print(f"\n  Resampling from {sr}Hz to 48000Hz...")
    # Simple linear interpolation resampling
    ratio = 48000 / sr
    new_length = int(len(audio) * ratio)
    audio_resampled = np.zeros((new_length, 2), dtype=audio.dtype)
    for ch in range(2):
        audio_resampled[:, ch] = np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio[:, ch],
        )
    audio = audio_resampled
    sr = 48000
    print(f"  New sample rate: {sr}Hz, shape: {audio.shape}")

# Test 1: Play with default device
print("\n" + "=" * 70)
print("TEST 1: Playing with DEFAULT device")
print("=" * 70)
try:
    print("Playing... (should hear sound)")
    sd.play(audio, samplerate=sr)
    sd.wait()
    print("✓ Playback completed")
    time.sleep(1)  # Wait before next test
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Try each available output device
print("\n" + "=" * 70)
print("TEST 2: Trying each output device explicitly")
print("=" * 70)
for i, device in enumerate(devices):
    if device["max_output_channels"] >= 2:
        print(f"\nTrying device {i}: {device['name']}")
        try:
            sd.play(audio, samplerate=sr, device=i)
            sd.wait()
            print(f"  ✓ Playback successful on device {i}")
            print("  >>> THIS DEVICE WORKS! <<<")
            time.sleep(1)  # Wait before next test
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            time.sleep(0.5)  # Brief pause even on failure

# Test 3: Compare with aplay (known to work)
print("\n" + "=" * 70)
print("TEST 3: Comparing with aplay (ALSA directly)")
print("=" * 70)
time.sleep(1)  # Pause before aplay test
print("Using aplay to play the same file...")
try:
    result = subprocess.run(
        ["aplay", "-D", "hw:0,0", "assets/audio/notify_asc.wav"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        print("✓ aplay worked successfully")
    else:
        print(f"✗ aplay failed: {result.stderr}")
except Exception as e:
    print(f"✗ Error running aplay: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If aplay works but sounddevice doesn't, the audio_playback_node")
print("should be modified to use subprocess + aplay instead of sounddevice.")
print("=" * 70)
