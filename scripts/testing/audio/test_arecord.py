import subprocess
import time

import numpy as np

# Configuration
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_MS = 200
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_SAMPLE = 2  # S16_LE
CHUNK_BYTES = CHUNK_SIZE * BYTES_PER_SAMPLE
DURATION = 10

print(f"Testing arecord capture for {DURATION} seconds...")
print(f"Rate: {SAMPLE_RATE}, Chunk: {CHUNK_SIZE} samples ({CHUNK_BYTES} bytes)")

# Command: arecord -D plughw:1,0 -r 44100 -c 1 -f S16_LE -t raw -
cmd = [
    "arecord",
    "-D",
    "plughw:1,0",
    "-r",
    str(SAMPLE_RATE),
    "-c",
    str(CHANNELS),
    "-f",
    "S16_LE",
    "-t",
    "raw",
    "-",  # Output to stdout
]

try:
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=CHUNK_BYTES * 2
    )

    start_time = time.time()
    chunks_read = 0
    total_bytes = 0

    while time.time() - start_time < DURATION:
        # Blocking read from stdout
        raw_data = process.stdout.read(CHUNK_BYTES)

        if not raw_data:
            break

        if len(raw_data) != CHUNK_BYTES:
            print(f"Warning: Incomplete chunk read: {len(raw_data)} bytes")
            continue

        # Convert to numpy array (simulate processing)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # Calculate RMS to verify we have signal
        rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))

        chunks_read += 1
        total_bytes += len(raw_data)

        if chunks_read % 5 == 0:
            elapsed = time.time() - start_time
            rate = chunks_read / elapsed
            print(f"Rate: {rate:.2f} chunks/s (Expected: {1000/CHUNK_MS:.1f}), RMS: {rms:.1f}")

    process.terminate()

    elapsed = time.time() - start_time
    expected_chunks = DURATION * (1000 / CHUNK_MS)

    print("\n--- Results ---")
    print(f"Total chunks: {chunks_read}")
    print(f"Expected chunks: {expected_chunks}")
    print(f"Capture efficiency: {chunks_read / expected_chunks * 100:.1f}%")

    stderr_output = process.stderr.read().decode()
    if stderr_output:
        print(f"\narecord stderr:\n{stderr_output}")

except Exception as e:
    print(f"Error: {e}")
