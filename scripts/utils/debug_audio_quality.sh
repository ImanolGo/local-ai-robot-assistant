#!/bin/bash
set -e

# Configuration
OUTPUT_FILE="/tmp/debug_mic.wav"
DURATION=5
DEVICE="plughw:1,0"
RATE=44100

echo "=================================================="
echo "Audio Quality Debug Tool"
echo "=================================================="
echo "This script will record $DURATION seconds of audio from $DEVICE."
echo "Please speak 'Hey Rover' clearly during the recording."
echo ""
echo "Recording will start in 3 seconds..."
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo "🔴 RECORDING NOW - SPEAK 'HEY JARVIS'!"

# Record using arecord
arecord -D $DEVICE -r $RATE -c 1 -f S16_LE -d $DURATION $OUTPUT_FILE

echo "✅ Recording complete: $OUTPUT_FILE"
echo ""
echo "Analyzing audio file..."

# Run analysis using test_audio_models.py
# We need to set PYTHONPATH to include venv site-packages
export PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages:$PYTHONPATH"

python3 scripts/testing/audio/test_audio_models.py --test-audio-files --file $OUTPUT_FILE

echo "=================================================="
echo "Debug complete."
