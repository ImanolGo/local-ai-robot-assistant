#!/bin/bash
# Set optimal audio levels for microphone and speaker

echo "Setting audio levels..."

# Try to set Microphone capture volume to 80%
if command -v amixer &> /dev/null; then
    # Try common control names
    amixer sset 'Mic' 80% 2>/dev/null || true
    amixer sset 'Capture' 80% 2>/dev/null || true
    amixer sset 'Master' 90% 2>/dev/null || true
    echo "Audio levels set via amixer."
else
    echo "amixer not found, skipping level adjustment."
fi

# If PulseAudio is used
if command -v pactl &> /dev/null; then
    # Set default source volume to 80%
    pactl set-source-volume @DEFAULT_SOURCE@ 80% 2>/dev/null || true
    echo "Audio levels set via pactl."
fi

echo "Audio level configuration complete."
