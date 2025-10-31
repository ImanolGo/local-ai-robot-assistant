"""Audio interface nodes for the Local AI Robot Assistant.

This package contains nodes for:
- Audio capture from USB microphone
- Wake word detection using openWakeWord
- Speech-to-text using Whisper
- Text-to-speech using Piper
- Audio playback through USB speakers
"""

__all__ = [
    "audio_capture_node",
    "wake_word_detector_node",
    "stt_node",
    "tts_node",
    "audio_playback_node",
]
__version__ = "0.1.0"
