# Audio system events and states

std_msgs/Header header

# Event type
string EVENT_WAKE_WORD_DETECTED = "wake_word_detected"
string EVENT_SPEECH_STARTED = "speech_started"
string EVENT_SPEECH_ENDED = "speech_ended"
string EVENT_ASR_PROCESSING = "asr_processing"
string EVENT_ASR_COMPLETE = "asr_complete"
string EVENT_TTS_STARTED = "tts_started"
string EVENT_TTS_COMPLETE = "tts_complete"
string EVENT_PLAYBACK_STARTED = "playback_started"
string EVENT_PLAYBACK_COMPLETE = "playback_complete"
string EVENT_AUDIO_ERROR = "audio_error"

string event_type            # One of the above constants
string data                  # Additional event data (e.g., transcribed text)
float32 confidence          # Confidence score for detection events (0.0-1.0)
float32 duration            # Duration of audio segment (seconds)
string device_id            # Audio device identifier
