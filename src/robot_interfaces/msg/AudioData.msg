# Raw audio data message (compatible with audio_common_msgs/AudioData)

std_msgs/Header header

# Audio format parameters
uint32 sample_rate    # Samples per second (e.g., 16000)
uint8 channels        # Number of audio channels (1 for mono, 2 for stereo)
uint8 sample_format   # Format of audio data (e.g., 16-bit PCM = 1)

# Raw audio data
uint8[] data          # PCM audio data as bytes
