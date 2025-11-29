# MultimodalQuery.msg
# Message for requesting multimodal AI processing

std_msgs/Header header

# Unique identifier for this query
string query_id

# Text component of the query
string text_query

# Whether to include the current camera image
bool include_current_image

# Audio data (optional, for future implementation)
uint8[] audio_data

# Audio format information (if audio_data is provided)
string audio_format
uint32 audio_sample_rate
uint16 audio_channels

# Processing preferences
float32 temperature
uint16 max_tokens
bool use_optimizations
