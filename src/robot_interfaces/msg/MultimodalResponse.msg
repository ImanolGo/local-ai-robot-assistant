# MultimodalResponse.msg
# Message for multimodal AI processing results

std_msgs/Header header

# Identifier matching the original query
string query_id

# Generated response text
string response_text

# Confidence score (0.0 to 1.0)
float32 confidence

# Processing time in seconds
float32 processing_time

# Model information
string model_used
string optimization_used

# Error information (if any)
bool has_error
string error_message
