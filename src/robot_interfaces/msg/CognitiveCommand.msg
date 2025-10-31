# Cognitive command from LLM to behavioral architecture

std_msgs/Header header

# Command types
string ACTION_NAVIGATE = "navigate"
string ACTION_SEARCH = "search"
string ACTION_FOLLOW = "follow"
string ACTION_STOP = "stop"
string ACTION_RETURN_HOME = "return_home"
string ACTION_SPEAK = "speak"
string ACTION_LISTEN = "listen"
string ACTION_EXPLORE = "explore"
string ACTION_PICKUP = "pickup"
string ACTION_CUSTOM = "custom"

string action                # One of the above action constants
string target_object         # Target object name (e.g., "red_ball")
geometry_msgs/Point target_location  # Target coordinates if known
string[] parameters          # Additional action parameters
float32 priority             # Command priority (0.0-1.0)
float32 timeout              # Maximum execution time (seconds)
string response_text         # Text response to speak to user
bool requires_confirmation   # Whether action needs user confirmation
