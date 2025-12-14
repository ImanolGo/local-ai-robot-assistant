#!/bin/bash
# Run audio capture node with proper environment

cd /home/imanolgo/repos/local-ai-robot-assistant
source .venv/bin/activate
source install/setup.bash

python install/audio_interface_nodes/lib/python3.10/site-packages/audio_interface_nodes/audio_capture_node.py
