#!/bin/bash
# Convenience script to activate the virtual environment
# Usage: source activate_env.sh

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi
