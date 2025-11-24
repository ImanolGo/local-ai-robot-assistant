#!/bin/bash
set -e

echo "Setting up Ollama..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Enable and start service
echo "Configuring systemd service..."
# Check if we have sudo privileges or if we need to ask user
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo to configure systemd service if needed."
    # We try to continue, maybe user has passwordless sudo or it's already running
    sudo systemctl enable ollama || true
    sudo systemctl start ollama || true
else
    systemctl enable ollama
    systemctl start ollama
fi

# Wait for service to be ready
echo "Waiting for Ollama service..."
timeout=30
while ! curl -s localhost:11434 > /dev/null; do
    sleep 1
    timeout=$((timeout - 1))
    if [ "$timeout" -le 0 ]; then
        echo "Timed out waiting for Ollama service."
        exit 1
    fi
done

# Pull Moondream model
echo "Pulling Moondream model..."
ollama pull moondream

echo "Setup complete!"
