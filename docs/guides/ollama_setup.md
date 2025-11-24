# Ollama Setup Guide

## Overview
This guide covers the setup of Ollama and the Moondream model for the local AI robot assistant.

## Prerequisites
- NVIDIA Jetson Orin Nano
- Internet connection

## Installation

### Automated Setup
Run the setup script:
```bash
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```

### Manual Installation
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Configure systemd service:
   ```bash
   sudo systemctl enable ollama
   sudo systemctl start ollama
   ```

3. Verify installation:
   ```bash
   curl localhost:11434
   ```

## Model Setup

### Pull Moondream
```bash
ollama pull moondream
```

### Verify Model
```bash
ollama list
```

## Testing
Run a simple inference test:
```bash
ollama run moondream "Hello, are you ready?"
```

## Troubleshooting
- If download fails, check internet connection.
- If service fails to start, check logs: `journalctl -u ollama`.
