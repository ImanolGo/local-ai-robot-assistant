# Environment Setup Troubleshooting

This document helps resolve common issues with the Python virtual environment and direnv setup.

## Common Issues

### 1. `.envrc` not working / Environment not activating

**Symptoms:**

- `which python` returns system Python instead of `.venv/bin/python`
- Import errors for project dependencies
- direnv showing errors

**Solutions:**

```bash
# Re-allow the .envrc file
cd /path/to/local-ai-robot-assistant
direnv allow

# If direnv is not installed
sudo apt install direnv
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc

# Manual activation (alternative to direnv)
source activate_env.sh
```

### 2. Virtual environment not found

**Symptoms:**

- `.venv` directory doesn't exist
- `activate_env.sh` shows "Virtual environment not found"

**Solution:**

```bash
# Recreate the virtual environment
rm -rf .venv
uv venv .venv
uv sync
direnv allow
```

### 3. `uv` command not found

**Symptoms:**

- `uv: command not found`

**Solution:**

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
# or
export PATH="$HOME/.local/bin:$PATH"
```

### 4. Permission issues

**Symptoms:**

- Permission denied errors
- Cannot write to virtual environment

**Solution:**

```bash
# Fix ownership
sudo chown -R $USER:$USER .venv
chmod -R u+w .venv
```

### 5. `.direnv` folder keeps getting created

**Answer:**
This is **normal behavior**. The `.direnv` folder is created automatically by direnv for caching and should not be deleted. It's already included in `.gitignore` so it won't be committed to git.

## Verification Steps

After fixing issues, verify the setup:

```bash
# Check if direnv is working
cd /path/to/local-ai-robot-assistant
direnv status

# Check Python executable
which python
# Should output: /path/to/local-ai-robot-assistant/.venv/bin/python

# Check virtual environment is active
python -c "import sys; print('Virtual env active:', hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))"
# Should output: Virtual env active: True

# Test dependencies
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

## Manual Environment Activation

If direnv is not working, you can manually activate the environment:

```bash
# Option 1: Use the convenience script
source activate_env.sh

# Option 2: Direct activation
source .venv/bin/activate

# Verify activation
echo $VIRTUAL_ENV
# Should output: /path/to/local-ai-robot-assistant/.venv
```

## Clean Reinstall

If all else fails, perform a clean reinstall:

```bash
# Remove existing environment
rm -rf .venv .direnv
rm -f .envrc

# Run setup again
./setup.sh

# Verify
direnv allow
which python
```

## Files Overview

- `.envrc` - direnv configuration file that activates the virtual environment
- `.venv/` - Python virtual environment directory
- `.direnv/` - direnv cache directory (automatically created, don't delete)
- `activate_env.sh` - Manual activation script (alternative to direnv)
- `uv.lock` - Dependency lock file
- `pyproject.toml` - Python project configuration
