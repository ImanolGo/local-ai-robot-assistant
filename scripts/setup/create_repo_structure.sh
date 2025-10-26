#!/usr/bin/env bash
# Create initial repository structure and placeholder files.
# Run from repo root: ./scripts/setup/create_repo_structure.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"

cd "$ROOT"

declare -a DIRS=(
  ".github/workflows"
  ".github/ISSUE_TEMPLATE"
  "docs/api"
  "docs/guides"
  "docs/images"
  "src/perception_nodes/perception_nodes"
  "src/perception_nodes/test"
  "src/localization_nodes/localization_nodes"
  "src/localization_nodes/test"
  "src/audio_interface_nodes/audio_interface_nodes"
  "src/audio_interface_nodes/test"
  "src/actuation_nodes/actuation_nodes"
  "src/actuation_nodes/test"
  "src/behavioral_nodes/behavioral_nodes/action_nodes"
  "src/behavioral_nodes/test"
  "src/web_interface_nodes/web_interface_nodes/static/css"
  "src/web_interface_nodes/web_interface_nodes/static/js"
  "src/robot_interfaces/msg"
  "src/robot_interfaces/srv"
  "config"
  "models/wake_word"
  "models/whisper_tiny_trt"
  "models/piper_voice"
  "models/yolo_trt"
  "models/depth_trt"
  "hardware_tests"
  "manual_tests"
  "integration_tests"
  "launch"
  "scripts/utils"
  "scripts/deploy"
  "docker"
  "benchmarks"
)

echo "Creating directories..."
for d in "${DIRS[@]}"; do
  mkdir -p "$d"
done

# Create .gitkeep for empty directories to keep them in git
echo "Adding .gitkeep placeholders..."
find . -type d \( -path "./.git" -o -path "./venv" \) -prune -o -mindepth 1 -maxdepth 5 -print | while read -r d; do
  # add .gitkeep only to directories we created above and that are empty
  if [[ -d "$d" && -z "$(ls -A "$d")" ]]; then
    touch "$d/.gitkeep"
  fi
done

# Top-level README hint (do not overwrite if exists)
if [[ ! -f README.md ]]; then
  cat > README.md <<'MD'
# Local AI Robot Assistant

See docs/ for architecture and implementation plan.
MD
fi

# Create docs placeholders
echo "Creating docs placeholders..."
: > docs/architecture.md
: > docs/prd.md
cat > docs/guides/quick_start.md <<'MD'
# Quick Start

Instructions to boot Jetson, install dependencies and run the system.
MD

# GitHub templates
echo "Adding GitHub templates..."
cat > .github/ISSUE_TEMPLATE/bug_report.md <<'MD'
---
name: Bug report
about: Create a report to help us improve
---

**Describe the bug**
Steps to reproduce:
1.
2.
3.

**Expected behaviour**
MD

cat > .github/ISSUE_TEMPLATE/feature_request.md <<'MD'
---
name: Feature request
about: Suggest an improvement
---
Describe the feature and use case.
MD

cat > .github/PULL_REQUEST_TEMPLATE.md <<'MD'
## Summary

## Changes

## Testing

## Checklist
- [ ] CI passing
- [ ] Docs updated
MD

# Minimal GitHub Action placeholders
echo "Creating CI placeholders..."
cat > .github/workflows/ci.yml <<'YML'
name: CI

on: [push, pull_request]

jobs:
  placeholder:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Add CI jobs (colcon build, pytest, linters) here"
YML

# Create minimal ROS2 package skeleton for perception_nodes (example)
echo "Creating minimal ROS2 package skeleton for perception_nodes..."
PKG_DIR="src/perception_nodes"
mkdir -p "$PKG_DIR/perception_nodes"
if [[ ! -f "$PKG_DIR/perception_nodes/__init__.py" ]]; then
  cat > "$PKG_DIR/perception_nodes/__init__.py" <<'PY'
"""perception_nodes package (placeholder)."""

__all__ = []
__version__ = "0.0.0"
PY
fi

cat > "$PKG_DIR/package.xml" <<'XML'
<?xml version="1.0"?>
<package format="2">
  <name>perception_nodes</name>
  <version>0.0.0</version>
  <description>Perception ROS2 nodes (placeholder)</description>
  <maintainer email="you@example.com">You</maintainer>
  <license>Apache-2.0</license>
</package>
XML

cat > "$PKG_DIR/setup.py" <<'PY'
from setuptools import setup

setup(
    name='perception_nodes',
    version='0.0.0',
    packages=['perception_nodes'],
)
PY

# Config placeholders
echo "Adding config placeholders..."
: > config/uart_config.yaml
: > config/camera_config.yaml
: > config/audio_config.yaml
: > config/perception_config.yaml

# Add top-level status and contributing if missing
if [[ ! -f STATUS.md ]]; then
  cat > STATUS.md <<'MD'
# STATUS

Project initialization completed.
MD
fi

if [[ ! -f CONTRIBUTING.md ]]; then
  cat > CONTRIBUTING.md <<'MD'
# Contributing

Follow coding standards described in .github/copilot-instructions.md
MD
fi

echo "Repository structure created. Run 'git add' and commit the new files."
