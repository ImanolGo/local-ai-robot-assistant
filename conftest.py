"""Pytest configuration for Local AI Robot Assistant.

This module configures pytest to properly handle imports from the ROS2 packages
and sets up the test environment.
"""

import sys
from pathlib import Path

# Add source packages to Python path
workspace_root = Path(__file__).parent
src_path = workspace_root / "src"

# Add each package directory to Python path
for package_dir in src_path.iterdir():
    if package_dir.is_dir() and not package_dir.name.startswith("."):
        package_path = str(package_dir)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

# Also add src directory itself
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
