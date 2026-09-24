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

# Merge the installed generated message package into the source package.
#
# `src/robot_interfaces/robot_interfaces/` holds pure-Python utilities
# (e.g. config_utils.py), while the ROS2-generated `robot_interfaces.msg`
# and `robot_interfaces.srv` submodules live in the colcon install tree.
# Because the source directory above shadows the installed package, import
# of `robot_interfaces.msg` would fail unless we extend the package search
# path with the generated location.
try:
    import robot_interfaces as _robot_interfaces

    _installed_pkg = (
        workspace_root
        / "install"
        / "robot_interfaces"
        / "local"
        / "lib"
        / "python3.10"
        / "dist-packages"
        / "robot_interfaces"
    )
    if _installed_pkg.is_dir() and str(_installed_pkg) not in _robot_interfaces.__path__:
        _robot_interfaces.__path__.append(str(_installed_pkg))
except ImportError:
    pass
