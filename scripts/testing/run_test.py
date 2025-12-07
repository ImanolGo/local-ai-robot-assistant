#!/usr/bin/env python3
"""Test runner for integration tests.

This script properly sets up the Python path and runs integration tests
with the correct environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_python_path():
    """Add source directories to Python path."""
    workspace_root = Path(__file__).parent.parent
    src_path = workspace_root / "src"

    # Add each package to Python path
    for package_dir in src_path.iterdir():
        if package_dir.is_dir() and not package_dir.name.startswith("."):
            sys.path.insert(0, str(package_dir))

    # Also add src to path for direct imports
    sys.path.insert(0, str(src_path))


def run_integration_test(test_name: str, *args):
    """Run a specific integration test."""
    setup_python_path()

    workspace_root = Path(__file__).parent.parent
    test_path = workspace_root / "integration_tests" / f"{test_name}.py"

    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return 1

    # Run the test with proper environment
    cmd = [sys.executable, str(test_path)] + list(args)

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(sys.path)

    try:
        result = subprocess.run(cmd, env=env, cwd=str(workspace_root))
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 130


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_test.py <test_name> [args...]")
        print("Example: python scripts/run_test.py test_uart_integration --port /dev/ttyTHS1")
        return 1

    test_name = sys.argv[1]
    test_args = sys.argv[2:]

    return run_integration_test(test_name, *test_args)


if __name__ == "__main__":
    sys.exit(main())
