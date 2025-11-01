#!/usr/bin/env python3
"""
Generate nvdewarper configuration from camera calibration.
Converts camera_calibration.yaml to dewarp_config.txt format.

Author: Local AI Robot Team
License: Apache-2.0
"""

import argparse
import os
import sys

import numpy as np
import yaml


def generate_dewarp_config(
    calib_path: str,
    output_path: str,
    output_width: int = None,
    output_height: int = None,
):
    """
    Generate nvdewarper config file from camera calibration.

    Args:
        calib_path: Path to camera_calibration.yaml
        output_path: Path to output dewarp_config.txt
        output_width: Output width (default: same as input)
        output_height: Output height (default: same as input)
    """
    try:
        # Load camera calibration
        with open(calib_path, "r") as f:
            calib_data = yaml.safe_load(f)

        # Extract parameters
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["distortion_coefficients"]).flatten()

        input_width = calib_data["image_width"]
        input_height = calib_data["image_height"]

        # Use input dimensions if output not specified
        if output_width is None:
            output_width = input_width
        if output_height is None:
            output_height = input_height

        # Extract camera matrix parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Extract distortion coefficients (k1, k2, p1, p2, k3)
        # Pad with zeros if less than 5 coefficients
        dist_padded = np.pad(dist_coeffs, (0, max(0, 5 - len(dist_coeffs))), "constant")
        k1, k2, p1, p2, k3 = dist_padded[:5]

        # Generate config file content
        config_content = f"""# NVIDIA nvdewarper Configuration File
# Generated from camera calibration: {calib_path}
# Input resolution: {input_width}x{input_height}
# Output resolution: {output_width}x{output_height}

[property]
# Output resolution
output-width={output_width}
output-height={output_height}

# Buffer configuration
num-batch-buffers=4

# Performance settings
cuda-memory-type=0  # 0=device, 1=pinned, 2=unified

[surface0]
# Surface configuration for camera 0
surface-index=0

# Projection type: 1=Perspective (recommended for most cameras)
projection-type=1

# Input dimensions
width={input_width}
height={input_height}

# Camera calibration parameters
# Use keys that the nvdewarper plugin recognizes based on
# the plugin binary strings: focal-length, optical-center,
# and distortion. Values are comma-separated where appropriate.
# Focal length: provide fx,fy
focal-length={fx},{fy}

# Also provide explicit src/dst focal-length keys which some
# nvdewarper versions expect
src-focal-length={fx},{fy}
dst-focal-length={fx},{fy}

# Optical center: provide cx,cy
optical-center={cx},{cy}

# Also provide src/dst optical-center variants
src-optical-center={cx},{cy}
dst-optical-center={cx},{cy}

# Distortion coefficients: k1,k2,p1,p2,k3
distortion={k1},{k2},{p1},{p2},{k3}

# Optional settings (uncomment to use)
# interpolation-method=1  # 0=nearest, 1=linear, 2=cubic
# border-mode=0          # 0=constant, 1=reflect, 2=wrap

# Optional: Region of Interest (ROI) - crops output
# roi-top=0
# roi-left=0
# roi-width={output_width}
# roi-height={output_height}

# Optional: Perspective correction
# pitch=0.0
# yaw=0.0
# roll=0.0
"""

        # Write config file
        with open(output_path, "w") as f:
            f.write(config_content)

        print(f"Generated nvdewarper config: {output_path}")
        print(f"Input: {input_width}x{input_height}, Output: {output_width}x{output_height}")
        print(f"Focal length: fx={fx:.2f}, fy={fy:.2f}")
        print(f"Optical center: cx={cx:.2f}, cy={cy:.2f}")
        print(f"Distortion: k1={k1:.4f}, k2={k2:.4f}, p1={p1:.4f}, p2={p2:.4f}, k3={k3:.4f}")

        return True

    except Exception as e:
        print(f"Error generating dewarp config: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate nvdewarper config from camera calibration"
    )
    parser.add_argument("calibration_file", help="Path to camera_calibration.yaml")
    parser.add_argument(
        "-o",
        "--output",
        default="dewarp_config.txt",
        help="Output path for dewarp_config.txt (default: dewarp_config.txt)",
    )
    parser.add_argument("--output-width", type=int, help="Output width (default: same as input)")
    parser.add_argument("--output-height", type=int, help="Output height (default: same as input)")

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.calibration_file):
        print(f"Error: Calibration file not found: {args.calibration_file}")
        sys.exit(1)

    # Generate config
    success = generate_dewarp_config(
        args.calibration_file, args.output, args.output_width, args.output_height
    )

    if not success:
        sys.exit(1)

    print("\nTo use this config:")
    print(f"1. Copy {args.output} to your robot")
    print("2. Set nvdewarper.use_nvdewarper: true in camera_config.yaml")
    print(f"3. Set nvdewarper.config_file: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
