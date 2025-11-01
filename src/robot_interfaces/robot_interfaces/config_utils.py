"""
Configuration loading utilities for ROS2 nodes.

This module provides utilities for loading and validating configuration files
for the Local AI Robot Assistant project.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rclpy.node import Node


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class ConfigLoader:
    """Utility class for loading and validating configuration files."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Default to config directory at workspace root
            current_dir = Path(__file__).parent
            # Navigate up from src/robot_interfaces/robot_interfaces to workspace root
            self.config_dir = current_dir.parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_file: Name of configuration file (e.g., 'uart_config.yaml')

        Returns:
            Dictionary containing configuration data

        Raises:
            ConfigError: If file doesn't exist or is invalid
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if config is None:
                raise ConfigError(f"Configuration file is empty: {config_path}")

            self.logger.info(f"Loaded configuration from {config_path}")
            return config

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Error reading {config_path}: {e}")

    def validate_config(self, config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that configuration contains required keys.

        Args:
            config: Configuration dictionary
            required_keys: List of required keys (supports nested keys with dots)

        Returns:
            True if valid

        Raises:
            ConfigError: If validation fails
        """
        missing_keys = []

        for key in required_keys:
            if not self._has_nested_key(config, key):
                missing_keys.append(key)

        if missing_keys:
            raise ConfigError(f"Missing required configuration keys: {missing_keys}")

        return True

    def _has_nested_key(self, config: Dict[str, Any], key: str) -> bool:
        """Check if nested key exists in configuration."""
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        return True

    def get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get value from nested configuration key.

        Args:
            config: Configuration dictionary
            key: Nested key (e.g., 'uart_config.motor_controller.max_speed')
            default: Default value if key doesn't exist

        Returns:
            Value at key or default
        """
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current


class ROS2ConfigLoader(ConfigLoader):
    """Configuration loader with ROS2 parameter integration."""

    def __init__(self, node: Node, config_dir: Optional[str] = None):
        """
        Initialize ROS2ConfigLoader.

        Args:
            node: ROS2 node instance
            config_dir: Path to configuration directory
        """
        super().__init__(config_dir)
        self.node = node

    def load_and_declare_parameters(
        self, config_file: str, parameter_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration and declare ROS2 parameters.

        Args:
            config_file: Configuration file name
            parameter_mapping: Mapping from config keys to parameter names

        Returns:
            Configuration dictionary
        """
        config = self.load_config(config_file)

        if parameter_mapping:
            self._declare_mapped_parameters(config, parameter_mapping)
        else:
            self._declare_all_parameters(config)

        return config

    def _declare_mapped_parameters(self, config: Dict[str, Any], mapping: Dict[str, str]):
        """Declare parameters based on mapping."""
        for config_key, param_name in mapping.items():
            value = self.get_nested_value(config, config_key)
            if value is not None:
                self.node.declare_parameter(param_name, value)
                self.node.get_logger().debug(f"Declared parameter '{param_name}' = {value}")

    def _declare_all_parameters(self, config: Dict[str, Any], prefix: str = ""):
        """Recursively declare all configuration as parameters."""
        for key, value in config.items():
            param_name = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                self._declare_all_parameters(value, param_name)
            elif isinstance(value, (str, int, float, bool, list)):
                # Declare primitive types and lists
                try:
                    self.node.declare_parameter(param_name, value)
                    self.node.get_logger().debug(f"Declared parameter '{param_name}' = {value}")
                except Exception as e:
                    self.node.get_logger().warn(f"Failed to declare parameter '{param_name}': {e}")


# Convenience functions for common configuration files


def load_uart_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load UART configuration."""
    loader = ConfigLoader(config_dir)
    config = loader.load_config("uart_config.yaml")

    # Validate required UART configuration
    required_keys = [
        "uart_config.port",
        "uart_config.baudrate",
        "uart_config.motor_controller.command_rate",
        "uart_config.imu_node.query_rate",
    ]
    loader.validate_config(config, required_keys)

    return config


def load_camera_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load camera configuration."""
    loader = ConfigLoader(config_dir)
    config = loader.load_config("camera_config.yaml")

    # Validate required camera configuration
    required_keys = [
        "camera.device_id",
        "camera.image.width",
        "camera.image.height",
        "camera.image.framerate",
    ]
    loader.validate_config(config, required_keys)

    return config


def load_audio_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load audio configuration."""
    loader = ConfigLoader(config_dir)
    config = loader.load_config("audio_config.yaml")

    # Validate required audio configuration
    required_keys = [
        "microphone.device",
        "speaker.device",
        "microphone.speech_recognition.sample_rate",
        "speaker.native_format.sample_rate",
    ]
    loader.validate_config(config, required_keys)

    return config


def load_perception_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load perception configuration."""
    loader = ConfigLoader(config_dir)
    config = loader.load_config("perception_config.yaml")

    # Validate required perception configuration
    required_keys = [
        "object_detection.model.engine_path",
        "depth_estimation.model.engine_path",
        "object_detection.inference.confidence_threshold",
    ]
    loader.validate_config(config, required_keys)

    return config


def load_camera_calibration(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load camera calibration data."""
    loader = ConfigLoader(config_dir)
    calibration = loader.load_config("camera_calibration.yaml")

    # Validate required calibration data
    required_keys = [
        "camera_matrix",
        "distortion_coefficients",
        "image_width",
        "image_height",
    ]
    loader.validate_config(calibration, required_keys)

    return calibration


# Configuration validation schemas
UART_CONFIG_SCHEMA = {
    "uart_config": {
        "port": str,
        "baudrate": int,
        "timeout": float,
        "motor_controller": {
            "command_rate": float,
            "max_speed": float,
            "wheelbase": float,
        },
        "imu_node": {
            "query_rate": float,
            "query_command": int,
        },
    }
}

CAMERA_CONFIG_SCHEMA = {
    "camera": {
        "device_id": int,
        "image": {
            "width": int,
            "height": int,
            "framerate": int,
        },
    }
}


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.

    Args:
        config: Configuration to validate
        schema: Schema definition

    Returns:
        True if valid

    Raises:
        ConfigError: If validation fails
    """

    def _validate_recursive(data, schema_part, path=""):
        for key, expected_type in schema_part.items():
            current_path = f"{path}.{key}" if path else key

            if key not in data:
                raise ConfigError(f"Missing required key: {current_path}")

            value = data[key]

            if isinstance(expected_type, dict):
                if not isinstance(value, dict):
                    raise ConfigError(f"Expected dict at {current_path}, got {type(value)}")
                _validate_recursive(value, expected_type, current_path)
            elif not isinstance(value, expected_type):
                raise ConfigError(f"Expected {expected_type} at {current_path}, got {type(value)}")

    try:
        _validate_recursive(config, schema)
        return True
    except Exception as e:
        raise ConfigError(f"Schema validation failed: {e}")


# Example usage for ROS2 nodes
class ExampleConfigNode(Node):
    """Example of using configuration loading in a ROS2 node."""

    def __init__(self):
        super().__init__("example_config_node")

        # Load configuration with parameter declaration
        config_loader = ROS2ConfigLoader(self)

        try:
            # Load UART config and declare relevant parameters
            self.uart_config = config_loader.load_and_declare_parameters(
                "uart_config.yaml",
                parameter_mapping={
                    "uart_config.port": "uart_port",
                    "uart_config.baudrate": "uart_baudrate",
                    "uart_config.motor_controller.command_rate": "motor_command_rate",
                },
            )

            # Access configuration values
            port = config_loader.get_nested_value(self.uart_config, "uart_config.port")
            baudrate = config_loader.get_nested_value(self.uart_config, "uart_config.baudrate")

            self.get_logger().info(f"UART configured: {port} at {baudrate} baud")

        except ConfigError as e:
            self.get_logger().error(f"Configuration error: {e}")
            raise


if __name__ == "__main__":
    # Test configuration loading
    try:
        # Test basic loading
        loader = ConfigLoader()

        uart_config = load_uart_config()
        print("UART config loaded successfully")

        camera_config = load_camera_config()
        print("Camera config loaded successfully")

        audio_config = load_audio_config()
        print("Audio config loaded successfully")

    except ConfigError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
