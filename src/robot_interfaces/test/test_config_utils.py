"""
Unit tests for configuration utilities.

Tests the ConfigLoader, ROS2ConfigLoader, and related utility functions.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from rclpy.node import Node

from robot_interfaces.config_utils import (
    CAMERA_CONFIG_SCHEMA,
    UART_CONFIG_SCHEMA,
    ConfigError,
    ConfigLoader,
    ROS2ConfigLoader,
    load_audio_config,
    load_camera_calibration,
    load_camera_config,
    load_perception_config,
    load_uart_config,
    validate_config_schema,
)


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_loader = ConfigLoader(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_config(self, filename: str, content: dict):
        """Create a test configuration file."""
        config_path = Path(self.temp_dir) / filename
        with open(config_path, "w") as f:
            yaml.dump(content, f)
        return config_path

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        test_config = {"test_section": {"param1": "value1", "param2": 42, "param3": True}}

        self.create_test_config("test.yaml", test_config)

        loaded_config = self.config_loader.load_config("test.yaml")
        assert loaded_config == test_config

    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            self.config_loader.load_config("nonexistent.yaml")

    def test_load_empty_config(self):
        """Test loading an empty configuration file."""
        empty_path = Path(self.temp_dir) / "empty.yaml"
        empty_path.touch()

        with pytest.raises(ConfigError, match="Configuration file is empty"):
            self.config_loader.load_config("empty.yaml")

    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        invalid_path = Path(self.temp_dir) / "invalid.yaml"
        with open(invalid_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            self.config_loader.load_config("invalid.yaml")

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = {
            "section1": {"param1": "value1", "param2": 42},
            "section2": {"nested": {"param3": True}},
        }

        required_keys = ["section1.param1", "section1.param2", "section2.nested.param3"]

        assert self.config_loader.validate_config(config, required_keys) is True

    def test_validate_config_missing_keys(self):
        """Test configuration validation with missing keys."""
        config = {
            "section1": {
                "param1": "value1"
                # param2 is missing
            }
        }

        required_keys = ["section1.param1", "section1.param2"]

        with pytest.raises(ConfigError, match="Missing required configuration keys"):
            self.config_loader.validate_config(config, required_keys)

    def test_get_nested_value_success(self):
        """Test getting nested values successfully."""
        config = {"level1": {"level2": {"level3": "target_value"}}}

        value = self.config_loader.get_nested_value(config, "level1.level2.level3")
        assert value == "target_value"

    def test_get_nested_value_default(self):
        """Test getting nested values with default."""
        config = {"existing": "value"}

        value = self.config_loader.get_nested_value(config, "nonexistent.key", "default")
        assert value == "default"

    def test_has_nested_key(self):
        """Test checking for nested keys."""
        config = {"level1": {"level2": {"level3": "value"}}}

        assert self.config_loader._has_nested_key(config, "level1.level2.level3") is True
        assert self.config_loader._has_nested_key(config, "level1.nonexistent") is False


class TestROS2ConfigLoader:
    """Test cases for ROS2ConfigLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_node = Mock(spec=Node)
        self.mock_logger = Mock()
        self.mock_node.get_logger.return_value = self.mock_logger
        self.config_loader = ROS2ConfigLoader(self.mock_node, self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_test_config(self, filename: str, content: dict):
        """Create a test configuration file."""
        config_path = Path(self.temp_dir) / filename
        with open(config_path, "w") as f:
            yaml.dump(content, f)
        return config_path

    def test_load_and_declare_parameters_with_mapping(self):
        """Test loading config and declaring parameters with mapping."""
        test_config = {
            "uart_config": {
                "port": "/dev/ttyTHS0",
                "baudrate": 115200,
                "motor_controller": {"command_rate": 20.0},
            }
        }

        self.create_test_config("test.yaml", test_config)

        parameter_mapping = {
            "uart_config.port": "uart_port",
            "uart_config.baudrate": "uart_baudrate",
            "uart_config.motor_controller.command_rate": "motor_rate",
        }

        loaded_config = self.config_loader.load_and_declare_parameters(
            "test.yaml", parameter_mapping
        )

        assert loaded_config == test_config

        # Verify parameters were declared
        expected_calls = [
            ("uart_port", "/dev/ttyTHS0"),
            ("uart_baudrate", 115200),
            ("motor_rate", 20.0),
        ]

        for param_name, param_value in expected_calls:
            self.mock_node.declare_parameter.assert_any_call(param_name, param_value)

    def test_declare_all_parameters(self):
        """Test declaring all parameters recursively."""
        test_config = {
            "simple_param": "value",
            "number_param": 42,
            "bool_param": True,
            "nested": {"param1": "nested_value", "param2": 3.14},
        }

        self.create_test_config("test.yaml", test_config)

        loaded_config = self.config_loader.load_and_declare_parameters("test.yaml")

        assert loaded_config == test_config

        # Verify all parameters were declared
        expected_calls = [
            ("simple_param", "value"),
            ("number_param", 42),
            ("bool_param", True),
            ("nested.param1", "nested_value"),
            ("nested.param2", 3.14),
        ]

        for param_name, param_value in expected_calls:
            self.mock_node.declare_parameter.assert_any_call(param_name, param_value)


class TestConvenienceFunctions:
    """Test cases for convenience configuration loading functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_config_file(self, filename: str, content: dict):
        """Create a configuration file."""
        config_path = Path(self.temp_dir) / filename
        with open(config_path, "w") as f:
            yaml.dump(content, f)
        return config_path

    def test_load_uart_config_valid(self):
        """Test loading valid UART configuration."""
        uart_config = {
            "uart_config": {
                "port": "/dev/ttyTHS0",
                "baudrate": 115200,
                "motor_controller": {"command_rate": 20.0},
                "imu_node": {"query_rate": 20.0},
            }
        }

        self.create_config_file("uart_config.yaml", uart_config)

        loaded = load_uart_config(self.temp_dir)
        assert loaded == uart_config

    def test_load_uart_config_missing_keys(self):
        """Test loading UART configuration with missing required keys."""
        uart_config = {
            "uart_config": {
                "port": "/dev/ttyTHS0"
                # Missing required keys
            }
        }

        self.create_config_file("uart_config.yaml", uart_config)

        with pytest.raises(ConfigError, match="Missing required configuration keys"):
            load_uart_config(self.temp_dir)

    def test_load_camera_config_valid(self):
        """Test loading valid camera configuration."""
        camera_config = {
            "camera": {
                "device_id": 0,
                "image": {"width": 1640, "height": 1232, "framerate": 30},
            }
        }

        self.create_config_file("camera_config.yaml", camera_config)

        loaded = load_camera_config(self.temp_dir)
        assert loaded == camera_config

    def test_load_audio_config_valid(self):
        """Test loading valid audio configuration."""
        audio_config = {
            "microphone": {
                "device": "hw:1,0",
                "speech_recognition": {"sample_rate": 16000},
            },
            "speaker": {"device": "hw:0,0", "native_format": {"sample_rate": 48000}},
        }

        self.create_config_file("audio_config.yaml", audio_config)

        loaded = load_audio_config(self.temp_dir)
        assert loaded == audio_config

    def test_load_perception_config_valid(self):
        """Test loading valid perception configuration."""
        perception_config = {
            "object_detection": {
                "model": {"engine_path": "/models/yolo.engine"},
                "inference": {"confidence_threshold": 0.5},
            },
            "depth_estimation": {"model": {"engine_path": "/models/depth.engine"}},
        }

        self.create_config_file("perception_config.yaml", perception_config)

        loaded = load_perception_config(self.temp_dir)
        assert loaded == perception_config

    def test_load_camera_calibration_valid(self):
        """Test loading valid camera calibration."""
        calibration = {
            "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "distortion_coefficients": [[0, 0, 0, 0, 0]],
            "image_width": 1640,
            "image_height": 1232,
        }

        self.create_config_file("camera_calibration.yaml", calibration)

        loaded = load_camera_calibration(self.temp_dir)
        assert loaded == calibration


class TestSchemaValidation:
    """Test cases for schema validation."""

    def test_validate_uart_schema_valid(self):
        """Test validating valid UART configuration against schema."""
        config = {
            "uart_config": {
                "port": "/dev/ttyTHS0",
                "baudrate": 115200,
                "timeout": 1.0,
                "motor_controller": {
                    "command_rate": 20.0,
                    "max_speed": 0.5,
                    "wheelbase": 0.16,
                },
                "imu_node": {"query_rate": 20.0, "query_command": 126},
            }
        }

        assert validate_config_schema(config, UART_CONFIG_SCHEMA) is True

    def test_validate_uart_schema_invalid_type(self):
        """Test validating UART configuration with wrong type."""
        config = {
            "uart_config": {
                "port": "/dev/ttyTHS0",
                "baudrate": "115200",  # Should be int, not string
                "timeout": 1.0,
                "motor_controller": {
                    "command_rate": 20.0,
                    "max_speed": 0.5,
                    "wheelbase": 0.16,
                },
                "imu_node": {"query_rate": 20.0, "query_command": 126},
            }
        }

        with pytest.raises(ConfigError, match="Expected.*got"):
            validate_config_schema(config, UART_CONFIG_SCHEMA)

    def test_validate_camera_schema_valid(self):
        """Test validating valid camera configuration against schema."""
        config = {
            "camera": {
                "device_id": 0,
                "image": {"width": 1640, "height": 1232, "framerate": 30},
            }
        }

        assert validate_config_schema(config, CAMERA_CONFIG_SCHEMA) is True


@pytest.fixture(scope="module")
def mock_rclpy():
    """Mock rclpy for testing."""
    with patch("rclpy.init"), patch("rclpy.shutdown"):
        yield


class TestIntegration:
    """Integration tests for configuration loading."""

    def test_end_to_end_config_loading(self, mock_rclpy):
        """Test complete configuration loading workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a complete configuration
            uart_config = {
                "uart_config": {
                    "port": "/dev/ttyTHS0",
                    "baudrate": 115200,
                    "timeout": 1.0,
                    "motor_controller": {
                        "command_rate": 20.0,
                        "max_speed": 0.5,
                        "wheelbase": 0.16,
                    },
                    "imu_node": {"query_rate": 20.0, "query_command": 126},
                }
            }

            config_path = Path(temp_dir) / "uart_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(uart_config, f)

            # Test loading with ConfigLoader
            loader = ConfigLoader(temp_dir)
            loaded_config = loader.load_config("uart_config.yaml")

            # Validate configuration
            required_keys = [
                "uart_config.port",
                "uart_config.baudrate",
                "uart_config.motor_controller.command_rate",
            ]

            assert loader.validate_config(loaded_config, required_keys) is True

            # Test nested value access
            port = loader.get_nested_value(loaded_config, "uart_config.port")
            assert port == "/dev/ttyTHS0"

            baudrate = loader.get_nested_value(loaded_config, "uart_config.baudrate")
            assert baudrate == 115200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
