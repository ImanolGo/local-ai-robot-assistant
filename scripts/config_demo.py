#!/usr/bin/env python3
"""
Example script demonstrating how to use configuration utilities in ROS2 nodes.

This script shows best practices for loading and using configuration files
in the Local AI Robot Assistant project.
"""

import rclpy
from rclpy.node import Node

from robot_interfaces.config_utils import ConfigError, ROS2ConfigLoader


class ExampleConfigNode(Node):
    """Example node showing configuration loading patterns."""

    def __init__(self):
        super().__init__("example_config_node")

        # Initialize configuration loader
        self.config_loader = ROS2ConfigLoader(self)

        # Load configurations
        self.load_configurations()

        # Set up a timer to demonstrate parameter access
        self.timer = self.create_timer(2.0, self.timer_callback)

        self.get_logger().info("Example configuration node initialized")

    def load_configurations(self):
        """Load all required configurations."""
        try:
            # Load UART configuration with parameter mapping
            self.uart_config = self.config_loader.load_and_declare_parameters(
                "uart_config.yaml",
                parameter_mapping={
                    "uart_config.port": "uart_port",
                    "uart_config.baudrate": "uart_baudrate",
                    "uart_config.motor_controller.command_rate": "motor_command_rate",
                    "uart_config.motor_controller.max_speed": "max_motor_speed",
                    "uart_config.imu_node.query_rate": "imu_query_rate",
                },
            )

            # Load camera configuration
            self.camera_config = self.config_loader.load_config("camera_config.yaml")

            # Access specific configuration values
            uart_port = self.config_loader.get_nested_value(self.uart_config, "uart_config.port")
            camera_width = self.config_loader.get_nested_value(
                self.camera_config, "camera.image.width"
            )

            self.get_logger().info(f"UART port: {uart_port}")
            self.get_logger().info(f"Camera width: {camera_width}")

            # Demonstrate parameter access
            self.demonstrate_parameter_access()

        except ConfigError as e:
            self.get_logger().error(f"Configuration error: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"Unexpected error loading configuration: {e}")
            raise

    def demonstrate_parameter_access(self):
        """Show how to access declared parameters."""
        # Get parameters that were declared from configuration
        uart_port = self.get_parameter("uart_port").get_parameter_value().string_value
        uart_baudrate = self.get_parameter("uart_baudrate").get_parameter_value().integer_value
        motor_rate = self.get_parameter("motor_command_rate").get_parameter_value().double_value

        self.get_logger().info("Declared parameters:")
        self.get_logger().info(f"  uart_port: {uart_port}")
        self.get_logger().info(f"  uart_baudrate: {uart_baudrate}")
        self.get_logger().info(f"  motor_command_rate: {motor_rate}")

    def timer_callback(self):
        """Periodic callback demonstrating runtime parameter access."""
        # Access configuration values at runtime
        max_speed = self.config_loader.get_nested_value(
            self.uart_config, "uart_config.motor_controller.max_speed"
        )

        camera_fps = self.config_loader.get_nested_value(
            self.camera_config, "camera.image.framerate"
        )

        self.get_logger().info(
            f"Runtime config access - Max speed: {max_speed}, Camera FPS: {camera_fps}"
        )


class MinimalConfigNode(Node):
    """Minimal example showing simplest configuration loading."""

    def __init__(self):
        super().__init__("minimal_config_node")

        try:
            # Simple configuration loading
            from robot_interfaces.config_utils import load_uart_config

            uart_config = load_uart_config()
            port = uart_config["uart_config"]["port"]
            baudrate = uart_config["uart_config"]["baudrate"]

            self.get_logger().info(f"Simple config loading: {port} at {baudrate} baud")

        except Exception as e:
            self.get_logger().error(f"Configuration loading failed: {e}")


def demonstrate_validation():
    """Demonstrate configuration validation without ROS2."""
    from robot_interfaces.config_utils import (
        UART_CONFIG_SCHEMA,
        ConfigLoader,
        validate_config_schema,
    )

    print("Demonstrating configuration validation...")

    # Load and validate UART configuration
    loader = ConfigLoader()
    try:
        uart_config = loader.load_config("uart_config.yaml")

        # Validate required keys
        required_keys = [
            "uart_config.port",
            "uart_config.baudrate",
            "uart_config.motor_controller.command_rate",
        ]

        loader.validate_config(uart_config, required_keys)
        print("✅ UART configuration validation passed")

        # Validate against schema
        validate_config_schema(uart_config, UART_CONFIG_SCHEMA)
        print("✅ UART schema validation passed")

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")


def main(args=None):
    """Main function demonstrating different usage patterns."""
    print("Configuration utilities demonstration")
    print("=" * 50)

    # Demonstrate validation without ROS2
    demonstrate_validation()
    print()

    # Initialize ROS2
    rclpy.init(args=args)

    try:
        # Choose which example to run
        import sys

        if len(sys.argv) > 1 and sys.argv[1] == "minimal":
            node = MinimalConfigNode()
        else:
            node = ExampleConfigNode()

        # Spin the node
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
