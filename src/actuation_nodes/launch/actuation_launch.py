from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="actuation_nodes",
                executable="uart_motor_controller",
                name="motor_controller",
                output="screen",
            ),
        ]
    )
