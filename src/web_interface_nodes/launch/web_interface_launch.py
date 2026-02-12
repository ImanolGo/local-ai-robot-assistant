from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="web_interface_nodes",
                executable="web_server",
                name="web_server",
                output="screen",
            ),
            Node(
                package="web_interface_nodes",
                executable="data_bridge_node",
                name="data_bridge",
                output="screen",
            ),
        ]
    )
