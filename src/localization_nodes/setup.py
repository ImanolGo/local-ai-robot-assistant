from setuptools import find_packages, setup

package_name = "localization_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            ["config/localization_config.yaml", "config/rtabmap_config.yaml"],
        ),
        (
            "share/" + package_name + "/launch",
            ["launch/localization_launch.py", "launch/slam_launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Local AI Robot Team",
    maintainer_email="developer@local-ai-robot.com",
    description="SLAM and localization nodes including IMU processing and sensor fusion",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "uart_imu_node = localization_nodes.uart_imu_node:main",
            "slam_health_monitor = localization_nodes.slam_node:main",
        ],
    },
)
