from setuptools import find_packages, setup

package_name = "web_interface_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/static", ["web_interface_nodes/static/*"]),
        ("share/" + package_name + "/templates", ["web_interface_nodes/templates/*"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Local AI Robot Team",
    maintainer_email="developer@local-ai-robot.com",
    description="Web interface nodes for monitoring and debugging the robot system",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "web_server = web_interface_nodes.web_server:main",
        ],
    },
)
