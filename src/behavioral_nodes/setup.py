from setuptools import find_packages, setup

package_name = "behavioral_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Local AI Robot Team",
    maintainer_email="developer@local-ai-robot.com",
    description="Behavioral architecture nodes using BehaviorTree.CPP for decision making and task\
          coordination",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "behavior_tree_executor = behavioral_nodes.behavior_tree_executor:main",
            "dialogue_manager = behavioral_nodes.dialogue_manager:main",
        ],
    },
)
