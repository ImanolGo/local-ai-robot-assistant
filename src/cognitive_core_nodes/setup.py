import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cognitive_core_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Local AI Robot Team",
    maintainer_email="developer@local-ai-robot.com",
    description="Cognitive core nodes for language understanding and reasoning using local LLMs",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cognitive_client_node = cognitive_core_nodes.cognitive_client_node:main",
        ],
    },
)
