from setuptools import find_packages, setup

package_name = "perception_nodes"

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
    description="Computer vision and perception nodes for object detection, depth estimation, and\
          camera processing",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_driver = perception_nodes.camera_driver:main",
            "image_undistort_node = perception_nodes.image_undistort_node:main",
            "object_detector = perception_nodes.object_detector:main",
            "depth_estimator = perception_nodes.depth_estimator:main",
            "depth_estimation_node = perception_nodes.depth_estimation_node:main",
            "pointcloud_generator = perception_nodes.pointcloud_generator:main",
        ],
    },
)
