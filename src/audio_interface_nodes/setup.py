from setuptools import find_packages, setup

package_name = "audio_interface_nodes"

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
    description="Audio processing nodes for voice interaction including wake word detection, ASR,\
          and TTS",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "audio_capture_node = audio_interface_nodes.audio_capture_node:main",
            "wake_word_detector_node = audio_interface_nodes.wake_word_detector_node:main",
            "stt_node = audio_interface_nodes.stt_node:main",
            "tts_node = audio_interface_nodes.tts_node:main",
            "audio_playback_node = audio_interface_nodes.audio_playback_node:main",
        ],
    },
)
