local-ai-robot-assistant/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # CI/CD pipeline
│   │   └── tests.yml                 # Automated testing
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── hardware_issue.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/
│   ├── architecture.md               # Architecture document
│   ├── prd.md                        # Product requirements
│   ├── api/                          # API documentation
│   ├── guides/
│   │   ├── hardware_setup.md
│   │   ├── software_installation.md
│   │   ├── quick_start.md
│   │   └── troubleshooting.md
│   └── images/                       # Documentation images
│
├── src/
│   ├── perception_nodes/
│   │   ├── perception_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── camera_driver.py
│   │   │   ├── image_undistort_node.py
│   │   │   ├── object_detector.py
│   │   │   └── depth_estimator.py
│   │   ├── test/
│   │   │   ├── test_camera_driver.py
│   │   │   ├── test_image_undistort.py
│   │   │   ├── test_object_detector.py
│   │   │   └── test_depth_estimator.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── localization_nodes/
│   │   ├── localization_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── uart_imu_node.py
│   │   │   └── slam_node.py
│   │   ├── launch/
│   │   │   └── localization_launch.py
│   │   ├── test/
│   │   │   ├── test_uart_imu.py
│   │   │   └── test_slam.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── audio_interface_nodes/
│   │   ├── audio_interface_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── audio_capture_node.py
│   │   │   ├── wake_word_detector_node.py
│   │   │   ├── stt_node.py
│   │   │   ├── tts_node.py
│   │   │   └── audio_playback_node.py
│   │   ├── launch/
│   │   │   └── audio_pipeline_launch.py
│   │   ├── test/
│   │   │   ├── test_audio_capture.py
│   │   │   ├── test_wake_word.py
│   │   │   ├── test_stt.py
│   │   │   └── test_tts.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── cognitive_core_node/
│   │   ├── cognitive_core_node/
│   │   │   ├── __init__.py
│   │   │   ├── nanollm_interface.py
│   │   │   └── world_state_manager.py
│   │   ├── test/
│   │   │   ├── test_nanollm.py
│   │   │   └── test_intent_extraction.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── behavioral_nodes/
│   │   ├── behavioral_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── behavior_tree_executor.py
│   │   │   ├── dialogue_manager.py
│   │   │   └── action_nodes/
│   │   │       ├── navigation_actions.py
│   │   │       ├── speech_actions.py
│   │   │       └── perception_actions.py
│   │   ├── behavior_trees/
│   │   │   ├── main_tree.xml
│   │   │   ├── navigation_tree.xml
│   │   │   └── dialogue_tree.xml
│   │   ├── test/
│   │   │   ├── test_behavior_nodes.py
│   │   │   └── test_behavior_tree.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── actuation_nodes/
│   │   ├── actuation_nodes/
│   │   │   ├── __init__.py
│   │   │   └── uart_motor_controller.py
│   │   ├── test/
│   │   │   └── test_uart_motor_controller.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── web_interface_nodes/
│   │   ├── web_interface_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── web_server.py
│   │   │   └── ros_bridge.py
│   │   ├── static/
│   │   │   ├── css/
│   │   │   ├── js/
│   │   │   └── index.html
│   │   ├── test/
│   │   │   └── test_web_server.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   └── robot_interfaces/
│       ├── msg/
│       │   ├── ObjectDetection.msg
│       │   ├── DepthImage.msg
│       │   ├── Intent.msg
│       │   └── RobotStatus.msg
│       ├── srv/
│       │   ├── NavigateTo.srv
│       │   └── EmergencyStop.srv
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── README.md
│
├── config/
│   ├── camera_calibration.yaml
│   ├── localization_config.yaml
│   ├── audio_config.yaml
│   ├── uart_config.yaml
│   ├── perception_config.yaml
│   ├── behavioral_config.yaml
│   └── web_interface_config.yaml
│
├── launch/
│   ├── full_system_launch.py
│   ├── perception_launch.py
│   ├── audio_pipeline_launch.py
│   ├── actuation_launch.py
│   └── monitoring_launch.py
│
├── models/
│   ├── README.md                     # Model download instructions
│   ├── .gitkeep                      # Keep directory in git
│   ├── wake_word/
│   │   └── .gitkeep
│   ├── whisper_tiny_trt/
│   │   └── .gitkeep
│   ├── piper_voice/
│   │   └── .gitkeep
│   ├── yolo_trt/
│   │   └── .gitkeep
│   ├── depth_trt/
│   │   └── .gitkeep
│   └── nanollm_quantized/
│       └── .gitkeep
│
├── hardware_tests/
│   ├── test_waveroever_uart.py
│   ├── test_camera_capture.py
│   ├── calibrate_camera.py
│   ├── test_undistortion.py
│   ├── test_audio_devices.py
│   └── test_thermal_power.py
│
├── manual_tests/
│   ├── test_yolo_realtime.py
│   ├── test_depth_accuracy.py
│   ├── test_wake_word_accuracy.py
│   ├── test_stt_accuracy.py
│   ├── test_tts_naturalness.py
│   ├── test_slam_accuracy.py
│   ├── test_localization_accuracy.py
│   ├── test_llm_reasoning.py
│   └── test_web_interface.py
│
├── integration_tests/
│   ├── test_voice_to_action.py
│   ├── test_navigation.py
│   ├── test_perception_to_action.py
│   └── test_full_system.py
│
├── scripts/
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── setup_jetson.sh
│   │   ├── download_models.sh
│   │   └── configure_environment.sh
│   ├── utils/
│   │   ├── convert_models_to_trt.py
│   │   ├── quantize_llm.py
│   │   └── backup_maps.py
│   └── deploy/
│       ├── build_all.sh
│       └── start_robot.sh
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── benchmarks/
│   ├── benchmark_perception.py
│   ├── benchmark_audio.py
│   ├── benchmark_llm.py
│   └── benchmark_system.py
│
├── .gitignore
├── .gitattributes
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── requirements.txt
├── setup.sh
└── STATUS.md                         # Implementation status tracker






