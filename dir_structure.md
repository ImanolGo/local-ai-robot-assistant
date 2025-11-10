# Local AI Robot Assistant - Directory Structure
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
│   │   │   ├── audio_buffer_node.py
│   │   │   ├── tts_node.py
│   │   │   ├── audio_playback_node.py
│   │   │   └── vad_node.py
│   │   ├── launch/
│   │   │   └── audio_pipeline_launch.py
│   │   ├── test/
│   │   │   ├── test_audio_capture.py
│   │   │   ├── test_wake_word.py
│   │   │   ├── test_audio_buffer.py
│   │   │   ├── test_tts.py
│   │   │   └── test_vad.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── cognitive_core_nodes/
│   │   ├── cognitive_core_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── gemma3n_interface_node.py
│   │   │   ├── multimodal_processor.py
│   │   │   ├── audio_encoder.py
│   │   │   ├── image_preprocessor.py
│   │   │   ├── intent_parser.py
│   │   │   ├── model_manager.py
│   │   │   └── world_state_manager.py
│   │   ├── test/
│   │   │   ├── test_gemma3n_interface.py
│   │   │   ├── test_multimodal_processor.py
│   │   │   ├── test_audio_encoder.py
│   │   │   ├── test_image_preprocessor.py
│   │   │   ├── test_intent_parser.py
│   │   │   ├── test_model_manager.py
│   │   │   └── test_world_state.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── behavioral_nodes/
│   │   ├── behavioral_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── behavior_tree_executor.py
│   │   │   ├── command_router_node.py
│   │   │   ├── navigate_with_tracking_node.py
│   │   │   ├── goal_verification_node.py
│   │   │   ├── dialogue_manager.py
│   │   │   ├── stuck_recovery_node.py
│   │   │   └── action_nodes/
│   │   │       ├── navigation_actions.py
│   │   │       ├── speech_actions.py
│   │   │       ├── perception_actions.py
│   │   │       └── multimodal_actions.py
│   │   ├── behavior_trees/
│   │   │   ├── main_tree.xml
│   │   │   ├── navigation_tree.xml
│   │   │   ├── dialogue_tree.xml
│   │   │   └── multimodal_tree.xml
│   │   ├── test/
│   │   │   ├── test_behavior_nodes.py
│   │   │   ├── test_behavior_tree.py
│   │   │   ├── test_command_router.py
│   │   │   ├── test_goal_verification.py
│   │   │   └── test_multimodal_actions.py
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
│   │   │   ├── ros_bridge.py
│   │   │   └── data_bridge_node.py
│   │   ├── static/
│   │   │   ├── css/
│   │   │   │   └── multimodal_styles.css
│   │   │   ├── js/
│   │   │   │   ├── multimodal_interface.js
│   │   │   │   ├── gemma3n_dashboard.js
│   │   │   │   └── robot_control.js
│   │   │   ├── index.html
│   │   │   └── uploads/
│   │   ├── test/
│   │   │   ├── test_web_server.py
│   │   │   ├── test_multimodal_interface.py
│   │   │   └── test_data_bridge.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   ├── monitoring_nodes/
│   │   ├── monitoring_nodes/
│   │   │   ├── __init__.py
│   │   │   ├── system_monitor_node.py
│   │   │   ├── memory_manager_node.py
│   │   │   ├── performance_profiler_node.py
│   │   │   └── health_checker_node.py
│   │   ├── test/
│   │   │   ├── test_system_monitor.py
│   │   │   ├── test_memory_manager.py
│   │   │   └── test_health_checker.py
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   │
│   └── robot_interfaces/
│       ├── msg/
│       │   ├── ObjectDetection.msg
│       │   ├── DepthImage.msg
│       │   ├── Intent.msg
│       │   ├── RobotStatus.msg
│       │   ├── MultimodalInput.msg
│       │   ├── MultimodalOutput.msg
│       │   ├── AudioTokens.msg
│       │   ├── VisionTokens.msg
│       │   ├── SystemHealth.msg
│       │   └── GoalVerification.msg
│       ├── srv/
│       │   ├── NavigateTo.srv
│       │   ├── EmergencyStop.srv
│       │   ├── ProcessMultimodal.srv
│       │   ├── VerifyGoal.srv
│       │   └── GetSystemStatus.srv
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── README.md
│
├── config/
│   ├── camera_calibration.yaml
│   ├── camera_config.yaml
│   ├── localization_config.yaml
│   ├── audio_config.yaml
│   ├── uart_config.yaml
│   ├── perception_config.yaml
│   ├── behavioral_config.yaml
│   ├── web_interface_config.yaml
│   ├── gemma3n_config.yaml
│   ├── multimodal_config.yaml
│   ├── memory_management_config.yaml
│   └── safety_config.yaml
│
├── launch/
│   ├── full_system_launch.py
│   ├── tier1_perception_launch.py
│   ├── tier2_cognitive_launch.py
│   ├── audio_pipeline_launch.py
│   ├── perception_launch.py
│   ├── actuation_launch.py
│   ├── web_interface_launch.py
│   ├── monitoring_launch.py
│   ├── minimal_system_launch.py         #  emergency mode (motors + wake word)
│   ├── simulation_launch.py             #  Gazebo simulation testing
│   └── debug_system_launch.py           #  verbose logging and diagnostics
│
├── models/
│   ├── README.md                        #  Gemma 3n download instructions
│   ├── model_registry.yaml             #  model version tracking
│   ├── .gitkeep                         # Keep directory in git
│   ├── wake_word/
│   │   ├── openWakeWord.onnx           #  more robust wake word
│   │   └── .gitkeep
│   ├── whisper_tiny_trt/               # KEPT for fallback STT
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── .gitkeep
│   ├── piper_voice/
│   │   ├── en_US-lessac-medium.onnx
│   │   ├── en_US-lessac-medium.onnx.json
│   │   └── .gitkeep
│   ├── yolo_trt/
│   │   ├── yolov11n.engine
│   │   └── .gitkeep
│   ├── depth_trt/
│   │   ├── rt_monodepth_s.engine
│   │   └── .gitkeep
│   └── gemma_3n_e2b/                   #  replaces nanollm_quantized/
│       ├── config.json                 # Gemma 3n E2B configuration
│       ├── model.safetensors           # Main model weights
│       ├── preprocessor_config.json    # Multimodal preprocessing
│       ├── tokenizer.json              # Tokenizer configuration
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
│   ├── test_stt_accuracy.py             # KEPT for fallback testing
│   ├── test_tts_naturalness.py
│   ├── test_slam_accuracy.py
│   ├── test_localization_accuracy.py
│   ├── test_gemma3n_multimodal.py       #  replaces test_llm_reasoning.py
│   ├── test_audio_encoding.py           #  6.25 tokens/sec validation
│   ├── test_vision_processing.py        #  256 tokens/image validation
│   ├── test_multimodal_conversation.py  #  cross-modal interaction
│   ├── test_goal_verification.py        #  multimodal verification
│   └── test_web_interface_multimodal.py #  multimodal web features
│
├── integration_tests/
│   ├── test_voice_to_action.py          #  multimodal voice commands
│   ├── test_navigation.py
│   ├── test_perception_to_action.py
│   ├── test_multimodal_pipeline.py      #  end-to-end multimodal flow
│   ├── test_goal_verification_flow.py   #  multimodal goal verification
│   ├── test_system_degradation.py       #  memory management testing
│   └── test_full_system_multimodal.py   #  complete multimodal system
│
├── scripts/
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── setup_jetson.sh
│   │   ├── download_models.sh
│   │   ├── setup_gemma3n.sh
│   │   └── configure_environment.sh
│   ├── utils/
│   │   ├── convert_models_to_trt.py
│   │   ├── optimize_gemma3n.py
│   │   ├── benchmark_multimodal.py
│   │   ├── validate_gemma3n.py
│   │   ├── backup_maps.py
│   │   └── memory_profiler.py
│   ├── deploy/
│   │   ├── build_all.sh
│   │   ├── start_robot.sh
│   │   └── emergency_mode.sh
│   └── monitoring/
│       ├── system_health_check.py
│       ├── performance_dashboard.py
│       └── thermal_monitor.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── benchmarks/
│   ├── benchmark_perception.py
│   ├── benchmark_audio.py
│   ├── benchmark_gemma3n.py
│   ├── benchmark_multimodal.py
│   ├── benchmark_memory_usage.py
│   ├── benchmark_system_integration.py
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
