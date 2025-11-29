cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorCommand.msg"
  "msg/ChassisState.msg"
  "msg/AudioData.msg"
  "msg/AudioEvent.msg"
  "msg/CognitiveCommand.msg"
  "msg/ObjectDetection.msg"
  "msg/DepthImage.msg"
  "msg/MultimodalQuery.msg"
  "msg/MultimodalResponse.msg"
  "srv/EmergencyStop.srv"
  "srv/SetMode.srv"
  "srv/CognitiveQuery.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
