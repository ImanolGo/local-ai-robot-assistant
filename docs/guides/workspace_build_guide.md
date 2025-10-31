# ROS2 Workspace Build Guide
## Local AI Robot Assistant Project

This guide covers building, verifying, and troubleshooting the ROS2 workspace for the Local AI Robot Assistant project.

---

## Prerequisites

### System Requirements
- **Platform**: NVIDIA Jetson Orin Nano (8GB RAM)
- **OS**: Ubuntu 22.04 LTS with JetPack SDK
- **ROS2**: Humble Hawksbill
- **Python**: 3.10+
- **Available RAM**: At least 4GB free for building

### Dependencies
Ensure you have the required system packages:
```bash
sudo apt update
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    build-essential \
    cmake \
    git \
    python3-flake8 \
    python3-pip \
    python3-pytest-cov \
    python3-setuptools
```

---

## Quick Start

### 1. Source ROS2 Environment
```bash
source /opt/ros/humble/setup.bash
```

### 2. Navigate to Workspace
```bash
cd /home/imanolgo/repos/local-ai-robot-assistant
```

### 3. Build All Packages
```bash
cd src
colcon build
```

### 4. Source Workspace
```bash
source install/setup.bash
```

---

## Detailed Build Instructions

### Build Configuration

The workspace includes optimized colcon configuration in `.colcon/defaults.yaml`:
- **Parallel workers**: 4 (optimized for Jetson Orin Nano)
- **Compiler flags**: Release mode with native optimizations
- **Symlink install**: Enabled for faster development iteration
- **Memory management**: Conservative settings to avoid OOM

### Build Commands

#### Full Workspace Build
```bash
cd src
colcon build
```

#### Build Specific Package
```bash
cd src
colcon build --packages-select <package_name>
```

Example:
```bash
cd src
colcon build --packages-select perception_nodes
```

#### Build with Dependencies
```bash
cd src
colcon build --packages-up-to <package_name>
```

#### Debug Build (for development)
```bash
cd src
colcon build --mixin debug
```

#### Clean Build (from scratch)
```bash
cd src
rm -rf build/ install/ log/
colcon build
```

#### Memory-Constrained Build
If you encounter memory issues:
```bash
cd src
colcon build --mixin low-memory
```

---

## Build Verification

### 1. Check Build Status
After building, verify all packages built successfully:
```bash
cd src
colcon list
```

Expected output (8 packages):
```
actuation_nodes         actuation_nodes         (ros.ament_python)
audio_interface_nodes   audio_interface_nodes   (ros.ament_python)
behavioral_nodes        behavioral_nodes        (ros.ament_python)
cognitive_core_nodes    cognitive_core_nodes    (ros.ament_python)
localization_nodes      localization_nodes      (ros.ament_python)
perception_nodes        perception_nodes        (ros.ament_python)
robot_interfaces        robot_interfaces        (ros.ament_cmake)
web_interface_nodes     web_interface_nodes     (ros.ament_python)
```

### 2. Verify Package Installation
Check that packages are properly installed:
```bash
cd src
ls install/
```

You should see directories for each package:
```
actuation_nodes/
audio_interface_nodes/
behavioral_nodes/
cognitive_core_nodes/
localization_nodes/
perception_nodes/
robot_interfaces/
web_interface_nodes/
setup.bash
setup.sh
...
```

### 3. Test Package Imports
Source the workspace and test Python package imports:
```bash
cd src
source install/setup.bash
python3 -c "import robot_interfaces; print('robot_interfaces imported successfully')"
```

### 4. Verify Custom Messages
Test that custom messages are available:
```bash
source install/setup.bash
ros2 interface list | grep robot_interfaces
```

Expected output:
```
robot_interfaces/msg/AudioEvent
robot_interfaces/msg/ChassisState
robot_interfaces/msg/CognitiveCommand
robot_interfaces/msg/DepthImage
robot_interfaces/msg/MotorCommand
robot_interfaces/msg/ObjectDetection
robot_interfaces/srv/CognitiveQuery
robot_interfaces/srv/EmergencyStop
robot_interfaces/srv/SetMode
```

### 5. Test Node Entry Points
Verify that node entry points are properly configured:
```bash
source install/setup.bash
ros2 pkg executables actuation_nodes
```

Expected output:
```
actuation_nodes uart_motor_controller
```

---

## Testing

### Run Package Tests
```bash
cd src
colcon test
```

### View Test Results
```bash
cd src
colcon test-result --verbose
```

### Test Specific Package
```bash
cd src
colcon test --packages-select robot_interfaces
```

---

## Troubleshooting

### Common Build Issues

#### 1. Memory Issues (OOM)
**Symptoms**: Build process killed, "Killed" messages
**Solution**:
```bash
# Use memory-constrained build
cd src
colcon build --mixin low-memory

# Or reduce parallel workers manually
colcon build --parallel-workers 2
```

#### 2. Missing Dependencies
**Symptoms**: Package not found errors
**Solution**:
```bash
# Update rosdep database
rosdep update

# Install dependencies
cd /home/imanolgo/repos/local-ai-robot-assistant
rosdep install --from-paths src --ignore-src -r -y
```

#### 3. Python Path Issues
**Symptoms**: Import errors for custom packages
**Solution**:
```bash
# Ensure workspace is sourced
cd src
source install/setup.bash

# Check PYTHONPATH
echo $PYTHONPATH
```

#### 4. CMake Configuration Issues
**Symptoms**: CMake configuration errors
**Solution**:
```bash
# Clean and rebuild
cd src
rm -rf build/ install/
colcon build --cmake-clean-cache
```

#### 5. Permission Issues
**Symptoms**: Permission denied errors
**Solution**:
```bash
# Fix ownership
sudo chown -R $USER:$USER /home/imanolgo/repos/local-ai-robot-assistant

# Ensure proper permissions
chmod -R 755 src/
```

### Build Performance Optimization

#### Monitor Build Performance
```bash
# Time the build process
cd src
time colcon build

# Monitor system resources during build
htop
```

#### Optimize for Jetson
```bash
# Increase swap if needed (already done in setup)
sudo swapon --show

# Monitor temperature during build
sudo tegrastats
```

---

## Development Workflow

### Iterative Development
When developing individual nodes:

1. **Edit code** in your package
2. **Build specific package**:
   ```bash
   cd src
   colcon build --packages-select <your_package>
   ```
3. **Source workspace**:
   ```bash
   source install/setup.bash
   ```
4. **Test your changes**:
   ```bash
   ros2 run <your_package> <your_node>
   ```

### Pre-commit Checks
Before committing code:
```bash
# Build entire workspace
cd src
colcon build

# Run tests
colcon test

# Check test results
colcon test-result --verbose

# Verify no warnings or errors
```

---

## Environment Setup Script

For convenience, you can add this to your `~/.bashrc`:

```bash
# ROS2 Local AI Robot Assistant Workspace
alias robot_setup='cd /home/imanolgo/repos/local-ai-robot-assistant/src && source /opt/ros/humble/setup.bash && source install/setup.bash'
alias robot_build='cd /home/imanolgo/repos/local-ai-robot-assistant/src && colcon build'
alias robot_test='cd /home/imanolgo/repos/local-ai-robot-assistant/src && colcon test'
alias robot_clean='cd /home/imanolgo/repos/local-ai-robot-assistant/src && rm -rf build/ install/ log/'
```

Then use:
```bash
robot_setup  # Setup environment
robot_build  # Build workspace
robot_test   # Run tests
robot_clean  # Clean build artifacts
```

---

## Performance Benchmarks

### Expected Build Times (Jetson Orin Nano)
- **Clean build**: ~12-15 minutes
- **Incremental build**: ~30-60 seconds
- **Single package**: ~10-30 seconds

### Memory Usage
- **Peak build memory**: ~3-4GB RAM
- **Parallel workers**: 4 (optimal for 8GB Jetson)
- **Swap usage**: <2GB during build

---

## Next Steps

After successful workspace build:
1. Proceed to **Phase 2.2**: UART Communication Node implementation
2. Set up hardware-in-the-loop testing
3. Begin implementing individual ROS2 nodes

For more information, see:
- `docs/implementation_plan.md` - Complete development roadmap
- `STATUS.md` - Current project status
- `hardware_tests/` - Hardware validation scripts
