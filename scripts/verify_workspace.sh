#!/bin/bash
# ROS2 Workspace Verification Script
# Local AI Robot Assistant Project

set -e  # Exit on any error

echo "🤖 ROS2 Workspace Build & Verification Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [[ ! -d "src" || ! -f ".colcon/defaults.yaml" ]]; then
    echo -e "${RED}❌ Error: Must be run from the workspace root directory${NC}"
    echo "Expected: /home/imanolgo/repos/local-ai-robot-assistant"
    echo "Current:  $(pwd)"
    exit 1
fi

echo -e "${BLUE}📍 Workspace: $(pwd)${NC}"
echo

# Step 1: Source ROS2 environment
echo -e "${YELLOW}🔧 Step 1: Sourcing ROS2 environment...${NC}"
source /opt/ros/humble/setup.bash
echo -e "${GREEN}✅ ROS2 Humble sourced${NC}"
echo

# Step 2: Change to src directory
echo -e "${YELLOW}🔧 Step 2: Navigating to src directory...${NC}"
cd src
echo -e "${GREEN}✅ In src directory: $(pwd)${NC}"
echo

# Step 3: Clean previous build (optional)
if [[ "$1" == "clean" ]]; then
    echo -e "${YELLOW}🧹 Step 3: Cleaning previous build...${NC}"
    rm -rf build/ install/ log/
    echo -e "${GREEN}✅ Build artifacts cleaned${NC}"
    echo
fi

# Step 4: List packages
echo -e "${YELLOW}🔧 Step 4: Listing packages...${NC}"
PACKAGE_COUNT=$(colcon list | wc -l)
echo -e "${BLUE}Found ${PACKAGE_COUNT} packages:${NC}"
colcon list
echo

# Step 5: Build workspace
echo -e "${YELLOW}🔧 Step 5: Building workspace...${NC}"
echo -e "${BLUE}Using colcon configuration from ../.colcon/defaults.yaml${NC}"
START_TIME=$(date +%s)

if colcon build; then
    END_TIME=$(date +%s)
    BUILD_TIME=$((END_TIME - START_TIME))
    echo -e "${GREEN}✅ Build completed successfully in ${BUILD_TIME} seconds${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi
echo

# Step 6: Source workspace
echo -e "${YELLOW}🔧 Step 6: Sourcing workspace...${NC}"
source install/setup.bash
echo -e "${GREEN}✅ Workspace sourced${NC}"
echo

# Step 7: Verify package installation
echo -e "${YELLOW}🔧 Step 7: Verifying package installation...${NC}"
INSTALL_DIRS=$(ls install/ | grep -v -E '\.(bash|sh|ps1|zsh|py)$|setup|local_setup|COLCON_IGNORE' | wc -l)
echo -e "${BLUE}Installed packages: ${INSTALL_DIRS}${NC}"

if [[ $INSTALL_DIRS -eq $PACKAGE_COUNT ]]; then
    echo -e "${GREEN}✅ All packages installed correctly${NC}"
else
    echo -e "${RED}❌ Package count mismatch${NC}"
    echo "Expected: $PACKAGE_COUNT, Found: $INSTALL_DIRS"
fi
echo

# Step 8: Test custom messages
echo -e "${YELLOW}🔧 Step 8: Testing custom messages...${NC}"
CUSTOM_MSGS=$(ros2 interface list | grep robot_interfaces | wc -l)
if [[ $CUSTOM_MSGS -gt 0 ]]; then
    echo -e "${GREEN}✅ Found ${CUSTOM_MSGS} custom interfaces:${NC}"
    ros2 interface list | grep robot_interfaces | while read -r interface; do
        echo -e "${BLUE}  - ${interface}${NC}"
    done
else
    echo -e "${RED}❌ No custom interfaces found${NC}"
fi
echo

# Step 9: Test node executables
echo -e "${YELLOW}🔧 Step 9: Testing node executables...${NC}"
echo -e "${BLUE}Available executables:${NC}"
for package in actuation_nodes audio_interface_nodes behavioral_nodes cognitive_core_nodes localization_nodes perception_nodes web_interface_nodes; do
    EXECUTABLES=$(ros2 pkg executables $package 2>/dev/null || echo "")
    if [[ -n "$EXECUTABLES" ]]; then
        echo -e "${GREEN}  $package:${NC}"
        echo "$EXECUTABLES" | while read -r line; do
            echo -e "${BLUE}    - $line${NC}"
        done
    else
        echo -e "${YELLOW}  $package: No executables (placeholder package)${NC}"
    fi
done
echo

# Step 10: Memory and performance check
echo -e "${YELLOW}🔧 Step 10: System resource check...${NC}"
MEMORY_MB=$(free -m | awk 'NR==2{printf "%.1f", $3}')
AVAILABLE_MB=$(free -m | awk 'NR==2{printf "%.1f", $7}')
echo -e "${BLUE}Memory usage: ${MEMORY_MB}MB used, ${AVAILABLE_MB}MB available${NC}"

if (( $(echo "$AVAILABLE_MB > 1000" | bc -l) )); then
    echo -e "${GREEN}✅ Sufficient memory available${NC}"
else
    echo -e "${YELLOW}⚠️  Low memory available${NC}"
fi
echo

# Final summary
echo -e "${GREEN}🎉 Workspace Verification Complete!${NC}"
echo "=============================================="
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "${BLUE}  - Packages built: ${PACKAGE_COUNT}${NC}"
echo -e "${BLUE}  - Custom interfaces: ${CUSTOM_MSGS}${NC}"
echo -e "${BLUE}  - Build time: ${BUILD_TIME} seconds${NC}"
echo -e "${BLUE}  - Memory usage: ${MEMORY_MB}MB${NC}"
echo
echo -e "${GREEN}✅ Ready for next development phase!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "${BLUE}  1. Implement UART communication nodes${NC}"
echo -e "${BLUE}  2. Implement camera pipeline nodes${NC}"
echo -e "${BLUE}  3. Run hardware-in-the-loop tests${NC}"
echo

# Optional: Show build log summary if available
if [[ -f "log/latest_build/events.log" ]]; then
    echo -e "${YELLOW}📋 Build log summary (last 10 lines):${NC}"
    tail -10 log/latest_build/events.log
fi
