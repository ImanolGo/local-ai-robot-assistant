#!/bin/bash
# Quick test script for Phase 4 perception implementation
# Usage: ./quick_test_phase4.sh [level]
# Levels: unit | mock | model | integration | all
# Default: mock (safest option)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test level (default: mock)
TEST_LEVEL="${1:-mock}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 4 Perception Testing${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to print section header
print_header() {
    echo -e "\n${GREEN}==> $1${NC}\n"
}

# Function to check command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 not found. Please install it.${NC}"
        exit 1
    fi
}

# Check dependencies
print_header "Checking dependencies"
check_command python3
check_command colcon
check_command ros2

# Build workspace
if [ "$TEST_LEVEL" != "unit" ]; then
    print_header "Building workspace"
    source /opt/ros/humble/setup.bash
    colcon build --packages-select perception_nodes robot_interfaces --symlink-install
    source install/setup.bash
    echo -e "${GREEN}Build complete${NC}"
fi

# Run tests based on level
case "$TEST_LEVEL" in
    unit)
        print_header "Running Unit Tests"
        echo -e "${BLUE}Running tests with plugin autoload disabled to avoid ROS conflicts${NC}\n"

        echo "Testing object detector..."
        cd "$REPO_ROOT"
        source .venv/bin/activate
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/perception_nodes/test/test_object_detector.py -v --tb=short || true

        echo -e "\nTesting point cloud generator..."
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/perception_nodes/test/test_pointcloud_generator.py -v --tb=short || true

        echo -e "\nTesting image undistort..."
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/perception_nodes/test/test_image_undistort.py -v --tb=short || true

        echo -e "\n${GREEN}Unit tests complete${NC}"
        echo "Note: Tests run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to avoid ROS plugin conflicts"
        ;;

    mock)
        print_header "Running Mock Data Tests"
        echo "This will test nodes with simulated data"
        echo -e "${YELLOW}Note: You'll need to run nodes in separate terminals${NC}\n"

        echo "Test 1: Point Cloud Generator"
        echo "  Terminal 1: python3 manual_tests/test_pointcloud_mock.py"
        echo "  Terminal 2: ./scripts/run_perception_node.sh pointcloud_generator"
        echo ""

        echo "Test 2: Object Detector"
        echo "  Terminal 1: python3 manual_tests/test_object_detector_mock.py"
        echo "  Terminal 2: ./scripts/run_perception_node.sh object_detector"
        echo ""        echo -e "${BLUE}Starting point cloud mock test in 3 seconds...${NC}"
        sleep 3
        python3 manual_tests/test_pointcloud_mock.py
        ;;

    model)
        print_header "Running Model Tests"

        # Check if TensorRT engine exists
        if [ ! -f "models/yolo_trt/yolo11n_fp16.engine" ]; then
            echo -e "${RED}Error: YOLO TensorRT engine not found${NC}"
            echo "Please run: python3 tools/conversion/convert_yolo.py"
            exit 1
        fi

        echo "Testing YOLO inference..."
        python3 scripts/test_yolo.py \
            --models-dir models/yolo_trt \
            --no-huggingface \
            --iterations 50

        echo -e "\n${GREEN}Model tests complete${NC}"
        ;;

    integration)
        print_header "Running Integration Tests"
        echo "Launching full perception pipeline..."
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}\n"

        # Check if models exist
        if [ ! -f "models/yolo_trt/yolo11n_fp16.engine" ]; then
            echo -e "${YELLOW}Warning: YOLO model not found, pipeline may fail${NC}"
        fi

        ros2 launch perception_nodes perception_launch.py visualize:=true
        ;;

    all)
        print_header "Running All Tests"

        echo "1. Unit tests..."
        $0 unit

        echo -e "\n2. Model tests..."
        if [ -f "models/yolo_trt/yolo11n_fp16.engine" ]; then
            $0 model
        else
            echo -e "${YELLOW}Skipping model tests (TensorRT engine not found)${NC}"
        fi

        echo -e "\n${GREEN}All automated tests complete${NC}"
        echo -e "\n${BLUE}Manual tests remaining:${NC}"
        echo "  - Mock data tests (requires multiple terminals)"
        echo "  - Integration tests (requires full system)"
        ;;

    *)
        echo -e "${RED}Unknown test level: $TEST_LEVEL${NC}"
        echo "Usage: $0 [unit|mock|model|integration|all]"
        exit 1
        ;;
esac

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "For more details, see: docs/TESTING_GUIDE_PHASE4.md"
echo ""
