#!/bin/bash
#
# Depth Anything V2 Environment Validation and Conversion Script
# Validates environment and runs model conversion for Jetson deployment
#
# This script will:
# 1. Validate existing Python environment (NO PACKAGE INSTALLATION)
# 2. Run the conversion script to download and convert models
# 3. Validate the conversion process
#
# Note: Uses existing pyproject.toml environment - no package installation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo -e "${BLUE}=== Depth Anything V2 Environment Validation ===${NC}"
echo "Root directory: $ROOT_DIR"

# Check if we're on Jetson
check_jetson() {
    if [[ -f /etc/nv_tegra_release ]] || [[ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]]; then
        echo -e "${GREEN}✓ Running on NVIDIA Jetson platform${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Not running on Jetson - TensorRT conversion may be limited${NC}"
        return 1
    fi
}

# Validate Python dependencies (NO INSTALLATION)
validate_dependencies() {
    echo -e "${BLUE}Validating Python dependencies...${NC}"

    local missing_deps=()
    local available_deps=()

    # Check if packages are available
    if python3 -c "import torch" 2>/dev/null; then
        available_deps+=("torch")
    else
        missing_deps+=("torch")
    fi

    if python3 -c "import transformers" 2>/dev/null; then
        available_deps+=("transformers")
    else
        missing_deps+=("transformers")
    fi

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        available_deps+=("huggingface_hub")
    else
        missing_deps+=("huggingface_hub")
    fi

    if python3 -c "import cv2" 2>/dev/null; then
        available_deps+=("opencv-python")
    else
        missing_deps+=("opencv-python")
    fi

    if python3 -c "import numpy" 2>/dev/null; then
        available_deps+=("numpy")
    else
        missing_deps+=("numpy")
    fi

    if python3 -c "import PIL" 2>/dev/null; then
        available_deps+=("pillow")
    else
        missing_deps+=("pillow")
    fi

    # Optional TensorRT dependencies (Jetson only)
    if check_jetson; then
        if python3 -c "import tensorrt" 2>/dev/null; then
            available_deps+=("tensorrt")
        else
            missing_deps+=("tensorrt")
        fi

        if python3 -c "import pycuda" 2>/dev/null; then
            available_deps+=("pycuda")
        else
            missing_deps+=("pycuda")
        fi
    fi

    # Report results
    echo -e "${GREEN}Available packages: ${available_deps[*]:-none}${NC}"

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}✗ Missing dependencies: ${missing_deps[*]}${NC}"
        echo -e "${YELLOW}Please install missing dependencies using:${NC}"
        echo "  pip install ${missing_deps[*]}"
        echo "Or add to pyproject.toml dependencies and run: pip install -e ."
        echo "For Jetson setup, see: docs/guides/jetson_orin_setup.md"
        return 1
    else
        echo -e "${GREEN}✓ All required dependencies are available${NC}"
        return 0
    fi
}

# Check available space
check_space() {
    echo -e "${BLUE}Checking available disk space...${NC}"

    local models_dir="$ROOT_DIR/models"
    local available_space_kb=$(df "$models_dir" | awk 'NR==2 {print $4}')
    local available_space_mb=$((available_space_kb / 1024))

    # Need ~2GB for model download + conversion
    local required_space_mb=2048

    if [[ $available_space_mb -lt $required_space_mb ]]; then
        echo -e "${RED}✗ Insufficient disk space. Need ${required_space_mb}MB, have ${available_space_mb}MB${NC}"
        return 1
    else
        echo -e "${GREEN}✓ Sufficient disk space: ${available_space_mb}MB available${NC}"
        return 0
    fi
}

# Run model conversion
run_conversion() {
    echo -e "${BLUE}Starting model conversion...${NC}"

    cd "$ROOT_DIR"

    # Check if conversion script exists
    local conversion_script="tools/conversion/convert_depth.py"
    if [[ ! -f "$conversion_script" ]]; then
        echo -e "${RED}✗ Conversion script not found: $conversion_script${NC}"
        return 1
    fi

    # Check if test image exists
    local test_image="assets/images/bus.jpg"
    if [[ ! -f "$test_image" ]]; then
        echo -e "${YELLOW}⚠ Test image not found: $test_image${NC}"
        echo -e "${YELLOW}  Conversion will use dummy image${NC}"
    else
        echo -e "${GREEN}✓ Using test image: $test_image${NC}"
    fi

    # Run conversion
    echo -e "${BLUE}Running conversion script...${NC}"
    python3 "$conversion_script" \
        --model-name "depth-anything/Depth-Anything-V2-Small-hf"

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ Model conversion completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Model conversion failed with exit code $exit_code${NC}"
        return 1
    fi
}
# Validate conversion results
validate_conversion() {
    echo -e "${BLUE}Validating conversion results...${NC}"

    local models_dir="$ROOT_DIR/models/depth_trt"

    # Check for converted files
    if [[ -f "$models_dir/model.onnx" ]]; then
        echo -e "${GREEN}✓ ONNX model created${NC}"
    else
        echo -e "${YELLOW}⚠ ONNX model not found${NC}"
    fi

    if [[ -f "$models_dir/model.engine" ]]; then
        echo -e "${GREEN}✓ TensorRT engine created${NC}"
    else
        echo -e "${YELLOW}⚠ TensorRT engine not found (may not be available)${NC}"
    fi

    if [[ -f "$models_dir/conversion_config.json" ]]; then
        echo -e "${GREEN}✓ Conversion config saved${NC}"
        echo "Conversion details:"
        cat "$models_dir/conversion_config.json" | python3 -m json.tool
    else
        echo -e "${YELLOW}⚠ Conversion config not found${NC}"
    fi

    return 0
}

# Print usage information
print_usage() {
    echo -e "${GREEN}Setup completed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Test the depth estimation node:"
    echo "   ros2 run perception_nodes depth_estimation_node"
    echo ""
    echo "2. Launch the perception pipeline:"
    echo "   ros2 launch launch/perception_launch.py"
    echo ""
    echo "3. View depth output topics:"
    echo "   ros2 topic list | grep depth"
    echo "   rviz2  # visualize depth images"
    echo ""
    echo "Files created:"
    echo "- Model files: models/depth_trt/"
    echo "- ROS node: src/perception_nodes/perception_nodes/depth_estimation_node.py"
    echo "- Inference class: src/perception_nodes/perception_nodes/depth_anything_v2_trt.py"
}

# Main execution
main() {
    echo -e "${BLUE}=== Starting Depth Anything V2 Environment Validation ===${NC}"

    # Validate dependencies (don't exit on failure)
    if ! validate_dependencies; then
        echo -e "${RED}Environment validation failed. Please install missing dependencies.${NC}"
        exit 1
    fi

    # Check disk space
    if ! check_space; then
        echo -e "${RED}Insufficient disk space for conversion.${NC}"
        exit 1
    fi

    # Run model conversion
    if ! run_conversion; then
        echo -e "${RED}Model conversion failed.${NC}"
        exit 1
    fi

    # Validate results
    validate_conversion

    # Print usage information
    print_usage

    echo -e "${GREEN}🎉 Depth Anything V2 setup completed!${NC}"
}
        echo -e "${RED}❌ Setup completed with warnings. Check the output above.${NC}"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo "Validates environment and sets up Depth Anything V2 for Jetson deployment"
        echo ""
        echo "This script will:"
        echo "  - Validate Python dependencies (NO INSTALLATION)"
        echo "  - Check system requirements"
        echo "  - Download and convert Depth Anything V2 Small model"
        echo "  - Create ONNX and TensorRT formats"
        echo "  - Validate the conversion"
        echo ""
        echo "Note: Uses existing pyproject.toml environment"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
esac
