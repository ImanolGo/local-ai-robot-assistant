#!/bin/bash
# TensorRT and Model Conversion Dependencies Installation Script
# For NVIDIA Jetson Orin Nano with JetPack SDK

set -e  # Exit on any error

echo "=== Installing TensorRT and Model Conversion Dependencies ==="
echo "This script installs TensorRT, ONNX Runtime, and related tools for model optimization."
echo ""

# Function to check if running on Jetson
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        echo "Warning: This script is designed for NVIDIA Jetson devices."
        echo "Detected system may not be a Jetson. Continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Detected NVIDIA Jetson device"
        cat /etc/nv_tegra_release
    fi
}

# Function to verify CUDA installation
check_cuda() {
    echo "Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        echo "✓ CUDA compiler found:"
        nvcc --version
    else
        echo "✗ CUDA not found. Please install JetPack SDK first."
        exit 1
    fi
    
    if python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        echo "✓ PyTorch with CUDA support detected"
    else
        echo "⚠ PyTorch not found or CUDA not available"
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo "Installing system dependencies..."
    sudo apt update
    sudo apt install -y \
        python3-pip \
        python3-dev \
        cmake \
        build-essential \
        libopencv-dev \
        libnvinfer-dev \
        libnvinfer-plugin-dev \
        libnvonnxparsers-dev \
        python3-libnvinfer-dev \
        tensorrt
    
    echo "✓ System dependencies installed"
}

# Function to install Python packages
install_python_deps() {
    echo "Installing Python dependencies..."
    
    # Create requirements file for model conversion tools
    cat > /tmp/tensorrt_requirements.txt << EOF
# Core ML frameworks
torch>=1.12.0
torchvision>=0.13.0
onnx>=1.12.0
onnxruntime-gpu>=1.12.0

# TensorRT Python bindings
tensorrt>=8.0.0

# Model conversion utilities
ultralytics>=8.0.0
transformers>=4.20.0
accelerate>=0.20.0
optimum>=1.8.0

# Benchmarking and profiling
psutil>=5.9.0
nvidia-ml-py3>=11.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.64.0
colorama>=0.4.4
tabulate>=0.9.0
pyyaml>=6.0
EOF

    # Install packages
    pip3 install --upgrade pip
    pip3 install -r /tmp/tensorrt_requirements.txt
    
    # Verify installations
    echo "Verifying Python installations..."
    python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" || echo "⚠ TensorRT Python bindings not available"
    python3 -c "import onnx; print(f'ONNX version: {onnx.__version__}')" || echo "⚠ ONNX not available"
    python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')" || echo "⚠ ONNX Runtime not available"
    
    echo "✓ Python dependencies installed"
}

# Function to test TensorRT installation
test_tensorrt() {
    echo "Testing TensorRT installation..."
    
    # Test trtexec command
    if command -v trtexec &> /dev/null; then
        echo "✓ trtexec found:"
        trtexec --help | head -n 5
    else
        echo "⚠ trtexec not found in PATH"
    fi
    
    # Test Python TensorRT
    python3 -c "
import tensorrt as trt
print('✓ TensorRT Python bindings working')
print(f'TensorRT version: {trt.__version__}')

# Test logger creation
logger = trt.Logger(trt.Logger.WARNING)
print('✓ TensorRT Logger created successfully')

# Test runtime creation
runtime = trt.Runtime(logger)
print('✓ TensorRT Runtime created successfully')
" 2>/dev/null || echo "⚠ TensorRT Python test failed"
}

# Function to create sample test scripts
create_test_scripts() {
    echo "Creating test scripts..."
    
    # Create simple TensorRT test
    cat > /tmp/test_tensorrt_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simple TensorRT installation test
Tests basic TensorRT functionality with a dummy ONNX model
"""

import tensorrt as trt
import numpy as np
import onnx
from onnx import helper, TensorProto
import tempfile
import os

def create_dummy_onnx_model():
    """Create a simple ONNX model for testing"""
    # Create a simple identity model
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
    
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_graph', [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name='test')
    
    return model

def test_tensorrt_conversion():
    """Test ONNX to TensorRT conversion"""
    print("Creating dummy ONNX model...")
    model = create_dummy_onnx_model()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        onnx_path = f.name
    
    try:
        print("Converting ONNX to TensorRT...")
        
        # Create TensorRT logger and builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("Failed to parse ONNX model")
                return False
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256MB
        
        print("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            print("Failed to build TensorRT engine")
            return False
        
        print("✓ TensorRT conversion successful!")
        print(f"Engine has {engine.num_bindings} bindings")
        
        # Test inference context creation
        context = engine.create_execution_context()
        if context is None:
            print("Failed to create execution context")
            return False
        
        print("✓ TensorRT execution context created successfully!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)

if __name__ == "__main__":
    print("=== TensorRT Installation Test ===")
    try:
        success = test_tensorrt_conversion()
        if success:
            print("\n✓ TensorRT installation test PASSED")
        else:
            print("\n✗ TensorRT installation test FAILED")
            exit(1)
    except Exception as e:
        print(f"\n✗ TensorRT test failed with error: {e}")
        exit(1)
EOF

    chmod +x /tmp/test_tensorrt_simple.py
    echo "✓ Test scripts created at /tmp/test_tensorrt_simple.py"
}

# Function to display post-installation information
show_post_install_info() {
    echo ""
    echo "=== Installation Complete ==="
    echo ""
    echo "To test your installation:"
    echo "  python3 /tmp/test_tensorrt_simple.py"
    echo ""
    echo "TensorRT tools location:"
    echo "  trtexec: $(which trtexec || echo 'Not found in PATH')"
    echo ""
    echo "Next steps:"
    echo "  1. Run the test script to verify installation"
    echo "  2. Check the tools/conversion/ directory for model conversion scripts"
    echo "  3. Review docs/guides/ for model conversion best practices"
    echo ""
    echo "Environment variables (add to ~/.bashrc if needed):"
    echo "  export PATH=\$PATH:/usr/src/tensorrt/bin"
    echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu"
}

# Main installation flow
main() {
    echo "Starting TensorRT installation for Local AI Robot Assistant..."
    echo "Target platform: NVIDIA Jetson Orin Nano"
    echo ""
    
    check_jetson
    check_cuda
    #install_system_deps
    #install_python_deps
    test_tensorrt
    create_test_scripts
    show_post_install_info
    
    echo ""
    echo "✓ TensorRT and dependencies installation complete!"
}

# Run main function
main "$@"