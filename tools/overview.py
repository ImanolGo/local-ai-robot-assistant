#!/usr/bin/env python3
"""
Model Conversion Tools Overview
Summary script for all available model conversion utilities
"""

from pathlib import Path


def print_header():
    """Print header information"""
    print("=" * 80)
    print("MODEL CONVERSION TOOLS FOR LOCAL AI ROBOT ASSISTANT")
    print("NVIDIA Jetson Orin Nano Deployment")
    print("=" * 80)
    print()


def print_available_tools():
    """Print available conversion tools"""
    tools_dir = Path(__file__).parent

    tools = [
        {
            "name": "TensorRT Installation",
            "script": "install_tensorrt.sh",
            "description": "Install TensorRT, ONNX Runtime, and dependencies",
            "usage": "./tools/install_tensorrt.sh",
        },
        {
            "name": "YOLO Conversion",
            "script": "conversion/convert_yolo.py",
            "description": "Convert YOLOv8 models to TensorRT FP16",
            "usage": "python3 tools/conversion/convert_yolo.py\
                  --model yolov8n --output-dir ./models/yolo_trt",
        },
        {
            "name": "FastDepth Conversion",
            "script": "conversion/convert_depth.py",
            "description": "Convert FastDepth models to TensorRT FP16",
            "usage": "python3 tools/conversion/convert_depth.py --output-dir ./models/depth_trt",
        },
        {
            "name": "Whisper Conversion",
            "script": "conversion/convert_whisper.py",
            "description": "Convert Whisper to faster-whisper or TensorRT",
            "usage": "python3 tools/conversion/convert_whisper.py --model-size tiny\
                  --conversion-type faster-whisper",
        },
        {
            "name": "Model Profiling",
            "script": "benchmarking/profile_model.py",
            "description": "Benchmark model performance and resource usage",
            "usage": "python3 tools/benchmarking/profile_model.py --models-dir ./models",
        },
        {
            "name": "Conversion Pipeline Template",
            "script": "utils/conversion_pipeline.py",
            "description": "Standardized conversion framework",
            "usage": "python3 tools/utils/conversion_pipeline.py --model-type yolo",
        },
        {
            "name": "TensorRT Test",
            "script": "test_tensorrt.py",
            "description": "Test TensorRT installation and basic functionality",
            "usage": "python3 tools/test_tensorrt.py",
        },
    ]

    print("AVAILABLE TOOLS:")
    print("-" * 80)

    for i, tool in enumerate(tools, 1):
        script_path = tools_dir / tool["script"]
        status = "✓" if script_path.exists() else "✗"

        print(f"{i}. {tool['name']} {status}")
        print(f"   Script: {tool['script']}")
        print(f"   Description: {tool['description']}")
        print(f"   Usage: {tool['usage']}")
        print()


def print_architecture_targets():
    """Print performance targets from architecture"""
    print("PERFORMANCE TARGETS (from architecture.md):")
    print("-" * 80)

    targets = [
        ("YOLO Object Detection", "20+ FPS", "640x480", "600 MB RAM"),
        ("FastDepth Estimation", "15+ FPS", "320x240", "400 MB RAM"),
        ("Whisper Speech Recognition", "RTF < 0.3x", "Audio", "500 MB RAM"),
        ("NanoLLM Inference", "< 3s latency", "Text", "2.5 GB RAM"),
    ]

    for model, fps, resolution, memory in targets:
        print(f"• {model:<25} | {fps:<12} | {resolution:<10} | {memory}")
    print()


def print_conversion_pipeline():
    """Print the standardized conversion pipeline"""
    print("CONVERSION PIPELINE:")
    print("-" * 80)
    print("PyTorch/HuggingFace → ONNX (intermediate) → TensorRT Engine (deployment)")
    print()
    print("Steps:")
    print("1. Download or load PyTorch model")
    print("2. Export to ONNX format with optimization")
    print("3. Convert ONNX to TensorRT FP16 engine")
    print("4. Benchmark performance and validate targets")
    print("5. Deploy in ROS2 perception/audio nodes")
    print()


def print_quick_start():
    """Print quick start guide"""
    print("QUICK START GUIDE:")
    print("-" * 80)
    print("1. Install dependencies:")
    print("   ./tools/install_tensorrt.sh")
    print()
    print("2. Test TensorRT installation:")
    print("   python3 tools/test_tensorrt.py")
    print()
    print("3. Convert your first model (YOLO):")
    print("   python3 tools/conversion/convert_yolo.py \\")
    print("     --model yolov8n \\")
    print("     --output-dir ./models/yolo_trt \\")
    print("     --input-size 640 480")
    print()
    print("4. Profile model performance:")
    print("   python3 tools/benchmarking/profile_model.py \\")
    print("     --models-dir ./models/yolo_trt")
    print()
    print("5. Check documentation:")
    print("   docs/guides/model_conversion_best_practices.md")
    print()


def print_memory_strategy():
    """Print memory management strategy"""
    print("MEMORY MANAGEMENT STRATEGY (8GB Total):")
    print("-" * 80)
    print("• Perception Mode (default):    ~4.5 GB (SLAM + YOLO + Depth)")
    print("• Reasoning Mode (complex):     ~5.5 GB (+ LLM, unload vision)")
    print("• Emergency Mode (critical):    ~2.0 GB (motors + wake word only)")
    print()
    print("Dynamic Loading:")
    print("• Load LLM only when complex reasoning needed")
    print("• Unload models at 85% RAM usage threshold")
    print("• Use 16GB swap file on NVMe for overflow")
    print()


def main():
    """Main function"""
    print_header()
    print_available_tools()
    print_architecture_targets()
    print_conversion_pipeline()
    print_memory_strategy()
    print_quick_start()

    print("DOCUMENTATION:")
    print("-" * 80)
    print("• Architecture: docs/architecture.md")
    print("• Best Practices: docs/guides/model_conversion_best_practices.md")
    print("• Implementation Plan: docs/implementation_plan.md")
    print("• Project Status: STATUS.md")
    print()

    print("For support and updates, check:")
    print("https://github.com/your-repo/local-ai-robot-assistant")
    print()


if __name__ == "__main__":
    main()
