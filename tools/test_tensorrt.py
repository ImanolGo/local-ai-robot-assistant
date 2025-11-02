#!/usr/bin/env python3
"""
Simple TensorRT Installation Test
Tests basic TensorRT Python bindings without complex dependencies
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_tensorrt_import():
    """Test TensorRT Python bindings import"""
    try:
        import tensorrt as trt

        logger.info(f"✓ TensorRT imported successfully")
        logger.info(f"TensorRT version: {trt.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import TensorRT: {e}")
        return False


def test_tensorrt_logger():
    """Test TensorRT logger creation"""
    try:
        import tensorrt as trt

        logger_obj = trt.Logger(trt.Logger.WARNING)
        logger.info("✓ TensorRT Logger created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create TensorRT Logger: {e}")
        return False


def test_tensorrt_builder():
    """Test TensorRT builder creation"""
    try:
        import tensorrt as trt

        logger_obj = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_obj)
        logger.info("✓ TensorRT Builder created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create TensorRT Builder: {e}")
        return False


def test_tensorrt_runtime():
    """Test TensorRT runtime creation"""
    try:
        import tensorrt as trt

        logger_obj = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger_obj)
        logger.info("✓ TensorRT Runtime created successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to create TensorRT Runtime: {e}")
        return False


def test_basic_dependencies():
    """Test basic dependencies for model conversion"""
    dependencies = [
        ("torch", "PyTorch"),
        ("onnx", "ONNX"),
        ("numpy", "NumPy"),
        ("psutil", "PSUtil"),
    ]

    results = []
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name} available")
            results.append(True)
        except ImportError:
            logger.warning(f"⚠ {display_name} not available")
            results.append(False)

    return all(results)


def main():
    """Run all tests"""
    logger.info("=== TensorRT Installation Test ===")
    logger.info("")

    tests = [
        ("TensorRT Import", test_tensorrt_import),
        ("TensorRT Logger", test_tensorrt_logger),
        ("TensorRT Builder", test_tensorrt_builder),
        ("TensorRT Runtime", test_tensorrt_runtime),
        ("Basic Dependencies", test_basic_dependencies),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: FAILED - {e}")
            results.append(False)
        logger.info("")

    # Summary
    total_tests = len(tests)
    passed_tests = sum(results)

    logger.info("=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")

    if all(results):
        logger.info("🎉 All tests PASSED! TensorRT is ready for model conversion.")
        return 0
    else:
        logger.error("❌ Some tests FAILED. Check installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
