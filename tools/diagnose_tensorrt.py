#!/usr/bin/env python3
"""
TensorRT Diagnostic Script
Checks memory, TensorRT version, and creates a simple test with modern API
"""

import gc
import sys

import psutil


def check_system_memory():
    """Check available system memory"""
    print("=== System Memory Check ===")
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")

    if memory.available < 1024**3:  # Less than 1GB available
        print("⚠️  WARNING: Low available memory - this may cause allocation issues")
        return False
    return True


def check_gpu_memory():
    """Check GPU memory if nvidia-ml-py is available"""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print("=== GPU Memory Check ===")
        print(f"Total GPU RAM: {info.total / (1024**3):.1f} GB")
        print(f"Free GPU RAM: {info.free / (1024**3):.1f} GB")
        print(f"Used GPU RAM: {info.used / (1024**3):.1f} GB")

        if info.free < 500 * 1024**2:  # Less than 500MB free
            print("⚠️  WARNING: Low GPU memory available")
            return False
        return True
    except ImportError:
        print("=== GPU Memory Check ===")
        print("nvidia-ml-py not available - cannot check GPU memory")
        return True
    except Exception as e:
        print(f"GPU memory check failed: {e}")
        return True


def test_tensorrt_modern_api():
    """Test TensorRT with modern API (8.5+ compatible)"""
    print("=== TensorRT Modern API Test ===")

    try:
        import tensorrt as trt

        print(f"TensorRT version: {trt.__version__}")

        # Create logger
        logger = trt.Logger(trt.Logger.WARNING)

        # Create builder with modern API
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        # Use modern memory pool API instead of deprecated max_workspace_size
        if hasattr(config, "set_memory_pool_limit"):
            # TensorRT 8.5+ API
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1MB
            print("✓ Using modern memory pool API")
        elif hasattr(config, "max_workspace_size"):
            # Fallback for older versions
            config.max_workspace_size = 1 << 20  # 1MB
            print("✓ Using legacy workspace size API")
        else:
            print("⚠️  Cannot set workspace size - unknown API version")

        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Add a simple identity layer
        input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 1))
        identity = network.add_identity(input_tensor)
        network.mark_output(identity.get_output(0))

        print("✓ Created simple TensorRT network")

        # Try to build (this is where memory errors often occur)
        try:
            # TensorRT 10.x uses build_serialized_network instead of build_engine
            if hasattr(builder, "build_serialized_network"):
                # TensorRT 10.x API
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is not None:
                    print("✓ Successfully built TensorRT serialized network")
                    # Test deserializing
                    runtime = trt.Runtime(logger)
                    engine = runtime.deserialize_cuda_engine(serialized_engine)
                    if engine is not None:
                        print("✓ Successfully deserialized TensorRT engine")
                        return True
                    else:
                        print("✗ Failed to deserialize TensorRT engine")
                        return False
                else:
                    print("✗ Failed to build TensorRT serialized network (returned None)")
                    return False
            elif hasattr(builder, "build_engine"):
                # TensorRT 8.x API
                engine = builder.build_engine(network, config)
                if engine is not None:
                    print("✓ Successfully built TensorRT engine")
                    return True
                else:
                    print("✗ Failed to build TensorRT engine (returned None)")
                    return False
            else:
                print("✗ Unknown TensorRT API version - cannot build engine")
                return False
        except Exception as e:
            print(f"✗ Failed to build TensorRT engine: {e}")
            return False

    except ImportError as e:
        print(f"✗ TensorRT import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ TensorRT test failed: {e}")
        return False


def main():
    """Run all diagnostic checks"""
    print("TensorRT Diagnostic Tool")
    print("=" * 50)

    # Force garbage collection
    gc.collect()

    # Check system resources
    memory_ok = check_system_memory()
    gpu_ok = check_gpu_memory()

    print()

    # Test TensorRT
    tensorrt_ok = test_tensorrt_modern_api()

    print()
    print("=== Summary ===")
    if memory_ok and gpu_ok and tensorrt_ok:
        print("🎉 All checks passed!")
        return 0
    else:
        print("❌ Some issues detected:")
        if not memory_ok:
            print("  - Low system memory")
        if not gpu_ok:
            print("  - GPU memory issues")
        if not tensorrt_ok:
            print("  - TensorRT API issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
