"""
Convert RT-MonoDepth PyTorch model to TensorRT
Supports both torch2trt and native TensorRT export
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "perception_nodes"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_with_tensorrt(model, dummy_input, output_path, fp16_mode=True):
    """
    Convert model using native TensorRT (more control, more complex).

    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        output_path: Path to save TensorRT engine
        fp16_mode: Use FP16 precision
    """
    try:
        import tensorrt as trt
        from torch.onnx import export as onnx_export
    except ImportError:
        raise ImportError("TensorRT not installed")

    logger.info("Converting model with native TensorRT...")

    # Step 1: Export to ONNX
    onnx_path = output_path.parent / f"{output_path.stem}.onnx"
    logger.info(f"Exporting to ONNX: {onnx_path}")

    onnx_export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Step 2: Build TensorRT engine from ONNX
    logger.info("Building TensorRT engine from ONNX...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            logger.error("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return None

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled")

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return None

    # Save engine
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to {output_path}")

    # Clean up ONNX file if desired
    # onnx_path.unlink()

    return serialized_engine


def load_rtmonodepth_model(weight_path, model_variant="small", device="cuda"):
    """
    Load RT-MonoDepth model from checkpoint using model wrapper.

    Args:
        weight_path: Path to model weights
        model_variant: 'small' or 'full'
        device: Device to load model on

    Returns:
        Loaded RTMonoDepthModel instance
    """
    from perception_nodes.depth.rt_monodepth_model import RTMonoDepthModel

    logger.info(f"Loading PyTorch model from {weight_path}")

    # Use the model wrapper for cleaner loading
    model = RTMonoDepthModel(
        model_variant=model_variant, device=device, pretrained_path=weight_path
    )

    logger.info("Model loaded successfully")
    model.print_model_info()

    return model


def test_tensorrt_engine(engine_path, dummy_input, input_height, input_width):
    """
    Test a TensorRT engine with dummy input.

    Args:
        engine_path: Path to saved .engine file
        dummy_input: PyTorch tensor input
        input_height: Input height
        input_width: Input width

    Returns:
        numpy array output or None if failed
    """
    try:
        import numpy as np
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError as e:
        logger.error(f"Missing dependency for TensorRT testing: {e}")
        return None

    # Load engine
    logger.info(f"Loading TensorRT engine from {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        logger.error("Failed to load TensorRT engine")
        return None

    context = engine.create_execution_context()

    # Prepare input/output buffers
    input_np = dummy_input.cpu().numpy()

    # Allocate device memory
    d_input = cuda.mem_alloc(input_np.nbytes)

    # Get output shape (assuming single output)
    output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
    output_np = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_np.nbytes)

    # Set tensor addresses
    context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

    # Copy input to device
    cuda.memcpy_htod(d_input, input_np)

    # Run inference
    context.execute_async_v3(cuda.Stream().handle)

    # Copy output back to host
    cuda.memcpy_dtoh(output_np, d_output)

    logger.info("TensorRT inference successful")

    return output_np


def main():
    parser = argparse.ArgumentParser(description="Convert RT-MonoDepth to TensorRT")
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="Path to PyTorch model weights (.pth)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save TensorRT engine (.engine)",
    )

    parser.add_argument("--input_height", type=int, default=192, help="Model input height")
    parser.add_argument("--input_width", type=int, default=640, help="Model input width")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device for conversion")
    parser.add_argument(
        "--test_inference", action="store_true", help="Test inference after conversion"
    )

    args = parser.parse_args()

    # Validate paths
    weight_path = Path(args.weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    model = load_rtmonodepth_model(weight_path, args.device)

    # Create dummy input
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width).to(args.device)

    # Test PyTorch inference first
    logger.info("Testing PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)

    output_shape = pytorch_output.shape if isinstance(pytorch_output, torch.Tensor) else "dict"
    logger.info(f"PyTorch output shape: {output_shape}")

    # Convert model
    model_trt = convert_with_tensorrt(model, dummy_input, output_path, args.fp16)

    # Test TensorRT inference if requested
    if args.test_inference:
        if args.method == "torch2trt":
            logger.info("Testing TensorRT inference...")
            with torch.no_grad():
                trt_output = model_trt(dummy_input)
            logger.info(f"TensorRT output shape: {trt_output.shape}")

            # Compare outputs
            if isinstance(pytorch_output, torch.Tensor):
                diff = torch.abs(pytorch_output - trt_output).mean()
                logger.info(f"Mean absolute difference: {diff.item():.6f}")
            else:
                logger.info("Skipping output comparison (dict output)")

        elif args.method == "tensorrt":
            logger.info("Testing TensorRT inference...")
            trt_output = test_tensorrt_engine(
                output_path, dummy_input, args.input_height, args.input_width
            )

            if trt_output is not None:
                logger.info(f"TensorRT output shape: {trt_output.shape}")

                # Compare outputs
                if isinstance(pytorch_output, torch.Tensor):
                    # Convert back to torch tensor for comparison
                    trt_output_torch = torch.from_numpy(trt_output).to(args.device)
                    diff = torch.abs(pytorch_output - trt_output_torch).mean()
                    logger.info(f"Mean absolute difference: {diff.item():.6f}")
                else:
                    logger.info("Skipping output comparison (dict output)")

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
