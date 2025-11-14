#!/usr/bin/env python3
"""
Depth Anything V2 TensorRT Inference Class
Optimized inference implementation for Jetson Orin Nano

Provides high-performance depth estimation using TensorRT engine
with proper memory management and preprocessing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

logger = logging.getLogger(__name__)


class DepthAnythingV2TRT:
    """
    TensorRT-optimized inference for Depth Anything V2 Small

    Provides efficient depth estimation with proper memory management
    and preprocessing for Jetson Orin Nano deployment.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize TensorRT depth estimation model

        Args:
            engine_path: Path to TensorRT engine file
            config_path: Optional path to model configuration JSON
        """
        self.engine_path = Path(engine_path)
        self.config_path = Path(config_path) if config_path else None

        # Load configuration
        self.config = self._load_config()

        # TensorRT components
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None

        # Memory buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        # Model parameters
        self.input_shape = self.config.get("input_shape", [1, 3, 518, 518])
        self.output_shape = self.config.get("output_shape", [1, 518, 518])
        self.input_size = (self.input_shape[3], self.input_shape[2])  # (width, height)

        # Preprocessing parameters
        preproc = self.config.get("preprocessing", {})
        self.normalize_mean = np.array(preproc.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.normalize_std = np.array(preproc.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        # Performance tracking
        self.inference_times = []

        # Initialize TensorRT engine
        self._load_engine()
        self._allocate_buffers()

        logger.info(f"DepthAnythingV2TRT initialized with input size: {self.input_size}")

    def _load_config(self) -> Dict:
        """Load model configuration"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from: {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Default configuration
        return {
            "input_shape": [1, 3, 518, 518],
            "output_shape": [1, 518, 518],
            "preprocessing": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": [518, 518],
            },
        }

    def _load_engine(self) -> None:
        """Load TensorRT engine from file"""
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        try:
            # Create runtime
            self.runtime = trt.Runtime(self.logger)

            # Load engine
            with open(self.engine_path, "rb") as f:
                engine_data = f.read()

            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            # Create execution context
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context")

            logger.info(f"TensorRT engine loaded successfully from: {self.engine_path}")

        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise

    def _allocate_buffers(self) -> None:
        """Allocate GPU memory buffers for inference"""
        try:
            # Create CUDA stream for async operations
            self.stream = cuda.Stream()

            # Allocate buffers for each binding
            for binding in self.engine:
                # Get binding index
                _ = self.engine.get_binding_index(binding)

                # Get binding shape and data type
                if self.engine.binding_is_input(binding):
                    shape = self.input_shape
                else:
                    shape = self.output_shape

                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))

                # Allocate host and device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                # Add to bindings list
                self.bindings.append(int(device_mem))

                # Store in appropriate list
                if self.engine.binding_is_input(binding):
                    self.inputs.append(
                        {
                            "host": host_mem,
                            "device": device_mem,
                            "shape": shape,
                            "dtype": dtype,
                        }
                    )
                    logger.info(f"Input buffer allocated: shape={shape}, dtype={dtype}")
                else:
                    self.outputs.append(
                        {
                            "host": host_mem,
                            "device": device_mem,
                            "shape": shape,
                            "dtype": dtype,
                        }
                    )
                    logger.info(f"Output buffer allocated: shape={shape}, dtype={dtype}")

        except Exception as e:
            logger.error(f"Failed to allocate buffers: {e}")
            raise

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for depth estimation

        Args:
            image: Input image (BGR format, HWC)

        Returns:
            Preprocessed image tensor (CHW format)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize to model input size
        image_resized = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize to [0, 1]
        image_float = image_resized.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        image_normalized = (image_float - self.normalize_mean) / self.normalize_std

        # Convert from HWC to CHW format
        image_chw = image_normalized.transpose(2, 0, 1)

        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0).astype(np.float32)

        return image_batch

    def postprocess(
        self, depth_output: np.ndarray, original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Postprocess depth estimation output

        Args:
            depth_output: Raw model output
            original_size: Optional (width, height) to resize depth map

        Returns:
            Processed depth map
        """
        # Remove batch dimension if present
        if len(depth_output.shape) == 3:
            depth_map = depth_output[0]
        else:
            depth_map = depth_output

        # Ensure positive depth values
        depth_map = np.maximum(depth_map, 0.0)

        # Resize to original image size if specified
        if original_size is not None:
            depth_map = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LINEAR)

        return depth_map

    def infer(self, image: np.ndarray, return_original_size: bool = True) -> np.ndarray:
        """
        Run depth estimation inference

        Args:
            image: Input image (BGR format, HWC)
            return_original_size: Whether to resize output to original image size

        Returns:
            Depth map (same size as input if return_original_size=True)
        """
        start_time = time.time()

        try:
            # Store original size for postprocessing
            original_size = (image.shape[1], image.shape[0]) if return_original_size else None

            # Preprocess image
            input_tensor = self.preprocess(image)

            # Copy input data to device
            np.copyto(self.inputs[0]["host"], input_tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            # Copy output data to host
            cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)

            # Wait for completion
            self.stream.synchronize()

            # Reshape output
            output_data = self.outputs[0]["host"].reshape(self.output_shape)

            # Postprocess
            depth_map = self.postprocess(output_data, original_size)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only last 100 measurements for moving average
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            return depth_map

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get inference performance statistics

        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {"avg_inference_time": 0.0, "avg_fps": 0.0}

        avg_time = np.mean(self.inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            "avg_inference_time": avg_time,
            "avg_fps": avg_fps,
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "num_inferences": len(self.inference_times),
        }

    def visualize_depth(
        self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
        """
        Create colored visualization of depth map

        Args:
            depth_map: Input depth map
            colormap: OpenCV colormap for visualization

        Returns:
            Colored depth visualization (BGR format)
        """
        # Normalize depth map to 0-255 range
        depth_normalized = depth_map.copy()

        # Handle edge case where all depths are the same
        depth_range = depth_normalized.max() - depth_normalized.min()
        if depth_range > 0:
            depth_normalized = (depth_normalized - depth_normalized.min()) / depth_range
        else:
            depth_normalized = np.zeros_like(depth_normalized)

        # Convert to 8-bit
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)

        return depth_colored

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Free CUDA memory
            for inp in self.inputs:
                if "device" in inp:
                    inp["device"].free()

            for out in self.outputs:
                if "device" in out:
                    out["device"].free()

            # Destroy context and engine
            if self.context:
                del self.context
            if self.engine:
                del self.engine
            if self.runtime:
                del self.runtime

            logger.info("TensorRT resources cleaned up")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Convenience function for standalone usage
def create_depth_estimator(
    engine_path: Union[str, Path], config_path: Optional[Union[str, Path]] = None
) -> DepthAnythingV2TRT:
    """
    Create and initialize depth estimation model

    Args:
        engine_path: Path to TensorRT engine file
        config_path: Optional path to model configuration

    Returns:
        Initialized DepthAnythingV2TRT instance
    """
    return DepthAnythingV2TRT(engine_path, config_path)


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Depth Anything V2 TensorRT Inference Demo")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--config", help="Path to model configuration JSON")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save depth visualization")

    args = parser.parse_args()

    # Initialize model
    depth_model = DepthAnythingV2TRT(args.engine, args.config)

    # Load test image
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Could not load image: {args.image}")
        exit(1)

    print(f"Input image shape: {image.shape}")

    # Run inference
    depth_map = depth_model.infer(image)
    print(f"Output depth map shape: {depth_map.shape}")
    print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")

    # Get performance stats
    stats = depth_model.get_performance_stats()
    print(f"Inference time: {stats['avg_inference_time']*1000:.1f} ms")
    print(f"FPS: {stats['avg_fps']:.1f}")

    # Create visualization
    depth_colored = depth_model.visualize_depth(depth_map)

    # Save output if specified
    if args.output:
        cv2.imwrite(args.output, depth_colored)
        print(f"Depth visualization saved to: {args.output}")
    else:
        # Display result
        cv2.imshow("Original", image)
        cv2.imshow("Depth", depth_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Cleanup
    depth_model.cleanup()
