"""
RT-MonoDepth Standalone Inference Class
Supports both PyTorch and TensorRT models
Uses RTMonoDepthModel wrapper for cleaner architecture
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from .rt_monodepth_model import RTMonoDepthModel
from .rt_monodepth_preprocessing import RTMonoDepthPreprocessor


class RTMonoDepthInference:
    """Standalone inference wrapper for RT-MonoDepth model."""

    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = False,
        model_variant: str = "small",
        input_height: int = 192,
        input_width: int = 640,
        device: str = "cuda",
    ):
        """
        Initialize RT-MonoDepth inference engine.

        Args:
            model_path: Path to model weights (.pth) or TensorRT engine (.engine)
            use_tensorrt: Whether to use TensorRT for inference
            model_variant: Model size - 'small' or 'full' (for PyTorch only)
            input_height: Model input height
            input_width: Model input width
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.use_tensorrt = use_tensorrt and TRT_AVAILABLE
        self.model_variant = model_variant
        self.device = device
        self.input_height = input_height
        self.input_width = input_width

        # Initialize preprocessor
        self.preprocessor = RTMonoDepthPreprocessor(
            input_height=input_height, input_width=input_width, normalize=True
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Load model
        if self.use_tensorrt:
            self._load_tensorrt_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """Load PyTorch model using RTMonoDepthModel wrapper."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.logger.info(f"Loading PyTorch model from {self.model_path}")

        # Use the model wrapper for cleaner architecture
        self.model = RTMonoDepthModel(
            model_variant=self.model_variant,
            device=self.device,
            pretrained_path=str(self.model_path),
        )

        self.logger.info("PyTorch model loaded successfully")

        # Print model info (optional, can be disabled)
        # self.model.print_model_info()

    def _load_tensorrt_model(self):
        """Load TensorRT engine for optimized inference."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Install tensorrt and pycuda.")

        if not self.model_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found at {self.model_path}")

        self.logger.info(f"Loading TensorRT engine from {self.model_path}")

        # Create TensorRT logger and runtime
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        # Load engine
        with open(self.model_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self._allocate_trt_buffers()

        self.logger.info("TensorRT engine loaded successfully")

    def _allocate_trt_buffers(self):
        """Allocate GPU memory for TensorRT inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def predict(
        self,
        image_path: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
        return_tensor: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Run depth prediction on input image.

        Args:
            image_path: Path to image file
            image_array: NumPy array image
            return_tensor: If True, return torch.Tensor instead of numpy array

        Returns:
            Tuple of (depth_map, original_shape)
        """
        # Preprocess image
        input_tensor, original_shape = self.preprocessor.preprocess_image(
            image_path=image_path, image_array=image_array
        )

        # Run inference
        if self.use_tensorrt:
            depth_output = self._infer_tensorrt(input_tensor)
        else:
            depth_output = self._infer_pytorch(input_tensor)

        # Postprocess
        if return_tensor:
            # Return raw tensor without postprocessing
            return depth_output, original_shape
        else:
            depth_map = self.preprocessor.postprocess_depth(depth_output, original_shape)
            return depth_map, original_shape

    def _infer_pytorch(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using PyTorch model."""
        input_tensor = input_tensor.to(self.device)

        # Use the model wrapper's forward method
        model_output = self.model(input_tensor)

        # Extract depth tensor from output
        depth_output = self.model.get_output_tensor(model_output)

        return depth_output

    def _infer_tensorrt(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run inference using TensorRT engine."""
        # Convert tensor to numpy and copy to input buffer
        input_numpy = input_tensor.cpu().numpy().ravel()
        np.copyto(self.inputs[0]["host"], input_numpy)

        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)

        # Synchronize
        self.stream.synchronize()

        # Reshape output
        output_shape = (1, 1, self.input_height, self.input_width)
        depth_output = torch.from_numpy(self.outputs[0]["host"].reshape(output_shape))

        return depth_output

    def predict_batch(self, images: list, batch_size: int = 4) -> list:
        """
        Run batch prediction on multiple images.

        Args:
            images: List of image paths or numpy arrays
            batch_size: Number of images to process at once

        Returns:
            List of (depth_map, original_shape) tuples
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            for img in batch:
                if isinstance(img, str):
                    result = self.predict(image_path=img)
                else:
                    result = self.predict(image_array=img)
                results.append(result)

        return results

    def benchmark(self, num_iterations: int = 100) -> dict:
        """
        Benchmark inference performance.

        Args:
            num_iterations: Number of iterations to run

        Returns:
            Dictionary with timing statistics
        """
        import time

        # Create dummy input
        dummy_input = torch.randn(1, 3, self.input_height, self.input_width)

        if not self.use_tensorrt:
            dummy_input = dummy_input.to(self.device)

        # Warmup
        self.logger.info("Warming up...")
        for _ in range(10):
            if self.use_tensorrt:
                _ = self._infer_tensorrt(dummy_input)
            else:
                _ = self._infer_pytorch(dummy_input)

        # Benchmark
        self.logger.info(f"Running benchmark ({num_iterations} iterations)...")
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()

            if self.use_tensorrt:
                _ = self._infer_tensorrt(dummy_input)
            else:
                _ = self._infer_pytorch(dummy_input)

            if self.device == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "fps": 1000 / np.mean(times),
            "backend": "TensorRT" if self.use_tensorrt else "PyTorch",
            "device": self.device,
        }

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": str(self.model_path),
            "backend": "TensorRT" if self.use_tensorrt else "PyTorch",
            "device": self.device,
            "input_size": f"{self.input_width}x{self.input_height}",
        }

        if not self.use_tensorrt and hasattr(self, "model"):
            info.update(self.model.get_model_info())

        return info

    def print_info(self):
        """Print model and inference information."""
        info = self.get_model_info()
        print("\n" + "=" * 60)
        print("RT-MonoDepth Inference Engine")
        print("=" * 60)
        print(f"Model Path:    {info['model_path']}")
        print(f"Backend:       {info['backend']}")
        print(f"Device:        {info['device']}")
        print(f"Input Size:    {info['input_size']}")

        if "variant" in info:
            print(f"Model Variant: {info['variant']}")
            print(f"Parameters:    {info['total_parameters']:,}")
            print(f"Model Size:    {info['model_size_mb']:.2f} MB")

        print("=" * 60 + "\n")
