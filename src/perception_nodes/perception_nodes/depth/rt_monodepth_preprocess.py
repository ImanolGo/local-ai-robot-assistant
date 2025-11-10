"""
RT-MonoDepth Image Preprocessing Utilities
Adapted from RT-MonoDepth repository utils.py
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


class RTMonoDepthPreprocessor:
    """Handles image preprocessing for RT-MonoDepth model inference."""

    def __init__(self, input_height: int = 192, input_width: int = 640, normalize: bool = True):
        """
        Initialize preprocessor with target dimensions.

        Args:
            input_height: Target height for model input
            input_width: Target width for model input
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.input_height = input_height
        self.input_width = input_width
        self.normalize = normalize

    def preprocess_image(
        self, image_path: Optional[str] = None, image_array: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for RT-MonoDepth inference.

        Args:
            image_path: Path to image file (if loading from disk)
            image_array: NumPy array image (if already loaded, e.g., from ROS)

        Returns:
            Tuple of (preprocessed_tensor, original_shape)
        """
        # Load image
        if image_path is not None:
            img = self._load_image_from_path(image_path)
        elif image_array is not None:
            img = self._load_image_from_array(image_array)
        else:
            raise ValueError("Either image_path or image_array must be provided")

        original_height, original_width = img.shape[:2]

        # Resize to model input dimensions
        img_resized = cv2.resize(
            img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )

        # Convert to float and normalize if requested
        if self.normalize:
            img_resized = img_resized.astype(np.float32) / 255.0
        else:
            img_resized = img_resized.astype(np.float32)

        # Convert BGR to RGB (OpenCV loads as BGR)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert to PyTorch tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, (original_height, original_width)

    def _load_image_from_path(self, image_path: str) -> np.ndarray:
        """Load image from file path using OpenCV."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return img

    def _load_image_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Process image from NumPy array (e.g., ROS image)."""
        # Handle different input formats
        if len(image_array.shape) == 2:
            # Grayscale image, convert to BGR
            img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.shape[2] == 3:
            # Already BGR or RGB
            img = image_array.copy()
        elif image_array.shape[2] == 4:
            # RGBA, convert to BGR
            img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")

        return img

    def postprocess_depth(
        self,
        depth_output: torch.Tensor,
        original_shape: Tuple[int, int],
        min_depth: float = 0.1,
        max_depth: float = 100.0,
    ) -> np.ndarray:
        """
        Postprocess depth prediction to original image dimensions.

        Args:
            depth_output: Model output tensor (1, 1, H, W) or (1, H, W)
            original_shape: Original image (height, width)
            min_depth: Minimum depth value for clipping
            max_depth: Maximum depth value for clipping

        Returns:
            Depth map as NumPy array with original dimensions
        """
        # Remove batch dimension and convert to numpy
        if depth_output.dim() == 4:
            depth = depth_output.squeeze(0).squeeze(0).cpu().numpy()
        elif depth_output.dim() == 3:
            depth = depth_output.squeeze(0).cpu().numpy()
        else:
            depth = depth_output.cpu().numpy()

        # Clip depth values
        depth = np.clip(depth, min_depth, max_depth)

        # Resize to original dimensions
        original_height, original_width = original_shape
        depth_resized = cv2.resize(
            depth, (original_width, original_height), interpolation=cv2.INTER_LINEAR
        )

        return depth_resized

    def depth_to_colormap(
        self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA
    ) -> np.ndarray:
        """
        Convert depth map to colored visualization.

        Args:
            depth_map: Depth map array
            colormap: OpenCV colormap constant

        Returns:
            RGB colored depth map
        """
        # Normalize depth to [0, 255]
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 1e-6:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(
                np.uint8
            )
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)

        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return depth_colored
