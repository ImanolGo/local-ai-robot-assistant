"""
RT-MonoDepth Model Wrapper
Handles model architecture loading and weight management
Separates model concerns from inference logic
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class RTMonoDepthModel:
    """
    Wrapper class for RT-MonoDepth model architecture.
    Handles model initialization, weight loading, and basic forward pass.
    """

    def __init__(
        self,
        model_variant: str = "small",
        device: str = "cuda",
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize RT-MonoDepth model.

        Args:
            model_variant: Model size - 'small' or 'full'
            device: Device for model ('cuda' or 'cpu')
            pretrained_path: Path to pretrained weights
        """
        self.model_variant = model_variant
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Load model architecture
        self.model = self._load_architecture()

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_weights(pretrained_path)

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"RT-MonoDepth ({model_variant}) initialized on {device}")

    def _load_architecture(self) -> nn.Module:
        """
        Load the appropriate model architecture.

        Returns:
            PyTorch model instance
        """
        try:
            if self.model_variant == "small":
                from networks.rtmonodepth import RTMonoDepth_S

                model = RTMonoDepth_S()
                self.logger.info("Loaded RTMonoDepth_S architecture")
            elif self.model_variant == "full":
                from networks.rtmonodepth import RTMonoDepth_Full

                model = RTMonoDepth_Full()
                self.logger.info("Loaded RTMonoDepth_Full architecture")
            else:
                raise ValueError(f"Unknown model variant: {self.model_variant}")

            return model

        except ImportError as e:
            self.logger.error(
                "Failed to import RT-MonoDepth networks. "
                "Ensure the 'networks' folder is copied from RT-MonoDepth repository."
            )
            raise ImportError(
                "RT-MonoDepth networks not found. "
                "Please copy the 'networks/' folder from the RT-MonoDepth repository "
                "to src/perception_nodes/perception_nodes/depth/"
            ) from e

    def load_weights(self, weight_path: Union[str, Path]) -> None:
        """
        Load pretrained weights into the model.

        Args:
            weight_path: Path to model checkpoint (.pth file)
        """
        weight_path = Path(weight_path)

        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        self.logger.info(f"Loading weights from {weight_path}")

        # Load checkpoint
        checkpoint = torch.load(weight_path, map_location=self.device)

        # Extract state dict from checkpoint
        state_dict = self._extract_state_dict(checkpoint)

        # Handle potential key mismatches
        state_dict = self._handle_state_dict_keys(state_dict)

        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=True)
            self.logger.info("✓ Weights loaded successfully (strict mode)")
        except RuntimeError as e:
            self.logger.warning(f"Strict loading failed, trying non-strict mode: {e}")
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("✓ Weights loaded successfully (non-strict mode)")

    def _extract_state_dict(self, checkpoint: Dict) -> Dict:
        """
        Extract state dictionary from checkpoint.
        Handles various checkpoint formats.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            State dictionary
        """
        # Try different common checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            elif "model" in checkpoint:
                return checkpoint["model"]
            else:
                # Assume checkpoint itself is the state dict
                return checkpoint
        else:
            # Checkpoint is already a state dict
            return checkpoint

    def _handle_state_dict_keys(self, state_dict: Dict) -> Dict:
        """
        Handle potential key naming mismatches in state dict.

        Args:
            state_dict: Original state dictionary

        Returns:
            Modified state dictionary with corrected keys
        """
        # Check if keys have 'module.' prefix (from DataParallel)
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

        if has_module_prefix:
            self.logger.info("Removing 'module.' prefix from state dict keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            return new_state_dict

        return state_dict

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Depth prediction tensor or dictionary of outputs
        """
        with torch.no_grad():
            output = self.model(x)

        return output

    def __call__(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Allow calling the wrapper like a function."""
        return self.forward(x)

    def get_output_tensor(self, model_output: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Extract the main depth tensor from model output.
        Handles both tensor and dictionary outputs.

        Args:
            model_output: Raw model output

        Returns:
            Depth prediction tensor
        """
        if isinstance(model_output, torch.Tensor):
            return model_output
        elif isinstance(model_output, dict):
            # Try common keys
            for key in ["pred_depth", "depth", "disp", "output"]:
                if key in model_output:
                    return model_output[key]
            # If no common key found, return first value
            return list(model_output.values())[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(model_output)}")

    def get_model_info(self) -> Dict:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "variant": self.model_variant,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }

    def print_model_info(self) -> None:
        """Print model information to console."""
        info = self.get_model_info()
        print("\n" + "=" * 50)
        print("RT-MonoDepth Model Information")
        print("=" * 50)
        print(f"Variant:              {info['variant']}")
        print(f"Device:               {info['device']}")
        print(f"Total Parameters:     {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size:           {info['model_size_mb']:.2f} MB")
        print("=" * 50 + "\n")

    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save model weights to file.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with additional metadata
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_variant": self.model_variant,
            "model_info": self.get_model_info(),
        }

        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def to(self, device: str) -> "RTMonoDepthModel":
        """
        Move model to specified device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            Self for chaining
        """
        self.device = device
        self.model.to(device)
        self.logger.info(f"Model moved to {device}")
        return self

    def eval(self) -> "RTMonoDepthModel":
        """
        Set model to evaluation mode.

        Returns:
            Self for chaining
        """
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "RTMonoDepthModel":
        """
        Set model to training mode.

        Args:
            mode: Training mode flag

        Returns:
            Self for chaining
        """
        self.model.train(mode)
        return self


class RTMonoDepthModelFactory:
    """Factory class for creating RT-MonoDepth models with common configurations."""

    @staticmethod
    def create_small_model(
        weight_path: Optional[str] = None, device: str = "cuda"
    ) -> RTMonoDepthModel:
        """
        Create small model variant.

        Args:
            weight_path: Path to pretrained weights
            device: Device for model

        Returns:
            RTMonoDepthModel instance
        """
        return RTMonoDepthModel(model_variant="small", device=device, pretrained_path=weight_path)

    @staticmethod
    def create_full_model(
        weight_path: Optional[str] = None, device: str = "cuda"
    ) -> RTMonoDepthModel:
        """
        Create full model variant.

        Args:
            weight_path: Path to pretrained weights
            device: Device for model

        Returns:
            RTMonoDepthModel instance
        """
        return RTMonoDepthModel(model_variant="full", device=device, pretrained_path=weight_path)

    @staticmethod
    def from_checkpoint(checkpoint_path: str, device: str = "cuda") -> RTMonoDepthModel:
        """
        Create model from checkpoint that includes variant info.

        Args:
            checkpoint_path: Path to checkpoint
            device: Device for model

        Returns:
            RTMonoDepthModel instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Try to extract variant from checkpoint
        variant = checkpoint.get("model_variant", "small")

        return RTMonoDepthModel(
            model_variant=variant, device=device, pretrained_path=checkpoint_path
        )


# Convenience functions
def create_depth_model(
    model_type: str = "small", weight_path: Optional[str] = None, device: str = "cuda"
) -> RTMonoDepthModel:
    """
    Convenience function to create a depth estimation model.

    Args:
        model_type: 'small' or 'full'
        weight_path: Path to pretrained weights
        device: Device for model

    Returns:
        RTMonoDepthModel instance
    """
    return RTMonoDepthModel(model_variant=model_type, device=device, pretrained_path=weight_path)
