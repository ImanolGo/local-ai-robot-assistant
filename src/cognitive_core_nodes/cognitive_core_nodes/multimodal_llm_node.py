#!/usr/bin/env python3
"""
Multimodal Language Model Node for Local AI Robot Assistant

This node implements the Gemma 3n E2B multimodal model for text, audio, and vision processing.
Supports both HuggingFace Transformers and Optimum-NVIDIA optimization for enhanced performance.

Features:
- Text generation with multimodal context
- Image understanding and description
- Audio processing (future implementation)
- Memory-efficient model loading with 2B effective footprint
- GPU optimization with TensorRT/Optimum-NVIDIA support

Author: Local AI Robot Assistant Team
License: See LICENSE file in project root
"""

import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Import both standard transformers and optimum-nvidia
try:
    from optimum.nvidia import AutoModelForCausalLM as NvidiaAutoModelForCausalLM
    from optimum.nvidia.pipelines import pipeline as nvidia_pipeline

    OPTIMUM_NVIDIA_AVAILABLE = True
except ImportError:
    OPTIMUM_NVIDIA_AVAILABLE = False
    warnings.warn("Optimum-NVIDIA not available, falling back to standard transformers")

# Add project root to path for imports
import sys

from transformers import AutoProcessor, AutoTokenizer, Gemma3nForConditionalGeneration
from transformers import pipeline as transformers_pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.robot_interfaces.msg import MultimodalQuery, MultimodalResponse  # noqa E402


class MultimodalLLMNode(Node):
    """
    ROS2 node for multimodal language model processing using Gemma 3n E2B.

    Supports both HuggingFace Transformers and Optimum-NVIDIA optimization
    for enhanced performance on Jetson Orin Nano.
    """

    def __init__(self):
        super().__init__("multimodal_llm_node")

        # Declare parameters
        self._declare_parameters()

        # Initialize model configuration
        self.model_id = "google/gemma-3n-e2b"
        self.model_path = PROJECT_ROOT / "models" / "gemma_3n_e2b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # Initialize components
        self.cv_bridge = CvBridge()
        self.model = None
        self.processor = None
        self.pipeline = None
        self.tokenizer = None

        # Performance tracking
        self.inference_times = []
        self.memory_usage = []

        # Initialize model
        self._initialize_model()

        # Create publishers and subscribers
        self._setup_ros_interfaces()

        self.get_logger().info("Multimodal LLM Node initialized successfully")

    def _declare_parameters(self):
        """Declare ROS2 parameters for configuration."""
        self.declare_parameter("use_optimum_nvidia", True)
        self.declare_parameter("use_fp8", False)  # FP8 for advanced optimization
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("max_prompt_length", 1024)
        self.declare_parameter("max_output_length", 2048)
        self.declare_parameter("max_batch_size", 1)  # Conservative for Jetson
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("top_k", 40)
        self.declare_parameter("top_p", 0.9)
        self.declare_parameter("repetition_penalty", 1.1)

    def _setup_ros_interfaces(self):
        """Set up ROS2 publishers and subscribers."""
        # Subscribers
        self.image_subscription = self.create_subscription(
            Image, "/camera/undistorted", self.image_callback, 10
        )

        self.query_subscription = self.create_subscription(
            MultimodalQuery,
            "/cognitive/multimodal_query",
            self.multimodal_query_callback,
            10,
        )

        # Publishers
        self.response_publisher = self.create_publisher(
            MultimodalResponse, "/cognitive/multimodal_response", 10
        )

        self.status_publisher = self.create_publisher(String, "/cognitive/llm_status", 10)

        # Store latest image for multimodal queries
        self.latest_image = None
        self.latest_image_timestamp = None

    def _initialize_model(self):
        """Initialize the Gemma 3n E2B model with optimizations."""
        try:
            use_optimum = self.get_parameter("use_optimum_nvidia").value
            use_fp8 = self.get_parameter("use_fp8").value

            self.get_logger().info("Initializing Gemma 3n E2B model...")
            self.get_logger().info(f"Device: {self.device}, Use Optimum-NVIDIA: {use_optimum}")

            if use_optimum and OPTIMUM_NVIDIA_AVAILABLE and self.device == "cuda":
                self._initialize_optimum_nvidia_model(use_fp8)
            else:
                self._initialize_standard_model()

            # Test model with a simple query
            self._test_model_initialization()

        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {str(e)}")
            raise

    def _initialize_optimum_nvidia_model(self, use_fp8: bool = False):
        """Initialize model with Optimum-NVIDIA optimizations."""
        self.get_logger().info("Initializing with Optimum-NVIDIA optimizations...")

        try:
            # Initialize pipeline for multimodal tasks
            self.pipeline = nvidia_pipeline(
                "image-text-to-text",
                model=self.model_id,
                device="cuda",
                torch_dtype=self.dtype,
                use_fp8=use_fp8,
            )

            # Also initialize the generate API for more control
            self.model = NvidiaAutoModelForCausalLM.from_pretrained(
                self.model_id,
                use_fp8=use_fp8,
                max_prompt_length=self.get_parameter("max_prompt_length").value,
                max_output_length=self.get_parameter("max_output_length").value,
                max_batch_size=self.get_parameter("max_batch_size").value,
                torch_dtype=self.dtype,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")

            self.processor = AutoProcessor.from_pretrained(self.model_id)

            self.get_logger().info("✅ Optimum-NVIDIA model initialized successfully")

        except Exception as e:
            self.get_logger().warn(f"Optimum-NVIDIA initialization failed: {e}")
            self.get_logger().info("Falling back to standard transformers...")
            self._initialize_standard_model()

    def _initialize_standard_model(self):
        """Initialize model with standard HuggingFace Transformers."""
        self.get_logger().info("Initializing with standard HuggingFace Transformers...")

        # Use local model if available
        if self.model_path.exists():
            model_path = str(self.model_path)
            self.get_logger().info(f"Loading local model from {model_path}")
        else:
            model_path = self.model_id
            self.get_logger().info(f"Downloading model from HuggingFace: {model_path}")

        # Initialize processor first (lighter weight)
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Initialize model with memory management
        try:
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto" if self.device == "cuda" else "cpu",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # Avoid flash attention issues on Jetson
            ).eval()

            # Try pipeline after model loads successfully
            try:
                self.pipeline = transformers_pipeline(
                    "image-text-to-text",
                    model=self.model,
                    processor=self.processor,
                    device=self.device,
                    torch_dtype=self.dtype,
                )
            except Exception as e:
                self.get_logger().warn(f"Pipeline initialization failed: {e}")
                self.pipeline = None

        except torch.cuda.OutOfMemoryError as e:
            self.get_logger().warn(f"CUDA OOM, falling back to CPU: {e}")
            self.device = "cpu"
            self.dtype = torch.float32

            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).eval()

        self.get_logger().info("✅ Standard transformers model initialized successfully")

    def _test_model_initialization(self):
        """Test model with a simple query to ensure it's working."""
        try:
            test_prompt = "Hello, can you introduce yourself briefly?"

            if self.pipeline:
                # Test with pipeline
                start_time = time.time()
                output = self.pipeline(test_prompt)
                inference_time = time.time() - start_time

                self.get_logger().info("✅ Model test successful (Pipeline)")
                self.get_logger().info(f"   Test response: {output}")
                self.get_logger().info(f"   Inference time: {inference_time:.2f}s")

            elif self.model and self.processor:
                # Test with direct model
                start_time = time.time()

                # Test text-only generation
                if hasattr(self.processor, "tokenizer"):
                    inputs = self.processor.tokenizer(test_prompt, return_tensors="pt").to(
                        self.device
                    )
                else:
                    inputs = self.processor(text=test_prompt, return_tensors="pt").to(self.device)

                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=50, temperature=0.7, do_sample=True
                    )

                # Decode response
                input_length = inputs["input_ids"].shape[-1]
                generated_tokens = outputs[0][input_length:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)

                inference_time = time.time() - start_time

                self.get_logger().info("✅ Model test successful (Direct)")
                self.get_logger().info(f"   Test response: {response}")
                self.get_logger().info(f"   Inference time: {inference_time:.2f}s")

        except Exception as e:
            self.get_logger().error(f"Model test failed: {str(e)}")
            raise

    def image_callback(self, msg: Image):
        """Store latest image for multimodal queries."""
        try:
            # Convert ROS image to PIL Image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            pil_image = PILImage.fromarray(cv_image)

            self.latest_image = pil_image
            self.latest_image_timestamp = msg.header.stamp

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

    async def multimodal_query_callback(self, msg: MultimodalQuery):
        """Process multimodal query and generate response."""
        try:
            start_time = time.time()

            # Extract query components
            text_query = msg.text_query
            include_image = msg.include_current_image
            audio_data = msg.audio_data if hasattr(msg, "audio_data") else None

            self.get_logger().info(f"Processing multimodal query: {text_query}")

            # Generate response
            response_text = await self._generate_multimodal_response(
                text_query, include_image, audio_data
            )

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Publish response
            response_msg = MultimodalResponse()
            response_msg.query_id = msg.query_id
            response_msg.response_text = response_text
            response_msg.confidence = 0.9  # Placeholder
            response_msg.processing_time = inference_time
            response_msg.header.stamp = self.get_clock().now().to_msg()

            self.response_publisher.publish(response_msg)

            self.get_logger().info(f"Response generated in {inference_time:.2f}s")

        except Exception as e:
            self.get_logger().error(f"Failed to process multimodal query: {str(e)}")

            # Publish error response
            error_response = MultimodalResponse()
            error_response.query_id = msg.query_id
            error_response.response_text = f"Error processing query: {str(e)}"
            error_response.confidence = 0.0
            error_response.header.stamp = self.get_clock().now().to_msg()
            self.response_publisher.publish(error_response)

    async def _generate_multimodal_response(
        self,
        text_query: str,
        include_image: bool = False,
        audio_data: Optional[bytes] = None,
    ) -> str:
        """Generate response using multimodal model."""

        try:
            # Prepare inputs based on modalities
            if include_image and self.latest_image is not None:
                return await self._generate_image_text_response(text_query, self.latest_image)
            elif audio_data:
                # Future implementation for audio processing
                return await self._generate_audio_text_response(text_query, audio_data)
            else:
                return await self._generate_text_response(text_query)

        except Exception as e:
            self.get_logger().error(f"Response generation failed: {str(e)}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    async def _generate_image_text_response(self, text_query: str, image: PILImage.Image) -> str:
        """Generate response using both text and image."""

        # Prepare prompt with image token
        prompt = f"<image_soft_token> {text_query}"

        if self.pipeline:
            # Use pipeline for simplicity
            try:
                output = self.pipeline(image, text=prompt)
                if isinstance(output, list) and len(output) > 0:
                    return output[0].get("generated_text", "").replace(prompt, "").strip()
                return str(output)
            except Exception as e:
                self.get_logger().warn(f"Pipeline failed, using direct model: {e}")

        # Use direct model for more control
        if self.model and self.processor:
            model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
                self.device
            )

            input_length = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.get_parameter("max_new_tokens").value,
                    temperature=self.get_parameter("temperature").value,
                    top_k=self.get_parameter("top_k").value,
                    top_p=self.get_parameter("top_p").value,
                    repetition_penalty=self.get_parameter("repetition_penalty").value,
                    do_sample=True,
                )

                generated_tokens = generation[0][input_length:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)

                return response.strip()

        return "Model not available for image-text processing."

    async def _generate_audio_text_response(self, text_query: str, audio_data: bytes) -> str:
        """Generate response using both text and audio (future implementation)."""
        # TODO: Implement audio processing when Gemma 3n supports audio
        return f"Audio processing not yet implemented. Text query: {text_query}"

    async def _generate_text_response(self, text_query: str) -> str:
        """Generate text-only response."""

        if self.model and self.tokenizer:
            # Use optimized generate API if available
            model_inputs = self.tokenizer([text_query], return_tensors="pt", padding=True).to(
                self.device
            )

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.get_parameter("max_new_tokens").value,
                    temperature=self.get_parameter("temperature").value,
                    top_k=self.get_parameter("top_k").value,
                    top_p=self.get_parameter("top_p").value,
                    repetition_penalty=self.get_parameter("repetition_penalty").value,
                    do_sample=True,
                )

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Remove the input query from response
                response = response.replace(text_query, "").strip()
                return response

        elif self.processor:
            # Fallback to processor-based generation
            inputs = self.processor(text=text_query, return_tensors="pt").to(self.device)
            input_length = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=self.get_parameter("max_new_tokens").value,
                    temperature=self.get_parameter("temperature").value,
                    do_sample=True,
                )

                generated_tokens = generation[0][input_length:]
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)

                return response.strip()

        return "Model not available for text processing."

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {}

        return {
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "total_queries": len(self.inference_times),
            "device": self.device,
            "model_id": self.model_id,
        }

    def destroy_node(self):
        """Cleanup resources."""
        self.get_logger().info("Shutting down Multimodal LLM Node...")

        # Clear model from memory
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        if self.pipeline is not None:
            del self.pipeline

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        super().destroy_node()


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = MultimodalLLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in multimodal LLM node: {e}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
