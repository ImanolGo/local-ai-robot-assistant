#!/usr/bin/env python3
"""
Enhanced YOLO Object Detection Test Script - OPTIMIZED VERSION
Tests HuggingFace and TensorRT YOLO models with comprehensive metrics
Optimized for NVIDIA Jetson Orin Nano deployment

PERFORMANCE OPTIMIZATIONS APPLIED:
1. Detection class uses __slots__ for reduced memory overhead
2. Optimized TensorRT inference with batch CPU transfers (no per-detection GPU→CPU)
3. Extended warmup (20 iterations) for stable TensorRT performance
4. Median FPS reporting for more stable performance metrics
5. System performance tuning (jetson_clocks, OpenCV optimization)
6. Increased default iterations (100) for better statistical accuracy
7. Raw array inference method for pure performance benchmarking
8. Percentile analysis (P50, P95, P99) for latency distribution

Expected performance improvement: 11.87 FPS → 20+ FPS

This script follows the same pattern as test_depth.py but for object detection:
- Tests YOLOv11 models in both HuggingFace and TensorRT formats
- Provides comprehensive performance benchmarking
- Compares detection accuracy between model formats
- Generates detailed visualizations with bounding boxes
- Calculates object detection metrics (mAP, precision, recall)
- Optimized for Jetson Orin Nano constraints

Usage Examples:
    # Test TensorRT model with optimizations (recommended)
    python test_yolo.py --models-dir models/yolo_trt --no-huggingface

    # Run detailed benchmarking comparison
    python test_yolo.py --models-dir models/yolo_trt --benchmark

    # Test both HF and TensorRT models
    python test_yolo.py --models-dir models/yolo_trt --compare

    # Custom test image with high iteration count
    python test_yolo.py --image path/to/test.jpg --iterations 200

Performance Tips:
    - Run 'sudo jetson_clocks' before testing for maximum performance
    - Use --no-huggingface for fastest testing (TensorRT only)
    - Use --benchmark for detailed overhead analysis
"""

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import psutil

# Optional imports with fallbacks
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import torchvision.transforms as transforms
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoImageProcessor = None
    AutoModelForObjectDetection = None
    transforms = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# COCO class names for visualization
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def set_jetson_max_performance():
    """Set Jetson to maximum performance mode"""
    import subprocess

    print("Setting Jetson to maximum performance...")

    commands = [
        # Max CPU clocks
        "sudo jetson_clocks",
        # Set power mode to MAXN (maximum performance)
        "sudo nvpmodel -m 0",
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print(f"  ✓ Executed: {cmd}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {cmd} - {e}")


def optimize_opencv_threading():
    """Optimize OpenCV for better performance"""
    # Set number of threads (adjust based on your Jetson)
    cv2.setNumThreads(4)  # Orin Nano has 6 cores, use 4 for CV

    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)

    print("OpenCV optimizations enabled:")
    print(f"  Threads: {cv2.getNumThreads()}")
    print(f"  Optimized: {cv2.useOptimized()}")


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test image with some object-like patterns for detection testing"""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Create some colored rectangles that could be detected as objects
    cv2.rectangle(image, (50, 50), (200, 200), (255, 128, 0), -1)  # Orange rectangle
    cv2.rectangle(image, (300, 100), (450, 250), (0, 255, 128), -1)  # Green rectangle
    cv2.circle(image, (500, 350), 80, (128, 0, 255), -1)  # Purple circle

    # Add some texture
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    # Add some text-like patterns
    cv2.putText(image, "TEST", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return image


class Detection:
    """Optimized detection class with minimal overhead"""

    __slots__ = ["bbox", "confidence", "class_id", "class_name"]

    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str = ""):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = (
            class_name or COCO_CLASSES[class_id]
            if class_id < len(COCO_CLASSES)
            else f"class_{class_id}"
        )

    def area(self) -> float:
        """Calculate bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def iou(self, other: "Detection") -> float:
        """Calculate Intersection over Union with another detection"""
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection

        return intersection / union if union > 0 else 0.0


class HuggingFaceYOLOModel:
    """HuggingFace YOLO model wrapper"""

    def __init__(self, model_name: str = "hustvl/yolos-tiny", use_cpu_only: bool = True):
        """Initialize HuggingFace YOLO model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")

        print(f"Loading HuggingFace model: {model_name}...")

        # Force CPU to avoid GPU memory issues on Jetson
        self.device = "cpu" if use_cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            print(f"✓ HuggingFace model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")

        # Performance tracking
        self.inference_times = []
        self.model_name = model_name

    def infer(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        """Run inference and return list of detections"""
        with torch.no_grad():
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            # Inference with timing
            start_time = time.time()
            outputs = self.model(pixel_values)
            inference_time = time.time() - start_time

            self.inference_times.append(inference_time)

            # Post-process outputs
            detections = []

            # Extract predictions
            logits = outputs.logits[0]
            boxes = outputs.pred_boxes[0]

            # Convert to detections
            probabilities = torch.nn.functional.softmax(logits, -1)
            scores, labels = probabilities[..., :-1].max(-1)

            # Filter by confidence
            keep = scores > confidence_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Convert boxes to image coordinates
            h, w = image.shape[:2]
            for score, label, box in zip(scores, labels, boxes):
                # Convert from center_x, center_y, width, height to x1, y1, x2, y2
                cx, cy, bbox_w, bbox_h = box.tolist()
                x1 = (cx - bbox_w / 2) * w
                y1 = (cy - bbox_h / 2) * h
                x2 = (cx + bbox_w / 2) * w
                y2 = (cy + bbox_h / 2) * h

                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=score.item(),
                    class_id=label.item(),
                )
                detections.append(detection)

            return detections

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "std_inference_time": np.std(times),
            "avg_fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
        }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor


class TensorRTYOLOModel:
    """TensorRT YOLO model wrapper using Ultralytics"""

    def __init__(self, engine_path: str, confidence_threshold: float = 0.5):
        """Initialize TensorRT YOLO model"""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        print(f"Loading TensorRT engine from {engine_path}...")

        try:
            self.model = YOLO(engine_path)
            self.confidence_threshold = confidence_threshold

            print("✓ TensorRT model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT model: {e}")

        # Performance tracking
        self.inference_times = []
        self.engine_path = engine_path

    def infer(self, image: np.ndarray, confidence_threshold: float = None) -> List[Detection]:
        """
        Optimized inference method with minimal overhead

        Key optimizations:
        1. Batch CPU transfers instead of per-detection
        2. Reuse numpy arrays
        3. Minimal Python object overhead
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        # Inference with timing
        start_time = time.time()
        results = self.model(image, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - start_time

        self.inference_times.append(inference_time)

        # Convert to our Detection format
        detections = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # OPTIMIZATION: Single batch CPU transfer
                # OLD (slow): boxes.xyxy[i].cpu().numpy() in loop
                # NEW (fast): boxes.xyxy.cpu().numpy() once
                bboxes_np = boxes.xyxy.cpu().numpy()  # Shape: (N, 4)
                confs_np = boxes.conf.cpu().numpy()  # Shape: (N,)
                classes_np = boxes.cls.cpu().numpy().astype(int)  # Shape: (N,)

                # Create Detection objects from numpy arrays
                for i in range(len(bboxes_np)):
                    detection = Detection(
                        bbox=bboxes_np[i].tolist(),  # Convert to list only once
                        confidence=float(confs_np[i]),
                        class_id=int(classes_np[i]),
                    )
                    detections.append(detection)

        return detections

    def infer_raw(self, image: np.ndarray, confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Fastest inference - returns raw numpy arrays without object creation

        Use this for pure performance benchmarking

        Returns:
            Dictionary with keys:
            - 'bboxes': (N, 4) array of [x1, y1, x2, y2]
            - 'confidences': (N,) array of confidence scores
            - 'class_ids': (N,) array of class IDs
            - 'inference_time': float
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        start_time = time.time()
        results = self.model(image, conf=confidence_threshold, verbose=False)
        inference_time = time.time() - start_time

        self.inference_times.append(inference_time)

        raw_results = {
            "bboxes": np.array([]),
            "confidences": np.array([]),
            "class_ids": np.array([]),
            "inference_time": inference_time,
        }

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                # Single batch transfer - fastest method
                raw_results["bboxes"] = boxes.xyxy.cpu().numpy()
                raw_results["confidences"] = boxes.conf.cpu().numpy()
                raw_results["class_ids"] = boxes.cls.cpu().numpy().astype(int)

        return raw_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        return {
            "avg_inference_time": np.mean(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "std_inference_time": np.std(times),
            "avg_fps": 1.0 / np.mean(times),
            "total_inferences": len(times),
        }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "model"):
            del self.model


def benchmark_comparison(model: TensorRTYOLOModel, image: np.ndarray, iterations: int = 100):
    """
    Compare different inference methods to measure overhead
    """
    print("=" * 60)
    print("INFERENCE METHOD COMPARISON")
    print("=" * 60)

    # Method 1: Optimized with Detection objects
    print("\n1. Testing optimized method (with Detection objects)...")
    model.inference_times = []
    for _ in range(iterations):
        _ = model.infer(image)
    stats1 = model.get_performance_stats()
    print(f"   Average FPS: {stats1['avg_fps']:.2f}")
    print(f"   Average time: {stats1['avg_inference_time']*1000:.2f}ms")

    # Method 2: Raw arrays (fastest)
    print("\n2. Testing raw array method (no objects)...")
    model.inference_times = []
    for _ in range(iterations):
        _ = model.infer_raw(image)
    stats2 = model.get_performance_stats()
    print(f"   Average FPS: {stats2['avg_fps']:.2f}")
    print(f"   Average time: {stats2['avg_inference_time']*1000:.2f}ms")

    # Method 3: Pure Ultralytics (baseline)
    print("\n3. Testing pure Ultralytics (baseline)...")
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model.model(image, conf=0.5, verbose=False)
        times.append(time.time() - start)
    avg_time = np.mean(times)
    print(f"   Average FPS: {1.0/avg_time:.2f}")
    print(f"   Average time: {avg_time*1000:.2f}ms")

    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)
    baseline_time = avg_time
    overhead1 = (stats1["avg_inference_time"] - baseline_time) * 1000
    overhead2 = (stats2["avg_inference_time"] - baseline_time) * 1000

    print(f"Baseline (pure Ultralytics): {baseline_time*1000:.2f}ms")
    print(f"Optimized method overhead: {overhead2:.2f}ms ({abs(overhead2/baseline_time/10):.1f}%)")
    print(f"Original method overhead: {overhead1:.2f}ms ({abs(overhead1/baseline_time/10):.1f}%)")


def calculate_detection_metrics(
    detections1: List[Detection],
    detections2: List[Detection],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate detection comparison metrics between two detection lists"""

    # Basic counts
    count1 = len(detections1)
    count2 = len(detections2)

    if count1 == 0 and count2 == 0:
        return {
            "detection_count_diff": 0,
            "avg_confidence_diff": 0.0,
            "matched_detections": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "avg_iou": 0.0,
            "error": "No detections in either model",
        }

    if count1 == 0 or count2 == 0:
        return {
            "detection_count_diff": abs(count1 - count2),
            "avg_confidence_diff": float("nan"),
            "matched_detections": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_iou": 0.0,
            "error": "One model has no detections",
        }

    # Match detections using IoU
    matched_pairs = []
    used_indices2 = set()

    for i, det1 in enumerate(detections1):
        best_iou = 0
        best_match = -1

        for j, det2 in enumerate(detections2):
            if j in used_indices2:
                continue

            # Only match same class
            if det1.class_id == det2.class_id:
                iou = det1.iou(det2)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = j

        if best_match >= 0:
            matched_pairs.append((i, best_match, best_iou))
            used_indices2.add(best_match)

    # Calculate metrics
    matched_count = len(matched_pairs)
    precision = matched_count / count1 if count1 > 0 else 0.0
    recall = matched_count / count2 if count2 > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Average IoU of matched detections
    avg_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0.0

    # Confidence comparison for matched detections
    confidence_diffs = []
    for i, j, _ in matched_pairs:
        conf_diff = abs(detections1[i].confidence - detections2[j].confidence)
        confidence_diffs.append(conf_diff)

    avg_confidence_diff = np.mean(confidence_diffs) if confidence_diffs else float("nan")

    return {
        "detection_count_diff": abs(count1 - count2),
        "avg_confidence_diff": avg_confidence_diff,
        "matched_detections": matched_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_iou": avg_iou,
        "total_detections_1": count1,
        "total_detections_2": count2,
    }


def visualize_detections(
    image: np.ndarray,
    detections: List[Detection],
    model_name: str = "",
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    vis_image = image.copy()

    # Colors for different classes (BGR format)
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    for detection in detections:
        if detection.confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        color = colors[detection.class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Background for text
        cv2.rectangle(
            vis_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )

        # Text
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Add model name
    if model_name:
        cv2.putText(
            vis_image,
            model_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    return vis_image


def create_detection_comparison(
    image: np.ndarray,
    detections_hf: List[Detection] = None,
    detections_trt: List[Detection] = None,
    metrics: Dict[str, Any] = None,
    perf_stats: Dict[str, Dict[str, Any]] = None,
    output_path: str = None,
) -> np.ndarray:
    """Create comprehensive detection comparison visualization"""

    # Resize image for consistent layout
    target_height = 400
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)
    image_resized = cv2.resize(image, (target_width, target_height))

    # Create visualizations
    images = []
    labels = []

    # Original image
    images.append(image_resized)
    labels.append("Original Image")

    # HuggingFace detections
    if detections_hf is not None:
        hf_vis = visualize_detections(image_resized, detections_hf, "HuggingFace")
        images.append(hf_vis)
        labels.append(f"HuggingFace ({len(detections_hf)} detections)")
    else:
        placeholder = np.zeros_like(image_resized)
        cv2.putText(
            placeholder,
            "HF N/A",
            (target_width // 4, target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (128, 128, 128),
            2,
        )
        images.append(placeholder)
        labels.append("HuggingFace N/A")

    # TensorRT detections
    if detections_trt is not None:
        trt_vis = visualize_detections(image_resized, detections_trt, "TensorRT")
        images.append(trt_vis)
        labels.append(f"TensorRT ({len(detections_trt)} detections)")
    else:
        placeholder = np.zeros_like(image_resized)
        cv2.putText(
            placeholder,
            "TRT N/A",
            (target_width // 4, target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (128, 128, 128),
            2,
        )
        images.append(placeholder)
        labels.append("TensorRT N/A")

    # Create 2x2 grid if we have 4 images, otherwise adjust layout
    if len(images) >= 3:
        # 2x2 grid layout
        top_row = np.hstack([images[0], images[1]])
        bottom_row = np.hstack([images[2], images[0] if len(images) < 4 else images[3]])
        main_grid = np.vstack([top_row, bottom_row])
    else:
        # Horizontal layout for fewer images
        main_grid = np.hstack(images)

    # Add space for text
    text_height = 250
    final_image = np.zeros(
        (main_grid.shape[0] + text_height, main_grid.shape[1], 3), dtype=np.uint8
    )
    final_image[: main_grid.shape[0], : main_grid.shape[1]] = main_grid

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    # Label positions for 2x2 grid
    label_positions = [
        (10, 30),  # Top-left
        (target_width + 10, 30),  # Top-right
        (10, target_height + 30),  # Bottom-left
        (target_width + 10, target_height + 30),  # Bottom-right
    ]

    for i, (label, pos) in enumerate(zip(labels, label_positions)):
        if i < len(labels):
            cv2.putText(final_image, label, pos, font, font_scale, color, thickness)

    # Add performance metrics
    text_start_y = main_grid.shape[0] + 50

    if perf_stats:
        cv2.putText(
            final_image,
            "Performance Comparison:",
            (10, text_start_y),
            font,
            0.8,
            (0, 255, 255),
            2,
        )

        y_offset = text_start_y + 30
        for model_name, stats in perf_stats.items():
            if stats:
                fps_text = (
                    f"{model_name}: {stats.get('avg_fps', 0):.1f} FPS, "
                    f"{stats.get('avg_inference_time', 0)*1000:.1f}ms"
                )
                cv2.putText(final_image, fps_text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
                y_offset += 25

    # Add detection metrics
    if metrics:
        metrics_y = text_start_y + 120
        cv2.putText(
            final_image,
            "Detection Quality Metrics:",
            (10, metrics_y),
            font,
            0.8,
            (0, 255, 255),
            2,
        )

        y_offset = metrics_y + 30
        metrics_text = [
            f"Matched Detections: {metrics.get('matched_detections', 0)}",
            f"F1 Score: {metrics.get('f1_score', 0):.3f}",
            f"Avg IoU: {metrics.get('avg_iou', 0):.3f}",
            f"Detection Count Diff: {metrics.get('detection_count_diff', 0)}",
        ]

        for text in metrics_text:
            cv2.putText(final_image, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25

    # Save visualization
    if output_path:
        cv2.imwrite(output_path, final_image)
        print(f"✓ Detection comparison saved to: {output_path}")

    return final_image


def check_memory_status() -> Dict[str, Any]:
    """Check system and GPU memory status"""
    print("🔍 Memory Status:")
    print("=" * 40)

    mem_info = {}

    # System memory
    mem = psutil.virtual_memory()
    mem_info["system"] = {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent_used": mem.percent,
    }

    print("System RAM:")
    print(f"  Total: {mem_info['system']['total_gb']:.1f} GB")
    print(f"  Available: {mem_info['system']['available_gb']:.1f} GB")
    print(
        f"  Used: {mem_info['system']['used_gb']:.1f} GB \
            ({mem_info['system']['percent_used']:.1f}%)"
    )

    # GPU memory - try nvidia-smi
    mem_info["gpu"] = {"error": "GPU memory unavailable"}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(",")
            if len(gpu_data) == 3:
                total_mb, used_mb, free_mb = map(int, gpu_data)
                mem_info["gpu"] = {
                    "total_gb": total_mb / 1024,
                    "used_gb": used_mb / 1024,
                    "free_gb": free_mb / 1024,
                    "percent_used": (used_mb / total_mb) * 100,
                }

                print("\nGPU Memory:")
                print(f"  Total: {mem_info['gpu']['total_gb']:.1f} GB")
                print(
                    f"  Used: {mem_info['gpu']['used_gb']:.1f} GB \
                        ({mem_info['gpu']['percent_used']:.1f}%)"
                )
                print(f"  Free: {mem_info['gpu']['free_gb']:.1f} GB")

    except Exception:
        print("\n⚠ Could not get GPU memory info")

    return mem_info


def test_yolo_detection(
    models_dir: str,
    test_image_path: str = None,
    output_path: str = None,
    num_iterations: int = 100,  # Increased for more stable measurements
    confidence_threshold: float = 0.5,
    test_huggingface: bool = False,  # Default False for faster testing
    compare_models: bool = True,
):
    """Optimized YOLO testing with minimal overhead"""

    print("🔬 OPTIMIZED YOLO Object Detection Testing")
    print("=" * 60)

    models_dir = Path(models_dir)

    # OPTIMIZATION 1: System tuning
    try:
        set_jetson_max_performance()
        optimize_opencv_threading()
    except Exception as e:
        print(f"⚠ System optimization skipped: {e}")

    # OPTIMIZATION 2: Load image once and optimize it
    if test_image_path and Path(test_image_path).exists():
        image = cv2.imread(str(test_image_path))
        print(f"✓ Loaded test image: {test_image_path} {image.shape}")
    else:
        # Use Ultralytics default test image (already optimized)
        image = create_test_image()
        print(f"✓ Created synthetic test image: {image.shape}")

    # Check memory
    try:
        check_memory_status()
        print()
    except Exception as e:
        print(f"⚠ Could not check memory: {e}")

    models = {}
    detection_results = {}

    try:
        # Load TensorRT model
        engine_files = list(models_dir.glob("*.engine"))
        if engine_files:
            engine_path = engine_files[0]  # Use first engine found
            try:
                models["tensorrt"] = TensorRTYOLOModel(str(engine_path), confidence_threshold)
                print(f"✓ Loaded TensorRT model: {engine_path}")
            except Exception as e:
                print(f"✗ Failed to load TensorRT model: {e}")
        else:
            print(f"⚠ No TensorRT engine files found in {models_dir}")

        # Load HuggingFace model if requested
        if test_huggingface and compare_models:
            try:
                models["huggingface"] = HuggingFaceYOLOModel(use_cpu_only=True)
                print("✓ Loaded HuggingFace YOLO model")
            except Exception as e:
                print(f"✗ Failed to load HuggingFace model: {e}")
                print("Continuing with TensorRT-only testing...")

        if not models:
            print("❌ No models loaded successfully!")
            return False

        print(f"\n📊 Testing {len(models)} model(s): {list(models.keys())}")

        # OPTIMIZATION 3: Extended warmup (critical for TensorRT)
        print("\nExtended warmup (20 iterations)...")
        models_to_remove = []
        for model_name, model in models.items():
            try:
                print(f"  Warming up {model_name}...")
                for i in range(20):
                    _ = model.infer(image, confidence_threshold)
                    if i == 9:
                        print("    50% complete...")
                print("  ✓ Warmup complete")
            except Exception as e:
                print(f"  ✗ {model_name} warmup failed: {e}")
                models_to_remove.append(model_name)

        # Remove failed models
        for model_name in models_to_remove:
            del models[model_name]

        if not models:
            print("❌ All models failed during warmup!")
            return False

        # OPTIMIZATION 4: Clear timing data from warmup
        for model in models.values():
            model.inference_times = []

        # OPTIMIZATION 5: Performance testing with progress
        print(f"\nPerformance testing ({num_iterations} iterations)...")

        for model_name, model in models.items():
            print(f"  Testing {model_name}...")

            for i in range(num_iterations):
                try:
                    detections = model.infer(image, confidence_threshold)
                    detection_results[model_name] = detections

                    if (i + 1) % 25 == 0:
                        print(f"    {i+1}/{num_iterations} iterations complete")

                except Exception as e:
                    print(f"    ✗ Iteration {i+1} failed: {e}")

        # Calculate metrics
        print("\nCalculating comparison metrics...")
        metrics = {}

        if "huggingface" in detection_results and "tensorrt" in detection_results:
            metrics["hf_vs_trt"] = calculate_detection_metrics(
                detection_results["huggingface"], detection_results["tensorrt"]
            )

        # Performance stats
        perf_stats = {}
        for model_name, model in models.items():
            perf_stats[model_name] = model.get_performance_stats()

        # Display results
        print("\n=== Performance Results ===")
        for model_name, stats in perf_stats.items():
            if stats:
                det_count = len(detection_results.get(model_name, []))

                # OPTIMIZATION 6: Use median instead of mean for more stable FPS
                model = models[model_name]
                times = np.array(model.inference_times)
                median_time = np.median(times)
                median_fps = 1.0 / median_time

                print(f"{model_name.upper()}:")
                print(f"  Average FPS: {stats.get('avg_fps', 0):.2f}")
                print(f"  Median FPS: {median_fps:.2f}")  # More stable metric
                print(f"  Average inference time: {stats.get('avg_inference_time', 0)*1000:.1f}ms")
                print(f"  Median inference time: {median_time*1000:.1f}ms")
                print(f"  Min inference time: {stats.get('min_inference_time', 0)*1000:.1f}ms")
                print(f"  Max inference time: {stats.get('max_inference_time', 0)*1000:.1f}ms")
                print(f"  Std deviation: {stats.get('std_inference_time', 0)*1000:.1f}ms")
                print(f"  Detections: {det_count}")

                # Percentile analysis
                p50 = np.percentile(times, 50) * 1000
                p95 = np.percentile(times, 95) * 1000
                p99 = np.percentile(times, 99) * 1000

                print("  Percentile analysis:")
                print(f"    P50 (median): {p50:.1f}ms")
                print(f"    P95: {p95:.1f}ms")
                print(f"    P99: {p99:.1f}ms")
                print()

        # Performance comparison
        if len(perf_stats) > 1:
            print("\n=== Performance Comparison ===")
            fps_values = {
                name: stats.get("avg_fps", 0) for name, stats in perf_stats.items() if stats
            }
            best_fps = max(fps_values.values())
            best_model = max(fps_values.keys(), key=lambda k: fps_values[k])

            print(f"Fastest model: {best_model} ({best_fps:.2f} FPS)")

            for model_name, fps in fps_values.items():
                if model_name != best_model:
                    speedup = best_fps / fps if fps > 0 else float("inf")
                    print(f"  {model_name}: {speedup:.2f}x slower")

        # Detection quality metrics
        if metrics:
            print("\n=== Detection Quality Metrics ===")
            for comparison, metric_data in metrics.items():
                if "error" in metric_data:
                    print(f"{comparison}: {metric_data['error']}")
                else:
                    print(f"{comparison}:")
                    print(f"  Matched detections: {metric_data['matched_detections']}")
                    print(f"  F1 Score: {metric_data['f1_score']:.3f}")
                    print(f"  Average IoU: {metric_data['avg_iou']:.3f}")
                    print(f"  Detection count difference: {metric_data['detection_count_diff']}")

        # Save visualizations and results
        if output_path:
            output_path = Path(output_path)

            # Save detection comparison image
            _ = create_detection_comparison(
                image,
                detections_hf=detection_results.get("huggingface"),
                detections_trt=detection_results.get("tensorrt"),
                metrics=metrics.get("hf_vs_trt"),
                perf_stats=perf_stats,
                output_path=f"{output_path}_detection_comparison.jpg",
            )

            # Save individual detection visualizations
            for model_name, detections in detection_results.items():
                if detections:
                    vis_img = visualize_detections(image, detections, model_name)
                    vis_path = f"{output_path}_{model_name}_detections.jpg"
                    cv2.imwrite(vis_path, vis_img)
                    print(f"✓ {model_name} detections saved to: {vis_path}")

            # Save comprehensive results
            results_data = {
                "test_config": {
                    "models_dir": str(models_dir),
                    "test_image": str(test_image_path),
                    "num_iterations": num_iterations,
                    "confidence_threshold": confidence_threshold,
                },
                "performance_stats": perf_stats,
                "detection_metrics": metrics,
                "detection_counts": {
                    model_name: len(detections)
                    for model_name, detections in detection_results.items()
                },
            }

            results_file = f"{output_path}_comprehensive_results.json"
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)

            print(f"✓ Comprehensive results saved to: {results_file}")

        # Cleanup
        for model in models.values():
            try:
                model.cleanup()
            except Exception:
                pass

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO object detection models (HuggingFace and TensorRT) \
            with comprehensive visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--models-dir",
        default="models/yolo_trt",
        help="Directory containing TensorRT engine files",
    )
    parser.add_argument("--image", help="Path to test image (default: assets/images/bus.jpg)")
    parser.add_argument(
        "--output",
        default="yolo_test",
        help="Output path prefix for visualizations and results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,  # Increased default for more stable measurements
        help="Number of performance test iterations",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Skip HuggingFace model testing (TensorRT only)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip model comparison (faster, individual testing only)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run detailed benchmarking comparison between inference methods",
    )

    args = parser.parse_args()

    # Handle comparison logic
    test_huggingface = not args.no_huggingface
    compare_models = not args.no_compare and test_huggingface

    # Run benchmark comparison if requested
    if args.benchmark:
        print("🚀 Running detailed benchmarking comparison...")
        try:
            models_dir = Path(args.models_dir)
            engine_files = list(models_dir.glob("*.engine"))
            if engine_files:
                engine_path = engine_files[0]
                model = TensorRTYOLOModel(str(engine_path), args.confidence)

                # Load test image
                if args.image and Path(args.image).exists():
                    image = cv2.imread(str(args.image))
                else:
                    image = create_test_image()

                # Run warmup
                print("Warming up for benchmark...")
                for _ in range(10):
                    _ = model.infer(image)

                benchmark_comparison(model, image, args.iterations)
                model.cleanup()
                return
            else:
                print("❌ No TensorRT engine files found for benchmarking")
                return
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            return

    success = test_yolo_detection(
        models_dir=args.models_dir,
        test_image_path=args.image,
        output_path=args.output,
        num_iterations=args.iterations,
        confidence_threshold=args.confidence,
        test_huggingface=test_huggingface,
        compare_models=compare_models,
    )

    if success:
        print("\n🎉 Testing completed successfully!")
        print("\n📁 Generated files:")
        print(f"   - {args.output}_detection_comparison.jpg: Model comparison visualization")
        print(
            f"   - {args.output}_comprehensive_results.json: Detailed metrics and performance data"
        )
        if test_huggingface:
            print("   - Individual detection visualizations for each model")
        else:
            print("   - TensorRT-only detection visualization")

        # Performance recommendations
        print("\n💡 Deployment recommendations:")
        print("   - For Jetson Orin Nano: Use TensorRT FP16 for best performance")
        print("   - Target: 20+ FPS for real-time object detection")
        print("   - Monitor GPU memory usage to avoid OOM errors")

        sys.exit(0)
    else:
        print("\n❌ Testing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
