#!/usr/bin/env python3
"""
Audio Model Optimization Script

This script optimizes audio models for the Jetson Orin Nano to meet performance targets:
- openWakeWord: <5% CPU usage
- faster-whisper: <0.3x real-time factor, <300MB RAM

Optimization strategies:
1. Use smaller/optimized wake word models
2. Configure faster-whisper with aggressive optimizations
3. Memory management and cleanup
4. Multi-threading optimization for Jetson

Usage:
    python scripts/optimize_audio_models.py [--test] [--config-only]
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import yaml
from faster_whisper import WhisperModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_DIR = PROJECT_ROOT / "models"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioOptimizer:
    """Optimize audio models for Jetson Orin Nano performance."""

    def __init__(self):
        """Initialize the audio optimizer."""
        self.config = {
            "wake_word": {
                "model_path": str(MODEL_DIR / "wake_word" / "hey_roe_ver.onnx"),
                "inference_framework": "onnx",
                "vad_threshold": 0.5,
                "prediction_frequency": 0.05,  # Every 50ms instead of default 80ms
                "enable_speex_noise_suppression": True,
            },
            "whisper": {
                "model_size": "tiny",
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 2,  # Reduced from 4 for better CPU sharing
                "num_workers": 1,
                "beam_size": 1,  # Greedy decoding for speed
                "best_of": 1,
                "patience": 1.0,
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
                "no_speech_threshold": 0.6,
                "log_prob_threshold": -1.0,
                "compression_ratio_threshold": 2.4,
                "condition_on_previous_text": False,  # Disable for speed
                "initial_prompt": None,
                "word_timestamps": False,  # Disable for speed
                "prepend_punctuations": "\"'([{-",
                "append_punctuations": "\"'.,,!!??::)]}",
            },
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 800,  # Smaller chunks for wake word (50ms at 16kHz)
                "buffer_size": 2048,
            },
        }

    def create_optimized_config(self) -> None:
        """Create optimized audio configuration file."""
        config_path = CONFIG_DIR / "audio_config_optimized.yaml"

        logger.info(f"Creating optimized audio config: {config_path}")

        # Ensure config directory exists
        CONFIG_DIR.mkdir(exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info("✅ Optimized audio configuration created")

    def setup_optimized_wake_word(self):
        """Set up wake word detection with optimizations."""
        try:
            from openwakeword import Model as WakeWordModel

            logger.info("Setting up optimized wake word detection...")

            # Use optimized settings
            model = WakeWordModel(
                wakeword_models=[self.config["wake_word"]["model_path"]],
                inference_framework=self.config["wake_word"]["inference_framework"],
                vad_threshold=self.config["wake_word"]["vad_threshold"],
                prediction_frequency=self.config["wake_word"]["prediction_frequency"],
                enable_speex_noise_suppression=self.config["wake_word"][
                    "enable_speex_noise_suppression"
                ],
            )

            logger.info("✅ Optimized wake word model loaded")
            return model

        except Exception as e:
            logger.error(f"Failed to setup optimized wake word: {e}")
            return None

    def setup_optimized_whisper(self):
        """Set up Whisper with aggressive optimizations for Jetson."""
        try:
            logger.info("Setting up optimized Whisper model...")

            # Create model with optimized settings
            model = WhisperModel(
                model_size_or_path=self.config["whisper"]["model_size"],
                device=self.config["whisper"]["device"],
                compute_type=self.config["whisper"]["compute_type"],
                cpu_threads=self.config["whisper"]["cpu_threads"],
                num_workers=self.config["whisper"]["num_workers"],
            )

            logger.info("✅ Optimized Whisper model loaded")
            return model

        except Exception as e:
            logger.error(f"Failed to setup optimized Whisper: {e}")
            return None

    def test_optimized_models(self) -> Dict:
        """Test the optimized models to verify performance improvements."""
        results = {
            "wake_word": {"status": "not_tested"},
            "whisper": {"status": "not_tested"},
        }

        # Test wake word optimization
        logger.info("Testing optimized wake word model...")
        wake_word_model = self.setup_optimized_wake_word()

        if wake_word_model:
            # Create test audio (silence)
            test_audio = np.zeros(800, dtype=np.int16)  # 50ms of silence

            # Warm up
            for _ in range(5):
                wake_word_model.predict(test_audio)

            # Benchmark
            cpu_readings = []
            inference_times = []

            for _ in range(20):
                cpu_before = psutil.cpu_percent(interval=0.01)
                start_time = time.perf_counter()

                _ = wake_word_model.predict(test_audio)

                inference_time = time.perf_counter() - start_time
                cpu_after = psutil.cpu_percent(interval=0.01)

                cpu_readings.append((cpu_before + cpu_after) / 2)
                inference_times.append(inference_time * 1000)

            avg_cpu = np.mean(cpu_readings)
            avg_inference = np.mean(inference_times)

            results["wake_word"] = {
                "status": "tested",
                "avg_cpu_usage": avg_cpu,
                "avg_inference_time_ms": avg_inference,
                "meets_target": avg_cpu < 5.0,
            }

            logger.info(f"Wake word: CPU {avg_cpu:.1f}%, inference {avg_inference:.1f}ms")

        # Test Whisper optimization
        logger.info("Testing optimized Whisper model...")
        whisper_model = self.setup_optimized_whisper()

        if whisper_model:
            # Create test audio (2 seconds of sine wave)
            sample_rate = 16000
            duration = 2.0
            test_audio = np.sin(2 * np.pi * 440 * np.arange(0, duration, 1 / sample_rate))
            test_audio = (test_audio * 16383).astype(np.int16)  # Lower amplitude

            # Memory before
            process = psutil.Process()
            _ = process.memory_info().rss / 1024 / 1024  # MB

            # Warm up
            segments, _ = whisper_model.transcribe(
                test_audio[:8000],  # 0.5s warmup
                beam_size=self.config["whisper"]["beam_size"],
                best_of=self.config["whisper"]["best_of"],
                word_timestamps=self.config["whisper"]["word_timestamps"],
                condition_on_previous_text=self.config["whisper"]["condition_on_previous_text"],
            )

            # Benchmark
            start_time = time.perf_counter()

            segments, info = whisper_model.transcribe(
                test_audio,
                beam_size=self.config["whisper"]["beam_size"],
                best_of=self.config["whisper"]["best_of"],
                word_timestamps=self.config["whisper"]["word_timestamps"],
                condition_on_previous_text=self.config["whisper"]["condition_on_previous_text"],
            )

            inference_time = time.perf_counter() - start_time
            mem_after = process.memory_info().rss / 1024 / 1024  # MB

            # Calculate metrics
            rtf = inference_time / duration
            memory_usage = mem_after

            results["whisper"] = {
                "status": "tested",
                "inference_time_s": inference_time,
                "real_time_factor": rtf,
                "memory_usage_mb": memory_usage,
                "audio_duration_s": duration,
                "meets_rtf_target": rtf < 0.3,
                "meets_memory_target": memory_usage < 300,
            }

            logger.info(f"Whisper: RTF {rtf:.2f}x, memory {memory_usage:.1f}MB")

        return results

    def create_jetson_optimized_script(self) -> None:
        """Create a script with Jetson-specific optimizations."""
        script_content = '''#!/usr/bin/env python3
"""
Jetson-Optimized Audio Processing Node

This script implements audio processing optimized specifically for NVIDIA Jetson Orin Nano.
Includes CPU governor management, memory optimization, and performance monitoring.
"""

import gc
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil

# Set Jetson performance mode
def set_jetson_performance_mode():
    """Set Jetson to maximum performance mode."""
    try:
        # Set CPU governor to performance
        for cpu in range(psutil.cpu_count()):
            gov_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            if os.path.exists(gov_path):
                subprocess.run(["sudo", "sh", "-c", f"echo performance > {gov_path}"], check=True)

        # Set jetson_clocks if available (requires sudo)
        if os.path.exists("/usr/bin/jetson_clocks"):
            subprocess.run(["sudo", "/usr/bin/jetson_clocks"], check=True)

        print("✅ Jetson performance mode enabled")
        return True
    except Exception as e:
        print(f"⚠️ Could not set performance mode: {e}")
        return False

# Memory optimization
class MemoryManager:
    """Manage memory usage for optimal Jetson performance."""

    def __init__(self, max_memory_mb=6000):  # Reserve 1.5GB for system
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_threshold = 0.8  # Cleanup at 80% of max memory

    def check_memory(self):
        """Check current memory usage."""
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        return memory_usage

    def cleanup_if_needed(self):
        """Cleanup memory if threshold exceeded."""
        current_memory = self.check_memory()
        if current_memory > (self.max_memory * self.cleanup_threshold):
            print(f"🧹 Memory cleanup triggered: {current_memory/(1024*1024):.1f}MB")
            gc.collect()
            return True
        return False

# Example optimized audio processing class
class OptimizedAudioProcessor:
    """Audio processor optimized for Jetson Orin Nano."""

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.processing_count = 0

        # Set performance mode
        set_jetson_performance_mode()

    def process_audio_chunk(self, audio_data: np.ndarray):
        """Process audio with memory management."""
        self.processing_count += 1

        # Memory cleanup every 100 chunks
        if self.processing_count % 100 == 0:
            self.memory_manager.cleanup_if_needed()

        # Your audio processing here
        # ...

        return audio_data  # Placeholder

if __name__ == "__main__":
    print("🤖 Jetson Audio Optimization Demo")
    processor = OptimizedAudioProcessor()
    print("✅ Optimized audio processor ready")
'''

        script_path = PROJECT_ROOT / "scripts" / "jetson_audio_optimized.py"

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        logger.info(f"✅ Created Jetson-optimized script: {script_path}")

    def generate_optimization_report(self, test_results: Dict) -> str:
        """Generate a comprehensive optimization report."""
        report = []
        report.append("# Audio Model Optimization Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("## System Information")
        report.append(f"- Platform: {os.uname().machine}")
        report.append(f"- CPU cores: {psutil.cpu_count()}")
        report.append(f"- Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB\n")

        report.append("## Optimization Results\n")

        # Wake word results
        if test_results["wake_word"]["status"] == "tested":
            ww = test_results["wake_word"]
            status = "✅ PASS" if ww["meets_target"] else "❌ FAIL"
            report.append(f"### Wake Word Detection {status}")
            report.append(f"- CPU Usage: {ww['avg_cpu_usage']:.1f}% (target: <5%)")
            report.append(f"- Inference Time: {ww['avg_inference_time_ms']:.1f}ms")
            report.append(f"- Target Met: {ww['meets_target']}\n")

        # Whisper results
        if test_results["whisper"]["status"] == "tested":
            whisper = test_results["whisper"]
            rtf_status = "✅" if whisper["meets_rtf_target"] else "❌"
            mem_status = "✅" if whisper["meets_memory_target"] else "❌"

            report.append("### Speech Recognition")
            report.append(
                f"- Real-time Factor: {whisper['real_time_factor']:.2f}x (target: <0.3x)\
                     {rtf_status}"
            )
            report.append(
                f"- Memory Usage: {whisper['memory_usage_mb']:.1f}MB (target: <300MB) {mem_status}"
            )
            report.append(f"- Inference Time: {whisper['inference_time_s']*1000:.1f}ms")
            report.append(f"- Audio Duration: {whisper['audio_duration_s']:.1f}s\n")

        report.append("## Optimization Strategies Implemented\n")
        report.append("### Wake Word Optimization:")
        report.append("- Reduced prediction frequency to 50ms")
        report.append("- Enabled Speex noise suppression")
        report.append("- Optimized VAD threshold")

        report.append("\n### Whisper Optimization:")
        report.append("- Used int8 quantization")
        report.append("- Greedy decoding (beam_size=1)")
        report.append("- Disabled word timestamps")
        report.append("- Disabled condition_on_previous_text")
        report.append("- Reduced CPU threads to 2")

        report.append("\n## Recommendations\n")

        if not test_results["wake_word"].get("meets_target", False):
            report.append("### Wake Word Further Optimization:")
            report.append("- Consider using TensorRT optimization")
            report.append("- Try smaller VAD models")
            report.append("- Implement custom noise gate")

        if not test_results["whisper"].get("meets_rtf_target", False):
            report.append("### Whisper Further Optimization:")
            report.append("- Convert to TensorRT format")
            report.append("- Use even more aggressive quantization")
            report.append("- Implement streaming inference")

        return "\n".join(report)


def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description="Optimize audio models for Jetson")
    parser.add_argument("--test", action="store_true", help="Test optimized models")
    parser.add_argument("--config-only", action="store_true", help="Only create config files")

    args = parser.parse_args()

    optimizer = AudioOptimizer()

    # Always create optimized configuration
    optimizer.create_optimized_config()
    optimizer.create_jetson_optimized_script()

    if args.config_only:
        logger.info("✅ Configuration files created only")
        return 0

    # Test optimized models
    if args.test:
        logger.info("🧪 Testing optimized models...")
        results = optimizer.test_optimized_models()

        # Generate report
        report = optimizer.generate_optimization_report(results)

        # Save report
        report_path = PROJECT_ROOT / "audio_optimization_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"📋 Optimization report saved: {report_path}")

        # Print summary
        print("\n" + "=" * 50)
        print("AUDIO OPTIMIZATION SUMMARY")
        print("=" * 50)

        if results["wake_word"]["status"] == "tested":
            ww = results["wake_word"]
            status = "✅ PASS" if ww["meets_target"] else "❌ FAIL"
            print(f"Wake Word: {ww['avg_cpu_usage']:.1f}% CPU {status}")

        if results["whisper"]["status"] == "tested":
            whisper = results["whisper"]
            rtf_status = "✅" if whisper["meets_rtf_target"] else "❌"
            mem_status = "✅" if whisper["meets_memory_target"] else "❌"
            print(f"Whisper RTF: {whisper['real_time_factor']:.2f}x {rtf_status}")
            print(f"Whisper Memory: {whisper['memory_usage_mb']:.1f}MB {mem_status}")

    logger.info("✅ Audio model optimization completed!")
    return 0


if __name__ == "__main__":
    exit(main())
