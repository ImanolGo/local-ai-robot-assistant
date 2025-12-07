#!/usr/bin/env python3
"""
Jetson-Optimized Audio Processing Node

This script implements audio processing optimized specifically for NVIDIA Jetson Orin Nano.
Includes CPU governor management, memory optimization, and performance monitoring.
"""

import gc
import os
import subprocess

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
