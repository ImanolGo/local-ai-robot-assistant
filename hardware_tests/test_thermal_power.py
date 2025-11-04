#!/usr/bin/env python3
"""
Thermal and Power Testing for NVIDIA Jetson Orin Nano

This script tests and monitors:
- Power consumption under idle and full load
- CPU/GPU temperatures during extended operation
- Thermal throttling behavior
- Cooling solution adequacy
- Automatically generates comprehensive documentation

Usage:
    python3 test_thermal_power.py [--idle-only | --load-only | --doc-only | --help]

Author: Local AI Robot Assistant Project
License: MIT
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil


@dataclass
class ThermalReading:
    """Single thermal and power measurement."""

    timestamp: datetime
    cpu_temp: float
    gpu_temp: float
    power_consumption: float
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    cpu_freq: int
    gpu_freq: int
    fan_speed: Optional[int] = None


class JetsonMonitor:
    """Monitor Jetson Orin Nano thermal and power characteristics."""

    def __init__(self):
        self.readings: List[ThermalReading] = []
        self.monitoring = False
        self.monitor_thread = None

        # Thermal zones (Jetson-specific paths)
        self.thermal_zones = {
            "cpu": "/sys/class/thermal/thermal_zone0/temp",
            "gpu": "/sys/class/thermal/thermal_zone1/temp",
            "aux": "/sys/class/thermal/thermal_zone2/temp",
            "thermal": "/sys/class/thermal/thermal_zone3/temp",
        }

        # INA3221 power monitor paths (hwmon interface)
        self.ina3221_base = "/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040/hwmon/hwmon1"
        self.power_sensors = {
            "VDD_IN": {
                "voltage": f"{self.ina3221_base}/in1_input",
                "current": f"{self.ina3221_base}/curr1_input",
            },
            "VDD_CPU_GPU_CV": {
                "voltage": f"{self.ina3221_base}/in2_input",
                "current": f"{self.ina3221_base}/curr2_input",
            },
            "VDD_SOC": {
                "voltage": f"{self.ina3221_base}/in3_input",
                "current": f"{self.ina3221_base}/curr3_input",
            },
        }

        # Legacy power paths (for compatibility)
        self.power_paths = {
            "total": "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
            "cpu": "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power1_input",
            "gpu": "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power2_input",
        }

        # Frequency paths (updated for Orin Nano Super)
        self.freq_paths = {
            "cpu": "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
            "gpu": "/sys/kernel/debug/clk/gpc0clk/clk_rate",  # Updated for Orin Nano Super
        }

        print("Jetson Thermal & Power Monitor initialized")
        self._validate_sensors()

    def _validate_sensors(self) -> None:
        """Validate that thermal and power sensors are accessible."""
        print("Validating sensors...")

        # Check thermal zones
        available_thermal = []
        for name, path in self.thermal_zones.items():
            if os.path.exists(path):
                available_thermal.append(name)
                print(f"✓ {name} thermal sensor: {path}")
            else:
                print(f"✗ {name} thermal sensor not found: {path}")

        if not available_thermal:
            print("⚠️  No thermal sensors found! Results may be incomplete.")

        # Check INA3221 power sensors
        available_power = []
        for name, paths in self.power_sensors.items():
            voltage_path = paths["voltage"]
            current_path = paths["current"]
            if os.path.exists(voltage_path) and os.path.exists(current_path):
                available_power.append(name)
                print(f"✓ {name} power sensor: voltage & current")
            else:
                print(f"✗ {name} power sensor not found")

        # Check legacy power monitoring
        for name, path in self.power_paths.items():
            if os.path.exists(path):
                available_power.append(name)
                print(f"✓ {name} power sensor: {path}")
            else:
                print(f"✗ {name} power sensor not found: {path}")

        if not available_power:
            print("⚠️  No power sensors found! Will use CPU-based estimation.")

        # Check frequency monitoring
        for name, path in self.freq_paths.items():
            if os.path.exists(path):
                print(f"✓ {name} frequency sensor: {path}")
            else:
                print(f"✗ {name} frequency sensor not found: {path}")

    def _read_temp(self, zone_name: str) -> Optional[float]:
        """Read temperature from thermal zone (°C)."""
        path = self.thermal_zones.get(zone_name)
        if not path or not os.path.exists(path):
            return None

        try:
            with open(path, "r") as f:
                # Thermal zone temps are in millidegrees
                temp_millidegrees = int(f.read().strip())
                return temp_millidegrees / 1000.0
        except (IOError, ValueError) as e:
            print(f"Error reading {zone_name} temp: {e}")
            return None

    def _read_power(self, channel: str = "total") -> Optional[float]:
        """Read power consumption (watts)."""
        # Try legacy path first (for compatibility)
        path = self.power_paths.get(channel)
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    # Power is typically in microwatts
                    power_microwatts = int(f.read().strip())
                    return power_microwatts / 1000000.0  # Convert to watts
            except (IOError, ValueError) as e:
                print(f"Error reading {channel} power: {e}")

        # If legacy path failed, calculate from INA3221 hwmon voltage/current
        # Calculate total power from all rails
        total_power = 0.0
        rails_read = 0

        for rail_name, paths in self.power_sensors.items():
            voltage_path = paths["voltage"]
            current_path = paths["current"]

            if os.path.exists(voltage_path) and os.path.exists(current_path):
                try:
                    with open(voltage_path, "r") as f:
                        voltage_mv = int(f.read().strip())  # millivolts
                    with open(current_path, "r") as f:
                        current_ma = int(f.read().strip())  # milliamps

                    # Calculate power: P = V * I
                    # voltage is in mV, current is in mA
                    # (mV * mA) / 1000 = mW, then / 1000 = W
                    power_watts = (voltage_mv * current_ma) / 1000000.0
                    total_power += power_watts
                    rails_read += 1

                except (IOError, ValueError) as e:
                    print(f"Error reading {rail_name} power: {e}")
                    continue

        if rails_read > 0:
            return total_power

        return None

    def _read_frequency(self, component: str) -> Optional[int]:
        """Read frequency (Hz)."""
        path = self.freq_paths.get(component)
        if not path or not os.path.exists(path):
            return None

        try:
            with open(path, "r") as f:
                freq = int(f.read().strip())
                # CPU freq is in kHz, GPU freq is in Hz
                if component == "cpu":
                    return freq * 1000  # Convert kHz to Hz
                return freq
        except (IOError, ValueError) as e:
            print(f"Error reading {component} frequency: {e}")
            return None

    def _get_gpu_usage(self) -> float:
        """Get GPU utilization percentage using nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass
        return 0.0

    def _get_fan_speed(self) -> Optional[int]:
        """Get fan speed (RPM) if available."""
        fan_paths = [
            "/sys/class/hwmon/hwmon0/fan1_input",
            "/sys/class/hwmon/hwmon1/fan1_input",
            "/sys/class/thermal/cooling_device0/cur_state",
        ]

        for path in fan_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        return int(f.read().strip())
                except (IOError, ValueError):
                    continue
        return None

    def take_reading(self) -> ThermalReading:
        """Take a single thermal and power reading."""
        timestamp = datetime.now()

        # Temperature readings
        cpu_temp = self._read_temp("cpu") or 0.0
        gpu_temp = self._read_temp("gpu") or self._read_temp("thermal") or 0.0

        # Power reading
        power_consumption = self._read_power("total") or 0.0

        # System usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        gpu_usage = self._get_gpu_usage()
        memory_usage = psutil.virtual_memory().percent

        # Frequencies
        cpu_freq = self._read_frequency("cpu") or 0
        gpu_freq = self._read_frequency("gpu") or 0

        # Fan speed
        fan_speed = self._get_fan_speed()

        reading = ThermalReading(
            timestamp=timestamp,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            power_consumption=power_consumption,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            memory_usage=memory_usage,
            cpu_freq=cpu_freq,
            gpu_freq=gpu_freq,
            fan_speed=fan_speed,
        )

        return reading

    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous monitoring."""
        if self.monitoring:
            print("Monitoring already active")
            return

        self.monitoring = True
        self.readings.clear()

        def monitor_loop():
            while self.monitoring:
                reading = self.take_reading()
                self.readings.append(reading)
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Started continuous monitoring")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("Stopped monitoring")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from collected readings."""
        if not self.readings:
            return {}

        cpu_temps = [r.cpu_temp for r in self.readings if r.cpu_temp > 0]
        gpu_temps = [r.gpu_temp for r in self.readings if r.gpu_temp > 0]
        powers = [r.power_consumption for r in self.readings if r.power_consumption > 0]
        cpu_usage = [r.cpu_usage for r in self.readings]
        gpu_usage = [r.gpu_usage for r in self.readings]

        stats = {"duration": len(self.readings), "readings_count": len(self.readings)}

        if cpu_temps:
            stats["cpu_temp"] = {
                "min": min(cpu_temps),
                "max": max(cpu_temps),
                "avg": np.mean(cpu_temps),
                "std": np.std(cpu_temps),
            }

        if gpu_temps:
            stats["gpu_temp"] = {
                "min": min(gpu_temps),
                "max": max(gpu_temps),
                "avg": np.mean(gpu_temps),
                "std": np.std(gpu_temps),
            }

        if powers:
            stats["power"] = {
                "min": min(powers),
                "max": max(powers),
                "avg": np.mean(powers),
                "std": np.std(powers),
            }

        if cpu_usage:
            stats["cpu_usage"] = {
                "min": min(cpu_usage),
                "max": max(cpu_usage),
                "avg": np.mean(cpu_usage),
                "std": np.std(cpu_usage),
            }

        if gpu_usage:
            stats["gpu_usage"] = {
                "min": min(gpu_usage),
                "max": max(gpu_usage),
                "avg": np.mean(gpu_usage),
                "std": np.std(gpu_usage),
            }

        return stats


class LoadGenerator:
    """Generate CPU and GPU load for testing."""

    def __init__(self):
        self.cpu_processes = []
        self.gpu_processes = []
        self.running = False

    def start_cpu_load(self, num_cores: Optional[int] = None) -> None:
        """Start CPU stress test."""
        if num_cores is None:
            num_cores = psutil.cpu_count()

        print(f"Starting CPU load on {num_cores} cores...")

        def cpu_stress():
            # CPU-intensive computation
            while self.running:
                sum(i * i for i in range(10000))

        self.running = True
        for _ in range(num_cores):
            process = threading.Thread(target=cpu_stress, daemon=True)
            process.start()
            self.cpu_processes.append(process)

    def start_gpu_load(self) -> None:
        """Start GPU stress test using nvidia-smi or CUDA operations."""
        print("Starting GPU load...")

        # Try to start GPU burn test
        try:
            # Simple CUDA matrix multiplication stress test
            gpu_script = """import subprocess
import time
import sys

# Try method 1: Use jetson_clocks for max performance (try with sudo if needed)
try:
    print("Attempting to use jetson_clocks for maximum performance...", flush=True)
    # First try without sudo
    result = subprocess.run(['jetson_clocks'], check=True, capture_output=True, text=True)
    print("jetson_clocks enabled - running at maximum clocks", flush=True)
except subprocess.CalledProcessError as e:
    if "root user" in str(e.stderr) or "Permission denied" in str(e.stderr):
        print("jetson_clocks requires root - attempting with sudo...", flush=True)
        try:
            # Try with sudo, but don't fail the whole script if it doesn't work
            result = subprocess.run(['sudo', 'jetson_clocks'], check=True, capture_output=True,\
                  text=True)
            print("jetson_clocks enabled with sudo - running at maximum clocks", flush=True)
        except subprocess.CalledProcessError:
            print("sudo jetson_clocks failed - continuing with normal clocks", flush=True)
        except Exception:
            print("Could not run jetson_clocks with sudo - continuing with normal clocks",\
                  flush=True)
    else:
        print(f"jetson_clocks failed: {e.stderr}", flush=True)
except FileNotFoundError:
    print("jetson_clocks not found - running without clock optimization", flush=True)
except Exception as e:
    print(f"jetson_clocks error: {e}", flush=True)

# Try method 2: CuPy
try:
    import cupy as cp
    print("Using CuPy for GPU stress test", flush=True)

    while True:
        # Large matrix multiplication on GPU
        a = cp.random.random((4096, 4096), dtype=cp.float32)
        b = cp.random.random((4096, 4096), dtype=cp.float32)
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()

except ImportError:
    # Try method 3: TensorFlow
    print("CuPy not available - attempting tensorflow GPU stress", flush=True)
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Using TensorFlow with GPU: {physical_devices}", flush=True)
            while True:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([4096, 4096])
                    b = tf.random.normal([4096, 4096])
                    c = tf.matmul(a, b)
        else:
            print("No GPU found by TensorFlow", flush=True)
            sys.exit(1)
    except ImportError:
        # Try method 4: PyTorch
        print("TensorFlow not available - attempting PyTorch GPU stress", flush=True)
        try:
            import torch
            if torch.cuda.is_available():
                print(f"Using PyTorch with CUDA", flush=True)
                device = torch.device('cuda')
                while True:
                    a = torch.randn(4096, 4096, device=device)
                    b = torch.randn(4096, 4096, device=device)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
            else:
                print("CUDA not available in PyTorch", flush=True)
                sys.exit(1)
        except ImportError:
            print("ERROR: No GPU acceleration libraries available (CuPy/TensorFlow/PyTorch)",\
                  flush=True)
            print("Install one of: 'pip3 install cupy-cuda12x' (recommended for Jetson)",\
                  flush=True)
            sys.exit(1)
"""

            # Write temporary GPU stress script in current directory
            gpu_script_path = Path("gpu_stress_temp.py")
            with open(gpu_script_path, "w") as f:
                f.write(gpu_script)

            # Make it executable
            gpu_script_path.chmod(0o755)

            # Start GPU stress process (temporarily show output for debugging)
            process = subprocess.Popen(
                [sys.executable, str(gpu_script_path)]
                # Removed stdout/stderr suppression to see what's happening
            )
            self.gpu_processes.append(process)

        except Exception as e:
            print(f"Could not start GPU stress test: {e}")

    def stop_load(self) -> None:
        """Stop all load generation."""
        print("Stopping load generation...")
        self.running = False

        # Stop CPU processes
        for process in self.cpu_processes:
            if process.is_alive():
                process.join(timeout=1.0)
        self.cpu_processes.clear()

        # Stop GPU processes
        for process in self.gpu_processes:
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self.gpu_processes.clear()


def test_idle_power():
    """Test power consumption under idle conditions."""
    print("\n" + "=" * 60)
    print("IDLE POWER CONSUMPTION TEST")
    print("=" * 60)

    monitor = JetsonMonitor()

    print("Testing idle power consumption for 60 seconds...")
    print("Please ensure no intensive processes are running.")
    input("Press Enter to start idle test...")

    monitor.start_monitoring(interval=2.0)

    try:
        # Wait for 60 seconds of idle monitoring
        for i in range(30):
            time.sleep(2)
            if i % 5 == 0:
                print(f"Idle test progress: {i*2}/60 seconds")

    finally:
        monitor.stop_monitoring()

    stats = monitor.get_summary_stats()

    print("\n--- IDLE TEST RESULTS ---")
    if "power" in stats:
        print(f"Average power consumption: {stats['power']['avg']:.2f} W")
        print(f"Power range: {stats['power']['min']:.2f} - {stats['power']['max']:.2f} W")

    if "cpu_temp" in stats:
        print(f"Average CPU temperature: {stats['cpu_temp']['avg']:.1f}°C")

    if "gpu_temp" in stats:
        print(f"Average GPU temperature: {stats['gpu_temp']['avg']:.1f}°C")

    return stats


def test_load_power():
    """Test power consumption under full load."""
    print("\n" + "=" * 60)
    print("FULL LOAD POWER CONSUMPTION TEST")
    print("=" * 60)

    monitor = JetsonMonitor()
    load_gen = LoadGenerator()

    print("Testing power consumption under full CPU+GPU load for 300 seconds...")
    print("This will stress both CPU and GPU simultaneously.")
    input("Press Enter to start load test...")

    monitor.start_monitoring(interval=2.0)

    try:
        # Start load generation
        load_gen.start_cpu_load()
        load_gen.start_gpu_load()

        print("Load generation started. Monitoring for 5 minutes...")

        # Monitor for 5 minutes under load
        for i in range(150):  # 150 * 2 = 300 seconds
            time.sleep(2)
            if i % 15 == 0:  # Every 30 seconds
                current_reading = monitor.take_reading()
                print(
                    f"Progress: {i*2}/300s | "
                    f"CPU: {current_reading.cpu_temp:.1f}°C | "
                    f"GPU: {current_reading.gpu_temp:.1f}°C | "
                    f"Power: {current_reading.power_consumption:.1f}W"
                )

    finally:
        load_gen.stop_load()
        monitor.stop_monitoring()

    stats = monitor.get_summary_stats()

    print("\n--- FULL LOAD TEST RESULTS ---")
    if "power" in stats:
        print(f"Average power consumption: {stats['power']['avg']:.2f} W")
        print(f"Peak power consumption: {stats['power']['max']:.2f} W")
        print(f"Power range: {stats['power']['min']:.2f} - {stats['power']['max']:.2f} W")

    if "cpu_temp" in stats:
        print(f"Average CPU temperature: {stats['cpu_temp']['avg']:.1f}°C")
        print(f"Peak CPU temperature: {stats['cpu_temp']['max']:.1f}°C")

    if "gpu_temp" in stats:
        print(f"Average GPU temperature: {stats['gpu_temp']['avg']:.1f}°C")
        print(f"Peak GPU temperature: {stats['gpu_temp']['max']:.1f}°C")

    # Check for thermal throttling
    max_cpu_temp = stats.get("cpu_temp", {}).get("max", 0)
    max_gpu_temp = stats.get("gpu_temp", {}).get("max", 0)

    print("\n--- THERMAL ANALYSIS ---")
    if max_cpu_temp > 80:
        print(f"⚠️  CPU reached high temperature: {max_cpu_temp:.1f}°C")
    if max_gpu_temp > 80:
        print(f"⚠️  GPU reached high temperature: {max_gpu_temp:.1f}°C")

    if max_cpu_temp > 90 or max_gpu_temp > 90:
        print("🔥 WARNING: Temperatures exceeded 90°C - thermal throttling likely!")
    elif max_cpu_temp > 80 or max_gpu_temp > 80:
        print("⚠️  Temperatures exceeded 80°C - monitor cooling solution")
    else:
        print("✓ Temperatures remained within acceptable range")

    return stats


def test_thermal_throttling():
    """Test thermal throttling behavior during extended load."""
    print("\n" + "=" * 60)
    print("THERMAL THROTTLING TEST")
    print("=" * 60)

    monitor = JetsonMonitor()
    load_gen = LoadGenerator()

    print("Testing thermal throttling behavior during extended load...")
    print("This test will run until thermal throttling is detected or 20 minutes.")
    print("Monitor temperatures carefully!")

    response = input("Continue with thermal throttling test? (y/N): ")
    if response.lower() != "y":
        print("Thermal throttling test skipped.")
        return {}

    monitor.start_monitoring(interval=1.0)

    throttling_detected = False
    max_duration = 20 * 60  # 20 minutes max

    try:
        load_gen.start_cpu_load()
        load_gen.start_gpu_load()

        print("Load started. Monitoring for thermal throttling...")

        start_time = time.time()
        last_cpu_freq = None
        last_gpu_freq = None

        while (time.time() - start_time) < max_duration and not throttling_detected:
            time.sleep(5)
            current_reading = monitor.take_reading()

            elapsed = int(time.time() - start_time)
            print(
                f"Time: {elapsed:3d}s | "
                f"CPU: {current_reading.cpu_temp:5.1f}"
                f"°C ({current_reading.cpu_freq//1000000:4d}MHz) | "
                f"GPU: {current_reading.gpu_temp:5.1f}"
                f"°C ({current_reading.gpu_freq//1000000:4d}MHz) | "
                f"Power: {current_reading.power_consumption:5.1f}W"
            )

            # Check for frequency reduction (throttling)
            if last_cpu_freq and current_reading.cpu_freq < (last_cpu_freq * 0.8):
                print("🔥 CPU throttling detected!")
                throttling_detected = True

            if last_gpu_freq and current_reading.gpu_freq < (last_gpu_freq * 0.8):
                print("🔥 GPU throttling detected!")
                throttling_detected = True

            # Emergency stop if temperatures get too high
            if current_reading.cpu_temp > 95 or current_reading.gpu_temp > 95:
                print("🚨 EMERGENCY STOP: Temperature > 95°C!")
                break

            last_cpu_freq = current_reading.cpu_freq
            last_gpu_freq = current_reading.gpu_freq

    finally:
        load_gen.stop_load()
        time.sleep(5)  # Let system cool down a bit
        monitor.stop_monitoring()

    stats = monitor.get_summary_stats()

    print("\n--- THERMAL THROTTLING TEST RESULTS ---")
    if throttling_detected:
        print("✓ Thermal throttling detected - system is protecting itself")
    else:
        print("ℹ️  No thermal throttling detected during test period")

    if "cpu_temp" in stats:
        print(f"Peak CPU temperature: {stats['cpu_temp']['max']:.1f}°C")
    if "gpu_temp" in stats:
        print(f"Peak GPU temperature: {stats['gpu_temp']['max']:.1f}°C")

    return stats


def save_results(idle_stats: Dict, load_stats: Dict, throttle_stats: Dict):
    """Save test results to JSON file."""
    results = {
        "test_date": datetime.now().isoformat(),
        "jetson_model": "Orin Nano",
        "tests": {
            "idle_power": idle_stats,
            "full_load": load_stats,
            "thermal_throttling": throttle_stats,
        },
    }

    results_path = Path("thermal_power_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def generate_documentation(idle_stats: Dict, load_stats: Dict, throttle_stats: Dict):
    """Generate comprehensive markdown documentation of test results."""
    timestamp = datetime.now()

    # Prepare the documentation content
    doc_content = f"""# Thermal & Power Testing Results

**Test Date**: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}
**Platform**: NVIDIA Jetson Orin Nano (8GB)
**Test Duration**: {timestamp.strftime("%B %d, %Y")}

## Executive Summary

This document contains the results of comprehensive thermal and\
      power testing for the Jetson Orin Nano platform used in the Local AI Robot Assistant project.

## Test Overview

Three primary tests were conducted:
1. **Idle Power Consumption Test** - 60 seconds of baseline measurements
2. **Full Load Test** - 5 minutes of CPU+GPU stress testing
3. **Thermal Throttling Test** - Extended load until throttling detection

---

## Test Results

### 1. Idle Power Consumption
"""

    # Add idle power results
    if idle_stats:
        if "power" in idle_stats:
            doc_content += f"""
**Power Consumption (Idle)**:
- Average: {idle_stats['power']['avg']:.2f}W
- Range: {idle_stats['power']['min']:.2f}W - {idle_stats['power']['max']:.2f}W
- Standard Deviation: {idle_stats['power']['std']:.2f}W
"""
        else:
            doc_content += "\n⚠️ Power measurements not available during idle test.\n"

        if "cpu_temp" in idle_stats:
            doc_content += f"""
**Temperature (Idle)**:
- CPU Average: {idle_stats['cpu_temp']['avg']:.1f}°C
- CPU Range: {idle_stats['cpu_temp']['min']:.1f}°C - {idle_stats['cpu_temp']['max']:.1f}°C
"""

        if "gpu_temp" in idle_stats:
            doc_content += f"""- GPU Average: {idle_stats['gpu_temp']['avg']:.1f}°C
- GPU Range: {idle_stats['gpu_temp']['min']:.1f}°C - {idle_stats['gpu_temp']['max']:.1f}°C
"""
    else:
        doc_content += "\n❌ Idle test was not completed.\n"

    # Add full load results
    doc_content += "\n### 2. Full Load Test (CPU + GPU Stress)\n"

    if load_stats:
        if "power" in load_stats:
            doc_content += f"""
**Power Consumption (Full Load)**:
- Average: {load_stats['power']['avg']:.2f}W
- Peak: {load_stats['power']['max']:.2f}W
- Range: {load_stats['power']['min']:.2f}W - {load_stats['power']['max']:.2f}W
- Standard Deviation: {load_stats['power']['std']:.2f}W
"""
        else:
            doc_content += "\n⚠️ Power measurements not available during load test.\n"

        if "cpu_temp" in load_stats and "gpu_temp" in load_stats:
            cpu_max = load_stats["cpu_temp"]["max"]
            gpu_max = load_stats["gpu_temp"]["max"]
            peak_temp = max(cpu_max, gpu_max)

            doc_content += f"""
**Temperature (Full Load)**:
- CPU Average: {load_stats['cpu_temp']['avg']:.1f}°C
- CPU Peak: {cpu_max:.1f}°C
- GPU Average: {load_stats['gpu_temp']['avg']:.1f}°C
- GPU Peak: {gpu_max:.1f}°C
- Overall Peak: {peak_temp:.1f}°C
"""

        if "cpu_usage" in load_stats:
            doc_content += f"""
**System Utilization (Full Load)**:
- CPU Average: {load_stats['cpu_usage']['avg']:.1f}%
- CPU Peak: {load_stats['cpu_usage']['max']:.1f}%
"""

        if "gpu_usage" in load_stats:
            doc_content += f"- GPU Average: {load_stats['gpu_usage']['avg']:.1f}%\n- GPU Peak:\
                  {load_stats['gpu_usage']['max']:.1f}%\n"
    else:
        doc_content += "\n❌ Full load test was not completed.\n"

    # Add thermal throttling results
    doc_content += "\n### 3. Thermal Throttling Test\n"

    if throttle_stats:
        if "cpu_temp" in throttle_stats and "gpu_temp" in throttle_stats:
            cpu_max = throttle_stats["cpu_temp"]["max"]
            gpu_max = throttle_stats["gpu_temp"]["max"]
            peak_temp = max(cpu_max, gpu_max)

            doc_content += f"""
**Thermal Throttling Results**:
- CPU Peak Temperature: {cpu_max:.1f}°C
- GPU Peak Temperature: {gpu_max:.1f}°C
- Overall Peak Temperature: {peak_temp:.1f}°C
- Test Duration: {throttle_stats.get('duration', 'Unknown')} seconds
"""

            if peak_temp > 90:
                doc_content += "- **Throttling Status**: ⚠️ High temperature detected (>90°C)\n"
            elif peak_temp > 80:
                doc_content += "- **Throttling Status**: ⚠️ Elevated temperature (>80°C)\n"
            else:
                doc_content += "- **Throttling Status**: ✅ Temperature within safe range\n"
        else:
            doc_content += "\n⚠️ Temperature data not available for throttling test.\n"
    else:
        doc_content += "\n❌ Thermal throttling test was not completed or skipped.\n"

    # Analysis and recommendations
    doc_content += "\n---\n\n## Analysis & Recommendations\n"

    # Power analysis
    max_power = load_stats.get("power", {}).get("max", 0) if load_stats else 0
    avg_power = load_stats.get("power", {}).get("avg", 0) if load_stats else 0
    idle_power = idle_stats.get("power", {}).get("avg", 0) if idle_stats else 0

    if max_power > 0:
        doc_content += f"""
### Power Consumption Analysis

- **Idle Power**: {idle_power:.2f}W
- **Average Load Power**: {avg_power:.2f}W
- **Peak Power**: {max_power:.2f}W
- **Power Increase**: {(avg_power/idle_power - 1)*100:.1f}% from idle to load
"""

        if max_power > 15:
            doc_content += f"""
⚠️ **Power Recommendation**: Peak power consumption ({max_power:.1f}W) is high. Consider:
- Upgrading to a higher capacity power supply (20W+ recommended)
- Monitoring power consumption during extended AI workloads
- Implementing power management strategies for battery operation
"""
        elif max_power > 10:
            doc_content += f"""
✅ **Power Status**: Peak power consumption ({max_power:.1f}W) is moderate but acceptable.
- Current power supply should be adequate for most workloads
- Monitor power during simultaneous AI model inference
"""
        else:
            doc_content += f"""
✅ **Power Status**: Peak power consumption ({max_power:.1f}W) is excellent.
- Well within Jetson Orin Nano specifications
- Suitable for battery-powered operation
"""

    # Thermal analysis
    max_temp = 0
    if load_stats:
        cpu_max = load_stats.get("cpu_temp", {}).get("max", 0)
        gpu_max = load_stats.get("gpu_temp", {}).get("max", 0)
        max_temp = max(cpu_max, gpu_max)

    if max_temp > 0:
        doc_content += f"""
### Thermal Analysis

- **Peak Operating Temperature**: {max_temp:.1f}°C
"""

        if max_temp > 90:
            doc_content += f"""
🔥 **Thermal Recommendation**: Critical temperature reached ({max_temp:.1f}°C).\
      URGENT ACTION REQUIRED:
- Install active cooling (fan) immediately
- Consider heat sink upgrade
- Reduce ambient temperature
- Implement thermal throttling in software
- Monitor for thermal damage
"""
        elif max_temp > 85:
            doc_content += f"""
⚠️ **Thermal Recommendation**: High temperature detected ({max_temp:.1f}°C). Action needed:
- Install active cooling (fan) recommended
- Ensure adequate ventilation around device
- Monitor temperatures during extended AI workloads
- Consider heat sink upgrade
"""
        elif max_temp > 80:
            doc_content += f"""
⚠️ **Thermal Status**: Elevated temperature ({max_temp:.1f}°C). Monitor during extended use:
- Current cooling may be adequate for short workloads
- Consider fan for continuous operation
- Ensure good airflow around device
"""
        elif max_temp > 75:
            doc_content += f"""
✅ **Thermal Status**: Temperature ({max_temp:.1f}°C) is acceptable but monitor:
- Good for continuous operation
- Passive cooling appears adequate
- Keep ambient temperature reasonable
"""
        else:
            doc_content += f"""
✅ **Thermal Status**: Excellent temperature control ({max_temp:.1f}°C).
- Well within safe operating range
- Current cooling solution is more than adequate
- Suitable for enclosed environments
"""

    # AI workload implications
    doc_content += """
### AI Workload Implications

Based on these results, the platform can handle:

**Recommended AI Model Loading Strategy**:
"""

    if max_power < 12 and max_temp < 80:
        doc_content += """
- ✅ Simultaneous YOLO + Depth estimation + LLM inference
- ✅ Continuous perception pipeline operation
- ✅ Real-time audio processing alongside vision
- ✅ Extended autonomous operation (>30 minutes)
"""
    elif max_power < 15 and max_temp < 85:
        doc_content += """
- ✅ YOLO + Depth estimation simultaneously
- ⚠️ LLM inference (with thermal monitoring)
- ✅ Sequential model loading (unload vision during LLM inference)
- ⚠️ Extended operation with monitoring
"""
    else:
        doc_content += """
- ⚠️ Sequential model loading only (avoid simultaneous inference)
- ⚠️ Implement mandatory cooling periods
- ⚠️ Monitor power supply capacity
- ❌ Avoid extended continuous operation without thermal management
"""

    doc_content += f"""
**Memory Management Recommendations**:
- Implement lazy loading for LLM (load only when needed)
- Unload perception models during complex LLM inference
- Monitor system temperature before loading models
- Set thermal limits in software (max temp: {75 if max_temp > 80 else 80}°C)

---

## Hardware Configuration

**Cooling Solution**: {"Active cooling required" if max_temp > 80 else "Passive cooling adequate"}
**Power Supply**: {"Upgrade recommended (>20W)" if max_power > 15 else "Current supply adequate"}
**Operating Environment**: {
    "Controlled temperature environment recommended"
    if max_temp > 85
    else "Standard environment acceptable"
}

## Test Configuration

- **Test Script**: `hardware_tests/test_thermal_power.py`
- **Results File**: `thermal_power_results.json`
- **Documentation**: Generated automatically

---

*This document was automatically generated by the thermal/power testing script.*
"""

    # Save the documentation
    docs_dir = Path("docs")
    if not docs_dir.exists():
        docs_dir = Path("../docs")  # Try parent directory if running from hardware_tests/

    if docs_dir.exists():
        doc_path = docs_dir / "thermal_power_validation_report.md"
    else:
        doc_path = Path("thermal_power_validation_report.md")

    with open(doc_path, "w") as f:
        f.write(doc_content)

    print(f"\n📄 Documentation generated: {doc_path}")
    return doc_path


def update_implementation_plan(doc_path: Path):
    """Update the implementation plan to mark thermal testing as complete."""
    impl_plan_path = Path("docs/implementation_plan.md")
    if not impl_plan_path.exists():
        impl_plan_path = Path("../docs/implementation_plan.md")

    if not impl_plan_path.exists():
        print("⚠️  Could not find implementation_plan.md to update")
        return

    try:
        with open(impl_plan_path, "r") as f:
            content = f.read()

        # Update the thermal testing section
        updated_content = content.replace(
            "- [x] Create `hardware_tests/test_thermal_power.py`",
            f"- [x] Create `hardware_tests/test_thermal_power.py`\n- [x]\
                  Document thermal/power validation results: `{doc_path.name}`",
        )

        # Add a note about completion
        if (
            "**Test Script Requirements**:" in updated_content
            and "- [x] Document thermal/power validation results:" in updated_content
        ):
            section_marker = "# hardware_tests/test_thermal_power.py"
            if section_marker in updated_content:
                updated_content = updated_content.replace(
                    "```",
                    "# ✅ COMPLETED: Full thermal/power validation documented\n```",
                    1,  # Only replace the first occurrence in the thermal section
                )

        with open(impl_plan_path, "w") as f:
            f.write(updated_content)

        print(f"✅ Updated implementation plan: {impl_plan_path}")

    except Exception as e:
        print(f"⚠️  Could not update implementation plan: {e}")


def load_existing_results() -> tuple[Dict, Dict, Dict]:
    """Load existing test results if available."""
    results_path = Path("thermal_power_results.json")
    if not results_path.exists():
        return {}, {}, {}

    try:
        with open(results_path, "r") as f:
            data = json.load(f)

        tests = data.get("tests", {})
        return (
            tests.get("idle_power", {}),
            tests.get("full_load", {}),
            tests.get("thermal_throttling", {}),
        )
    except Exception as e:
        print(f"⚠️  Could not load existing results: {e}")
        return {}, {}, {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Thermal and Power Testing for NVIDIA Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_thermal_power.py              # Run all tests
  python3 test_thermal_power.py --idle-only  # Run only idle test
  python3 test_thermal_power.py --load-only  # Run only load test
  python3 test_thermal_power.py --doc-only   # Generate docs from existing results
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--idle-only",
        action="store_true",
        help="Run only the idle power consumption test",
    )
    group.add_argument("--load-only", action="store_true", help="Run only the full load test")
    group.add_argument(
        "--throttle-only",
        action="store_true",
        help="Run only the thermal throttling test",
    )
    group.add_argument(
        "--doc-only",
        action="store_true",
        help="Generate documentation from existing results (no testing)",
    )

    return parser.parse_args()


def main():
    """Run thermal and power tests based on command line arguments."""
    args = parse_arguments()

    print("NVIDIA Jetson Orin Nano - Thermal & Power Testing")
    print("=" * 60)

    # Check if running as root (needed for some power sensors)
    if os.geteuid() != 0:
        print("⚠️  Running without root privileges. Some sensors may not be accessible.")
        print("⚠️  Script will attempt to use 'sudo jetson_clocks' automatically for GPU stress.")
        print("   For full power monitoring without sudo prompts, you can run with:")
        print(f"   sudo {sys.executable} {' '.join(sys.argv)}")
        print("   (This preserves your virtual environment)")

    # Load existing results
    idle_stats, load_stats, throttle_stats = load_existing_results()

    # Run tests based on arguments
    if args.doc_only:
        print("\n📄 Generating documentation from existing results...")
        if not any([idle_stats, load_stats, throttle_stats]):
            print("❌ No existing results found. Run tests first.")
            sys.exit(1)
    elif args.idle_only:
        print("\n🔋 Running idle power test only...")
        idle_stats = test_idle_power()
    elif args.load_only:
        print("\n🔥 Running full load test only...")
        load_stats = test_load_power()
    elif args.throttle_only:
        print("\n🌡️  Running thermal throttling test only...")
        throttle_stats = test_thermal_throttling()
    else:
        # Run all tests
        idle_stats = test_idle_power()
        load_stats = test_load_power()
        throttle_stats = test_thermal_throttling()

    # Save results (only if we ran tests)
    if not args.doc_only:
        save_results(idle_stats, load_stats, throttle_stats)

    # Generate comprehensive documentation
    doc_path = generate_documentation(idle_stats, load_stats, throttle_stats)

    # Update implementation plan
    update_implementation_plan(doc_path)

    print("\n" + "=" * 60)
    print("THERMAL & POWER TESTING COMPLETE")
    print("=" * 60)

    # Summary recommendations
    print("\n--- SUMMARY RECOMMENDATIONS ---")

    max_power = load_stats.get("power", {}).get("max", 0)
    if max_power > 15:
        print(
            f"⚠️  Peak power consumption ({max_power:.1f}W)"
            f" is high. Consider power supply capacity."
        )
    elif max_power > 0:
        print(f"✓ Peak power consumption ({max_power:.1f}W) is within expected range.")
    else:
        print("ℹ️  Power measurements not available")

    max_temp = max(
        load_stats.get("cpu_temp", {}).get("max", 0),
        load_stats.get("gpu_temp", {}).get("max", 0),
    )

    if max_temp > 85:
        print(f"🔥 Peak temperature ({max_temp:.1f}°C) requires improved cooling.")
    elif max_temp > 75:
        print(f"⚠️  Peak temperature ({max_temp:.1f}°C) - monitor cooling during extended use.")
    elif max_temp > 0:
        print(f"✓ Peak temperature ({max_temp:.1f}°C) is acceptable.")
    else:
        print("ℹ️  Temperature measurements not available")

    print("\n📄 Detailed results: thermal_power_results.json")
    print(f"📄 Full documentation: {doc_path}")
    print("\n✅ Documentation automatically generated and implementation plan updated!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)
