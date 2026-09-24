import argparse
import base64
import io
import os
import threading
import time

import numpy as np
import psutil
import requests

# --- CONFIGURATION ---
MODEL_NAME = "moondream"  # The best small VLM for Jetson
IMAGE_PATH = "assets/test_image.png"  # Will check for this or create a dummy
OLLAMA_URL = "http://localhost:11434/api/generate"

NUM_RUNS = 10
MAX_TOKENS = 128

# Global variable for memory monitoring
max_memory_mb = 0
monitoring_active = True


def monitor_memory():
    global max_memory_mb
    while monitoring_active:
        try:
            # Find ollama_llama_server process
            for proc in psutil.process_iter(["pid", "name", "memory_info"]):
                if "ollama_llama_server" in proc.info["name"] or "ollama" in proc.info["name"]:
                    try:
                        mem = proc.info["memory_info"].rss / (1024 * 1024)  # Convert to MB
                        if mem > max_memory_mb:
                            max_memory_mb = mem
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception:
            pass
        time.sleep(0.1)


def get_base64_image(path):
    if not os.path.exists(path):
        # Create a dummy image if it doesn't exist
        print(f"Warning: {path} not found. Creating a dummy image.")
        try:
            from PIL import Image

            img = Image.new("RGB", (640, 480), color=(73, 109, 137))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path)
            print(f"Created dummy image at {path}")
        except ImportError:
            print("PIL not installed. Cannot create dummy image.")
            return None
        except Exception as e:
            print(f"Error creating dummy image: {e}")
            return None

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_unique_b64(path, seed):
    """Return a base64 JPEG that differs per ``seed``.

    Ollama caches the KV prefix for an identical image, so reusing the same
    frame reports an unrealistically low ``prompt_eval_duration`` (~24 ms).
    Real robot queries always carry a new frame, so perturb one pixel to force
    a fresh vision encode and get an honest number. Falls back to the plain
    image if PIL is unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        return get_base64_image(path)

    img = Image.open(path).convert("RGB")
    px = img.load()
    px[seed % img.width, (seed * 7) % img.height] = (
        seed % 255,
        (seed * 3) % 255,
        (seed * 5) % 255,
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(img_b64):
    payload = {
        "model": MODEL_NAME,
        "prompt": "Describe this image in detail.",
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": 0, "num_ctx": 512},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Inference error: {e}")
        if "response" in locals():
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
        return None


def main():
    global monitoring_active

    parser = argparse.ArgumentParser(description="Benchmark Ollama Moondream")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs")
    parser.add_argument("--image", default=IMAGE_PATH, help="Test image path")
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Reuse the same frame each run (reproduces Ollama's KV-prefix cache "
        "hit; not representative of real robot use).",
    )
    args = parser.parse_args()

    print("==================================================")
    print(f" OLLAMA BENCHMARK: {MODEL_NAME}")
    print("==================================================")
    if not args.cached:
        print("[-] Using a unique frame per run (defeats Ollama image-prefix cache)")

    # Start memory monitoring in background
    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.daemon = True
    monitor_thread.start()

    img_b64 = get_base64_image(args.image)
    if not img_b64:
        monitoring_active = False
        return

    def frame_for(run):
        return img_b64 if args.cached else make_unique_b64(args.image, run + 1)

    # Warmup
    print("[-] Warming up...")
    if not run_inference(img_b64):
        print("Error: Is 'ollama serve' running?")
        monitoring_active = False
        return

    print(f"[-] Benchmarking ({args.runs} runs)...")

    vision_lats = []
    throughputs = []
    total_times = []

    for i in range(args.runs):
        res = run_inference(frame_for(i))

        if not res:
            print(f"    Run {i+1}: Failed")
            continue

        # Metrics
        vis_ms = res.get("prompt_eval_duration", 0) / 1e6
        gen_ms = res.get("eval_duration", 0) / 1e6
        total_s = res.get("total_duration", 0) / 1e9
        tokens = res.get("eval_count", 0)
        tps = tokens / (gen_ms / 1000) if gen_ms > 0 else 0

        vision_lats.append(vis_ms)
        throughputs.append(tps)
        total_times.append(total_s)

        print(f"    Run {i+1}: Vision={vis_ms:.0f}ms | Speed={tps:.1f} t/s | Total={total_s:.2f}s")

    # Stop monitoring
    monitoring_active = False
    monitor_thread.join(timeout=1.0)

    # Comparison Output
    avg_vis = np.mean(vision_lats) if vision_lats else 0
    avg_tps = np.mean(throughputs) if throughputs else 0
    avg_tot = np.mean(total_times) if total_times else 0

    print("\n" + "=" * 65)
    print(f"{'METRIC':<20} | {'MOONDREAM':<15} | {'FASTVLM (TRT)'}")
    print("-" * 65)
    print(f"{'Vision Latency':<20} | {avg_vis:<8.0f} ms      | ~212 ms")
    print(f"{'Generation Speed':<20} | {avg_tps:<8.1f} tok/s   | ~11 tok/s")
    print(f"{'Total Time':<20} | {avg_tot:<8.2f} s       | ~11.8 s")
    print(f"{'Peak Memory (RSS)':<20} | {max_memory_mb:<8.1f} MB      | N/A")
    print("=" * 65)
    if "res" in locals() and res:
        print(f"\nResponse: {res.get('response', '').strip()}")


if __name__ == "__main__":
    main()
