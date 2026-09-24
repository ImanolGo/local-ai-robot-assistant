#!/usr/bin/env python3
"""
Benchmark the in-process llama.cpp Moondream bridge (Phase 2 gate).

Mirrors ``scripts/testing/llm/test_ollama_moondream.py`` so the two runtimes can
be compared on identical inputs. Instead of an HTTP round trip, it drives
:class:`LlamaCppBridge` directly and reports wall-clock latency, tokens/sec and
the host-process peak RSS.

Usage:
    python scripts/testing/llm/test_llamacpp_moondream.py
    python scripts/testing/llm/test_llamacpp_moondream.py --model /path/model.gguf \
        --mmproj /path/mmproj.gguf --runs 10
    python scripts/testing/llm/test_llamacpp_moondream.py --no-flash-attn   # A/B

Note: compare against ``test_ollama_moondream.py`` (unique frames by default).
Ollama's KV-prefix cache makes its ``prompt_eval`` look ~24 ms on repeated
frames, which is not representative of real robot use.
"""

import argparse
import base64
import os
import threading
import time

import numpy as np
import psutil

# Reuse the same image fallback behaviour as the Ollama benchmark.
IMAGE_PATH = "assets/test_image.png"
NUM_RUNS = 10
MAX_TOKENS = 128


class _PeakRssMonitor:
    """Tracks the peak RSS of the current process (llama.cpp runs in-proc)."""

    def __init__(self):
        self.peak_mb = 0.0
        self._active = True
        self._proc = psutil.Process(os.getpid())

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._active:
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, rss)
            except psutil.Error:
                pass
            time.sleep(0.1)

    def stop(self):
        self._active = False
        self._thread.join(timeout=1.0)


def get_base64_image(path):
    """Return a base64 JPEG string, creating a dummy image if missing."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Creating a dummy image.")
        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp Moondream bridge")
    parser.add_argument("--model", default="", help="GGUF model path (default: auto-discover)")
    parser.add_argument("--mmproj", default="", help="Multimodal projector GGUF path")
    parser.add_argument("--image", default=IMAGE_PATH, help="Test image path")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of benchmark runs")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size")
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="GPU layers to offload (-1 = all)"
    )
    parser.add_argument(
        "--flash-attn",
        dest="flash_attn",
        action="store_true",
        default=True,
        help="Enable flash attention (default; major vision e2e win on Orin)",
    )
    parser.add_argument(
        "--no-flash-attn",
        dest="flash_attn",
        action="store_false",
        help="Disable flash attention (for A/B comparison)",
    )
    parser.add_argument("--prompt", default="Describe this image in detail.", help="Prompt")
    args = parser.parse_args()

    # Import lazily so --help works without llama-cpp-python installed.
    from cognitive_core_nodes.llama_cpp_bridge import LlamaCppBridge

    print("=" * 66)
    print(" LLAMA.CPP BENCHMARK: moondream (in-process)")
    print("=" * 66)

    print("[-] Loading model...")
    load_start = time.time()
    bridge = LlamaCppBridge(
        model_path=args.model,
        mmproj_path=args.mmproj,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        flash_attn=args.flash_attn,
    )
    load_time = time.time() - load_start
    if not bridge.is_available():
        print(f"Error: llama.cpp bridge not available: {bridge._last_error}")
        return 1
    print(f"    Loaded in {load_time:.1f}s: {bridge.model_path}")

    img_b64 = get_base64_image(args.image)

    monitor = _PeakRssMonitor()
    monitor.start()

    print("[-] Warming up...")
    warm = bridge.generate(args.prompt, image_base64=img_b64, num_predict=MAX_TOKENS, temperature=0)
    if "error" in warm:
        monitor.stop()
        print(f"Error during warmup: {warm['error']}")
        return 1

    print(f"[-] Benchmarking vision ({args.runs} runs)...")
    vision_lats, throughputs, total_times = [], [], []
    for i in range(args.runs):
        res = bridge.generate(
            args.prompt, image_base64=img_b64, num_predict=MAX_TOKENS, temperature=0
        )
        if "error" in res:
            print(f"    Run {i + 1}: Failed ({res['error']})")
            continue

        total_s = res["total_duration"] / 1e9
        tokens = res.get("eval_count", 0)
        tps = tokens / total_s if total_s > 0 else 0
        vision_lats.append(total_s * 1000)
        throughputs.append(tps)
        total_times.append(total_s)
        print(
            f"    Run {i + 1}: Total={total_s:.2f}s | "
            f"Speed(incl. vision encode)={tps:.1f} t/s | Tokens={tokens}"
        )

    # Pure text-only generation speed isolates the LLM from the vision encoder.
    print("[-] Measuring text-only generation speed...")
    text_res = bridge.generate(
        "Write a long paragraph about robots.", num_predict=MAX_TOKENS, temperature=0
    )
    text_tps = 0.0
    if "error" not in text_res:
        text_s = text_res["total_duration"] / 1e9
        text_tps = text_res.get("eval_count", 0) / text_s if text_s > 0 else 0

    monitor.stop()

    avg_vis = np.mean(vision_lats) if vision_lats else 0
    avg_tps = np.mean(throughputs) if throughputs else 0
    avg_tot = np.mean(total_times) if total_times else 0

    print("\n" + "=" * 66)
    print(f"{'METRIC':<26} | {'LLAMA.CPP':<18}")
    print("-" * 66)
    print(f"{'Vision latency (e2e)':<26} | {avg_vis:<10.0f} ms")
    print(f"{'Vision tok/s (incl. encode)':<26} | {avg_tps:<10.1f} tok/s")
    print(f"{'Vision total time':<26} | {avg_tot:<10.2f} s")
    print(f"{'Text-only generation':<26} | {text_tps:<10.1f} tok/s")
    print(f"{'Peak RSS (host proc)':<26} | {monitor.peak_mb:<10.1f} MB")
    print("=" * 66)
    last = warm.get("response", "")
    if last:
        print(f"\nResponse: {last.strip()[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
