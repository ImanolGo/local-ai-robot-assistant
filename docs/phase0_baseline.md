# Phase 0 — Baseline Diagnosis (Moondream / Ollama GPU residency)

**Date**: 2026-09-24
**Device**: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (Orin Nano 8GB)
**L4T / JetPack**: R36.4.7 (JetPack 6.2)
**Tooling**: `jetson-device-skills` (NVIDIA-AI-IOT) — `jetson-diagnostic`, `jetson-memory-audit`

## Goal

Determine whether the Moondream/Ollama memory overshoot (Known Issue #1 in
`STATUS.md`: ~3 GB actual vs 1.8 GB budgeted) is a GPU-offload failure or
ordinary runtime overhead.

## Method

1. Ran `jetson-diagnostic/scripts/snapshot.sh` and
   `jetson-memory-audit/scripts/audit.sh` for a full device health snapshot.
2. Fired a real Moondream vision request (`assets/images/bus.jpg`, 128 max
   tokens, `num_ctx=512`) while sampling `ollama ps` (`/api/ps`), the `ollama`
   process RSS, and `tegrastats` `GR3D_FREQ` in the background.

## Measurements

### Device / power
| Field | Value |
|-------|-------|
| SKU | `orin-nano-8gb` |
| Product model | nvidia jetson orin nano engineering reference developer kit super |
| L4T version | 36.4.7 |
| Default systemd target | `multi-user.target` (headless, no GUI) |
| `nvpmodel` | id 0, **15W** |
| CPU clocks (idle sample) | 729 MHz |
| Idle RAM | 1242 / 7620 MB (`tegrastats`); 6.4 GB `available` (`/proc/meminfo`) |
| Swap | 16 GB configured, 0 used |
| Thermal (idle) | CPU 46.4 °C, GPU 47.5 °C, tj 48.0 °C |

### Moondream under active inference
| Metric | Value |
|--------|-------|
| Model | `moondream:latest`, family phi2+clip, 1B, Q4_0 |
| `/api/ps` `size_vram` | **1,325,721,600 bytes (1.33 GB) — 100 % of model in VRAM** |
| `/api/ps` `size` | 1,325,721,600 bytes (same as `size_vram` → no CPU spill) |
| `GR3D_FREQ` | max **99 %**, avg 50.7 % over 44 samples |
| `ollama` process peak RSS | **3020.7 MB (~3.0 GB)** |
| Generation | 73 tokens / 2.42 s ≈ **30.2 tok/s** |
| Cold request wall time | 8.59 s (includes first-load of weights) |

> `nvmap` per-process GPU attribution was not readable without root
> (`gpu_source: "none"`). The `/api/ps` `size_vram` field is the authoritative
> signal here: Ollama reports the full model resident in VRAM.

## Conclusion (Phase 0 gate)

**Is Moondream fully GPU-resident under the current Ollama install? → YES.**

- `ollama ps` reports `size_vram == size` (1.33 GB) → 100 % GPU, no CPU offload.
- `GR3D_FREQ` spikes to 99 % during inference → the GPU is actually doing the work.
- The ~3 GB figure is the **`ollama` process RSS**, i.e. the 1.33 GB of model
  weights + runtime/KV-cache/allocator overhead — not a broken offload.

## Outcome branch selected

This is the plan's **first branch**: the overshoot is Ollama baseline overhead +
KV-cache, not a broken install. Per `Plan.md`, Phase 2 is therefore *lower
priority* — Phase 1 can be the stopping point if RAM headroom is otherwise
fine. We proceed with Phase 1 regardless, and will re-evaluate Phase 2 against
the measured headroom (idle available ≈ 6.4 GB; ~3 GB peak while Moondream is
resident).

## Notes / follow-ups

- Root (`sudo`) is required for `nvmap` GPU attribution and for the power-mode
  changes in Phase 1; this was not available to the agent session.
- `docs/model_performance.md` and `STATUS.md` reference the Ollama benchmark at
  the stale path `scripts/test_ollama_moondream.py`; the real path is
  `scripts/testing/llm/test_ollama_moondream.py`.
