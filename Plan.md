# Migration Record — v3.1 → v4.0 (COMPLETED)

**Executed**: 24 Sep 2026 · **Branch merged**: `plan-v4-migration` → `main` (PR #1)
**Device**: NVIDIA Jetson Orin Nano 8GB Super · JetPack 6.2 / L4T 36.4.7

This document records the outcome of the v3.1 → v4.0 migration & simplification
effort. It is a **completed record**, not an active plan — a fresh plan will be
created separately for the next phase of work.

## Outcome summary

| Phase | Status | Result |
|-------|--------|--------|
| **0 — Baseline diagnosis** | ✅ Done | Moondream is **fully GPU-resident** under Ollama (`size_vram == size` = 1.33 GB, GR3D peaks 99%). The ~3 GB figure is `ollama` process RSS overhead, not a broken offload. → `docs/phase0_baseline.md` |
| **1 — JetPack / power profile** | ✅ Done | JetPack already 6.2. Set `MAXN_SUPER` + `jetson_clocks`. YOLO 41.9→**98.2 FPS**, Depth 28.1→**55.1 FPS**, Moondream 30.4→**49.0 tok/s**; max tj 53.5 °C. → `docs/model_performance.md` |
| **Fork — JetPack 7.2.1** | ❌ Not taken | Phase 0 disproved the precondition (offload works). Re-flash would invalidate hardware validation for no demonstrated need. |
| **2 — Ollama → in-process llama.cpp** | 🟡 Implemented, **not promoted** *(later promoted — see Addendum)* | `llama-cpp-python 0.3.35` + CUDA built; `LlamaCppBridge`, `cognitive_backend:=ollama|llamacpp` flag (default `ollama`), GBNF/JSON output, cross-backend tests. Peak RSS 2856 ≤ 2966 MB and all layers on CUDA0, but **vision e2e regressed** (3.79 s vs 2.16 s) due to slower clip encoding → default stays Ollama, old path retained. |
| **3 — Model upgrade (Qwen2.5-VL-3B)** | ⏭️ Not run | Optional; current Moondream meets targets and a 3B model would erode the 8 GB budget. |
| **4 — Audio stack consolidation** | ⏭️ Not triggered | Gated on RAM pressure; none present. |
| **5 — Finish the project** | ✅ 4/5 | Visual verification loop (`visual_verification_node.py`) + minimal FastAPI `/health` + `/status`; SLAM/BT scope decisions recorded. Full-system 60-min soak deferred. |

## Key decisions

- **No SLAM for the MVP.** Reactive "go to visible object" using Tier 1 YOLO +
  depth. No map, no RTAB-Map dependency.
- **No BehaviorTree.CPP.** Keep the `command_router_node` regex + cognitive
  forward pattern; add a dedicated `visual_verification_node`.
- **In-process llama.cpp is now the default cognitive backend** (promoted
  24 Sep 2026 after enabling flash attention; Ollama retained as an automatic
  fallback and for model swapping).
- **JetPack 7.2.1 fork declined.**

## Verification

- Unit/integration suite: **68 passed, 2 failed** (the 2 are pre-existing
  `tests/test_wake_word.py` issues, unrelated to the migration).
- `conftest.py` fixed so the source `robot_interfaces` package no longer hides
  the generated `robot_interfaces.msg`/`srv`.

## Addendum — 24 Sep 2026: llama.cpp promoted to default

The Phase 2 decision above was revisited after two findings:

1. The original comparison reused one image per run; Ollama's KV-prefix cache
   made its prompt-eval look like ~24 ms (total 1.32 s). With a unique frame per
   run, Ollama is ~2.61 s.
2. The clip encoder was already GPU-offloaded via `mtmd use_gpu=True` in
   llama-cpp-python 0.3.35; the real missing optimization was **flash attention**
   on the LLM context (2.34 s → 1.92 s).

Honest result: llama.cpp **1.92 s** vs Ollama **2.61 s** vision end-to-end, with
lower RSS (2666 MB vs 3022 MB). `cognitive_backend` now defaults to `llamacpp`
(`llm_flash_attn:=true`), with automatic fallback to Ollama. See
`docs/model_performance.md`.

## References

- `docs/phase0_baseline.md` — Phase 0 measurements
- `docs/model_performance.md` — 15 W vs MAXN SUPER + llama.cpp comparison
- `docs/architecture.md` — v4.0 architecture (SLAM removed)
- `STATUS.md` — project source of truth
