# Migration & Simplification Plan — v3.1 → v4.0

### For execution by a coding agent (opencode) against `local-ai-robot-assistant`

## How to use this document

Feed this file to the agent as its working spec (e.g. `opencode run --file PLAN.md` or paste into the agent's context). Each phase is self-contained: a goal, why it matters, exact tasks, files touched, a benchmark/test gate, and a rollback. **The agent should not start a phase until the previous phase's gate has passed**, and should commit each phase on its own branch/PR so a regression can be reverted without losing later work.

Ground rules for the agent, every phase:

1. Run the existing test suite (`colcon test`) before touching anything, and after. Never merge a phase that drops the passing-test count.
2. Update `STATUS.md` at the end of every phase (progress %, Recent Updates entry, any new Known Issues). This file is the project's source of truth — keep it honest, including when a benchmark *doesn't* hit the target.
3. Never delete a working subsystem before its replacement passes its own gate. Land the new thing behind a flag/launch-arg, prove it, then remove the old one in a separate commit.
4. Where this plan cites a number from external research (JetPack Super Mode clock speeds, Moonshine WER, etc.), treat it as a *hypothesis to verify on this hardware*, not a fact to assume. The gate in each phase exists for this reason — run it and record the real number.

---

## Phase 0 — Baseline diagnosis (do this before anything else) ✅ DONE — 24 Sep 2026

> **Outcome**: Moondream is **fully GPU-resident** (`ollama ps` reports `size_vram == size` = 1.33 GB; `GR3D_FREQ` peaks 99%). The ~3 GB is `ollama` process RSS overhead, **not** a broken offload. Recorded in `docs/phase0_baseline.md`. Gate answered **YES**; plan's first outcome branch selected (Phase 2 is lower priority — can stop after Phase 1 if RAM headroom is fine).

**Goal**: Find out whether the Moondream/Ollama memory overshoot (3GB actual vs 1.8GB budgeted, per STATUS.md Known Issue #1) is a GPU-offload failure or something else. This determines whether Phase 2 is a bug fix or a redesign.

**Tasks**:

1. ~~Install `jetson-device-skills`~~ ✅ Installed (`/tmp/opencode/jetson-device-skills`, linked into `~/.claude/skills`, `~/.codex/skills`, `~/.agents/skills`, `~/.cursor/skills`). Used `jetson-diagnostic` + `jetson-memory-audit` scripts.

   ```
   git clone https://github.com/NVIDIA-AI-IOT/jetson-device-skills.git
   cd jetson-device-skills && ./install.sh
   ```

   Use its `jetson-diagnostic` skill for a full device health snapshot (memory, GPU, thermal, power, services) and `jetson-memory-audit` specifically to check DRAM/NvMap usage and whether the Moondream model's memory is actually being reclaimed correctly vs fragmenting. This replaces hand-rolling a diagnostic script — it's maintained, Jetson-aware, and already knows the right `tegrastats`/`/proc/meminfo` fields to check.
2. ~~Separately, still confirm GPU residency directly: run `ollama ps` during an active Moondream request to see whether it reports `100% GPU` or a split `CPU/GPU` allocation.~~ ✅ Done — `/api/ps` reports `size_vram == size` (100% GPU). This is the one thing `jetson-memory-audit` won't tell you on its own — it audits system memory pressure, not which runtime is holding which tensors.
3. ~~Capture and record: peak RSS of the `ollama` process, GPU utilization during inference (from `jetson-diagnostic`'s output), and current JetPack version (`cat /etc/nv_tegra_release`).~~ ✅ Done — peak `ollama` RSS 3020.7 MB; `GR3D_FREQ` max 99%; JetPack R36.4.7.

**Gate**: ✅ **YES** — Moondream is fully GPU-resident (see `docs/phase0_baseline.md`).

**Outcome branches**:

- If GPU utilization spikes to near-100% during inference and `ollama ps` shows 100% GPU: the memory overshoot is likely just Ollama's baseline overhead + KV-cache, not a broken install. Phase 2 becomes lower priority — you can stop after Phase 1 if RAM headroom is otherwise fine.
- If GPU utilization stays low/flat or `ollama ps` shows partial CPU offload: confirms the suspected GPU-offload issue. Proceed with Phase 2 as a priority fix, not a nice-to-have.

---

## Phase 1 — JetPack / power-profile upgrade (low-risk, do regardless of Phase 0 outcome)

**Goal**: Get "Super Mode" clocks if not already on JetPack 6.1/6.2, for free throughput on everything downstream, without a full OS/kernel/ROS-distro change.

**Tasks**:

1. Check current JetPack version (`cat /etc/nv_tegra_release`). If \< 6.1, flash/upgrade via SDK Manager to JetPack 6.2. **This is destructive to the OS image — back up `/home` and any model weights first.**
2. Set the power mode: `sudo nvpmodel -m 2` (MAXN SUPER) and `sudo jetson_clocks`.
3. Re-run the existing benchmarking scripts you already have:
   - `scripts/test_yolo.py`
   - `scripts/test_depth.py`
   - `scripts/test_ollama_moondream.py`
4. Record before/after FPS and tok/s in `docs/model_performance.md` under a new "JetPack 6.2 / MAXN SUPER" section. Do not overwrite the old numbers — append, so regressions are visible.

**Gate**: YOLO/Depth FPS did not regress, and VLM tok/s improved (any improvement counts — don't block on hitting a specific multiplier from the research doc).

**Rollback**: `sudo nvpmodel -m 1` reverts to the 15W profile if thermals or stability regress under sustained load (re-check Phase "Thermal Stability" target from architecture.md §13).

---

## Fork in the road — JetPack 7.2.1, decide explicitly, don't default into it

JetPack 7.2.1 is real for the Orin Nano (Ubuntu 24.04, CUDA 13.2, TensorRT 10.16, ROS2 Jazzy, ISO-only flash, ships Super Mode by default) and it brings two genuinely useful, verified pieces: `PyNvVideoCodec 2.2` (DLPack/CUDA-buffer zero-copy video frames — a real, confirmed alternative to the base64/HTTP path this plan already removes in Phase 2 for VLM frames), and the `jetson-device-skills` tooling used in Phase 0 above.

**Treat the specific memory/speed numbers in the source research with caution before using them to justify this**: the "\~40% lower memory, 41.8% faster prompt processing, 27.9% faster token generation" figures are a real published benchmark, but measured on a **32GB AGX Orin running a 27B-parameter model**, not an 8GB Orin Nano running a 2–3B VLM. Most of that win is fixed OS/driver overhead shrinking as a fraction of a *huge* model's footprint — expect a real but much smaller effect on Moondream-class models. Re-measure on this hardware before trusting a number. Also drop "NeMoClaw" from consideration for this project: it's real, but it's a sandboxing/ security layer for general always-on chat agents (OpenClaw), not a vision/speech/behavior orchestration tool — it doesn't solve anything this robot needs.

**Why this is a fork, not a phase**: unlike Phases 1–4 above, this is a full re-flash — new kernel, new Ubuntu, new ROS2 distro (Humble → Jazzy). It invalidates the *software* side of Phase 0/1 hardware validation you already completed and tested (UART protocol code, camera DeepStream pipeline, audio device configs) — the physical wiring doesn't change, but every driver interaction needs re-validation, and `faster-whisper`/CTranslate2 and other pinned aarch64 wheels have far less community mileage on Python 3.12 + CUDA 13.2 than on the JetPack 6 path you're already on. At 78% project completion with 100+ passing tests built against the current stack, this is a real regression-risk decision, not a checkbox.

**Decision gate — only take this fork if**:

1. Phase 0's diagnosis shows the Ollama GPU-offload problem is a JetPack-6-level driver limitation that Phase 2's llama.cpp swap doesn't fix (i.e., you're still memory-constrained after Phase 2, *and* Phase 4's audio consolidation also isn't enough headroom), **and**
2. You're willing to re-run the full Phase 1 (hardware validation) test suite from STATUS.md after the flash, not just the model benchmarks — treat it as re-earning that 100% Phase 1 completion, not assuming it carries over, **and**
3. You confirm ahead of time that `faster-whisper`/CTranslate2 (or whatever ASR you land on from Phase 4) has a working aarch64 + CUDA 13.2 + Python 3.12 build path — don't discover this mid-flash.

If you do take this fork, sequence it as its own branch off the *current* JetPack 6 baseline (not layered on top of Phases 2–4), so you can A/B the same llama.cpp cognitive core on both JetPack 6.2/MAXN-SUPER and JetPack 7.2.1 and get a real, comparable number for your actual model before committing the whole project to the re-flash.

---

## Phase 2 — Replace the Ollama HTTP layer with in-process llama.cpp

**Priority**: highest-value change in this plan. Directly targets STATUS.md Known Issue #1 (memory overshoot) and Known Issue #2 (whisper memory, addressed separately in Phase 4).

**Goal**: Remove the HTTP/JSON/base64 round-trip and Ollama's daemon overhead without changing the `cognitive_client_node`'s external contract (topics in/out stay the same, so nothing downstream — command router, behavior tree — needs to change).

**Preconditions**: Phase 0 diagnosis complete and documented.

**Tasks**:

1. Add `llama-cpp-python` (with `CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87"`) to the venv. Build from source — no prebuilt aarch64+CUDA wheel exists; expect this to be the most time-consuming step in the phase.
2. Pull a GGUF build of the current model (`moondream2` GGUF, or whatever Phase 0 confirmed is working) — do **not** simultaneously switch models and switch runtimes; isolate the variable.
3. Create `cognitive_core_nodes/llama_cpp_bridge.py`, mirroring the existing `OllamaBridge` class's public interface (`analyze_scene(image_cv2, user_prompt, world_context) -> str`) so `cognitive_client_node.py` only needs a one-line swap of which bridge it imports.
   - Pass the JPEG bytes directly into the `llama-cpp-python` chat-completion call's image parameter. This alone removes the base64 string round-trip — don't chase the research doc's "zero-copy CUDA pointer" framing, that requires custom C++ tensor-sharing work with uncertain payoff; skip it for this pass.
   - Set an explicit, bounded `n_ctx` (start at the same value currently used, e.g. 512–2048) so KV-cache size is deterministic instead of whatever Ollama was allocating.
4. Add a feature flag / launch argument `cognitive_backend:=ollama|llamacpp` to `cognitive_launch.py`, defaulting to `ollama` until Phase 2's gate passes.
5. Use structured output instead of the markdown-fence-stripping parser: pass a JSON grammar (GBNF) or `response_format={"type": "json_object"}` if supported by your `llama-cpp-python` version, matching the existing `{"action", "target", "explanation"}` schema. Keep `parse_json_intent()` as a defensive fallback rather than deleting it outright — don't remove a tested safety net until the grammar path has proven itself in Phase 2's gate.
6. Update the 8 existing intent-parsing unit tests to run against both backends via parametrization, so you have a regression check either way.

**Gate**:

- `ollama ps`/`tegrastats` comparison: llama.cpp path shows equal-or-lower peak RSS than the Phase 0 baseline, with GPU utilization confirmed (not assumed) via `tegrastats`.
- Vision latency and tok/s: measure with the same benchmarking script pattern as `scripts/test_ollama_moondream.py`, just pointed at the new bridge. Do not assume the research doc's numbers — record what you actually get.
- All existing cognitive-core and command-router unit tests pass unmodified in behavior (same inputs → same parsed intents).
- 60-minute soak test: fire the same "go to X" / verification-loop request repeatedly for an hour, watch for memory growth (fragmentation) or crashes — this was Known Issue #3 in the research (Ollama process instability over long runtimes); confirm the new path doesn't have the same problem before trusting it.

**Only after the gate passes**: flip the default in `cognitive_launch.py` to `llamacpp`, then in a *separate* follow-up commit, remove `OllamaBridge`, the `ollama` systemd service, and `scripts/setup_ollama.sh`. Update `docs/guides/ollama_setup.md` → rename/replace with `docs/guides/llamacpp_setup.md`.

**Rollback**: the launch-arg flag makes this a one-line revert (`cognitive_backend:=ollama`) until the old code is actually deleted.

---

## Phase 3 — (Optional, decide via a measurement) Model upgrade to Qwen2.5-VL-3B or Moondream2

**Goal**: Evaluate whether a newer small VLM gives materially better spatial reasoning / instruction-following without breaking the RAM budget, now that Phase 2 has given you a cleaner memory baseline to measure against.

**Do not do this phase blind.** Run it only if Phase 2's soak test shows comfortable RAM headroom (say, >1GB free at peak) — otherwise stay on the model you already have working.

**Tasks**:

1. Pull Qwen2.5-VL-3B (4-bit GGUF) alongside the existing model (don't remove the working one).
2. Re-run the exact same "Find the bottle" / visual-verification test scenarios described in architecture.md §Phase 6 Tasks, side by side on both models, same prompts, same images.
3. Score subjectively (does it correctly identify + locate the target) and objectively (tok/s, peak RSS, TTFT) for each. Record both in `docs/model_performance.md`.

**Gate**: New model wins on task success rate without exceeding the RAM budget confirmed safe in Phase 2. If it's a wash or worse on RAM, keep the current model — this phase is allowed to conclude "no change," and that's a valid, useful outcome to record.

---

## Phase 4 — (Optional, gated on actual RAM pressure) Audio stack consolidation

**Do not start this phase unless Phase 2 (and optionally Phase 3) leave you RAM-constrained.** Your audio pipeline (`audio_capture_node.py`) is Phase 5 in STATUS.md: **100% complete, tested, already refactored once for this exact kind of simplification** (removed raw audio streaming, event-driven notifications, single self-contained node). Replacing a working, tested subsystem for RAM you may not need is the wrong trade — only pull this trigger if the numbers say so.

**Goal, if triggered**: Replace `arecord` + `openWakeWord` + `faster-whisper` + `Piper` with a single `sherpa-onnx` node (Silero VAD + Moonshine ASR + Kokoro-82M TTS), matching the pattern your existing node already uses (self-contained, ROS2-lightweight, text-only topics out).

**Tasks**:

1. Build `sherpa-onnx` with CUDA execution provider for `sm_87` (expect this to need a source build, similar friction to Phase 2's llama.cpp build — no guaranteed prebuilt aarch64+CUDA wheel).
2. Pull model weights: Silero VAD ONNX (you already use this — no change), a Moonshine ASR model (`sherpa-onnx-moonshine-tiny-en-int8` or `-base-en`), and Kokoro-82M INT8 ONNX.
3. Create `audio_interface_nodes/unified_audio_node.py` implementing the **same state machine** already documented in architecture.md §2.4 (`IDLE → WAKE_WORD_DETECTED → RECORDING → TRANSCRIBING → IDLE`) and publishing to the **same topics** (`/audio/events`, `/audio/transcription`) so `command_router_node.py` needs zero changes.
4. Add a launch-arg flag exactly as in Phase 2 (`audio_backend:=legacy|sherpa`), defaulting to `legacy` until the gate passes.
5. Port your existing wake-word model (`hey_roe_ver.onnx`) — openWakeWord and sherpa-onnx use different formats, so this may require re-training the wake word via the same Colab pipeline referenced in architecture.md §2.4, not just a file copy. Budget time for this.

**Gate**:

- RAM: measure actual peak RSS of the new node vs the four old processes combined. Compare against the current *measured* baseline, not the research doc's "1.2GB → 250MB" claim.
- Latency: end-to-end wake-word-to-transcription time, same test harness style as `scripts/test_audio_playback_node.py`.
- ASR accuracy: re-run whatever informal accuracy check you used for faster-whisper (motor-noise test mentioned in architecture.md §9.2) against Moonshine, side by side. Moonshine's published benchmarks show it beating same-size Whisper models on clean speech WER, but your motor-noise environment is exactly the kind of condition that isn't captured in a public leaderboard — verify it yourself before trusting it.
- All 37 existing audio-related unit tests either pass unmodified or have a documented, reviewed replacement.

**Rollback**: flag-based, same pattern as Phase 2.

---

## Phase 5 — Finish the project (independent of Phases 2–4; can run in parallel)

This continues the scope-cutting from the earlier review, unaffected by the backend swap:

1. **Skip full SLAM (Phase 6 in STATUS.md) for the MVP.** Implement a reactive "go to visible object" behavior using existing Tier 1 YOLO + depth output instead — no map, no RTAB-Map dependency. Only revisit RTAB-Map if/when persistent multi-room memory becomes an actual requirement.
2. **Keep the existing `command_router_node.py` regex+cognitive-forward pattern** instead of standing up BehaviorTree.CPP (Phase 8.2). Introduce a real BT only when a second genuinely multi-step behavior (search → navigate → verify → retry) needs it.
3. **Implement Phase 7.3 (visual verification loop)**: this was already scoped in architecture.md §12 Phase 5 Task 3 — stop, snapshot, ask "is goal X achieved," retry with rotation if unsure. This is real, needed work, independent of the backend chosen in Phase 2.
4. **Web interface (Phase 9)**: minimal FastAPI health/status endpoint only. Do not build the full dashboard until everything else is stable.
5. **Integration testing (Phase 10)**: once Phases 2 (and optionally 3/4) are merged and their individual gates passed, run the full-system 60-minute soak test described in the research doc's Phase 4 (SLAM/nav skipped per item 1 above, so: object detection + depth + audio + cognitive core + visual verification, concurrently, for 60 minutes, logging `tegrastats` RSS and thermals). This is the real end-to-end validation — everything before this point was per-subsystem.

**Gate**: total RAM stays under budget with no swap thrashing, no thermal throttling, all subsystems stable for the full hour.

---

## Suggested execution order for the agent

```
Phase 0 (diagnose)
   │
Phase 1 (JetPack/power — quick win, do anytime)
   │
Phase 2 (Ollama → llama.cpp)  ──────────────► Phase 5 items 2–3 (can start in parallel,
   │                                           they don't depend on the backend)
   ├── gate passed? ── yes ──► Phase 3 (model upgrade, optional)
   │                              │
   │                              ▼
   └── RAM still tight? ── yes ──► Phase 4 (audio consolidation, optional)
                                     │
                                     ▼
                          Phase 5 item 5 (full integration soak test — last)
```

## What NOT to do

- Don't do Phase 4 first "because the report says it's the biggest RAM win." Phase 2 is higher value, lower risk (smaller, more isolated subsystem, and it's fixing a documented known bug rather than replacing something that already works), and gives you a cleaner baseline to decide *whether* Phase 4 is even necessary.
- Don't switch runtime and model in the same commit (Phase 2 vs Phase 3) — you'll never know which change caused which effect.
- Don't delete the old subsystem (Ollama service, openWakeWord/faster-whisper/Piper) until the replacement's gate has passed on real hardware. Flag-and-prove, then remove.
