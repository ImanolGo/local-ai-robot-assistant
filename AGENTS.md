# AGENTS.md — Local AI Robot Assistant

Canonical guidance for AI coding agents working in this repository. Read this
first, then `STATUS.md` (source of truth) and `docs/architecture.md` (v4.0).

## 1. Project Overview

A fully local, real-time, multimodal AI robot assistant running on an NVIDIA
Jetson Orin Nano (8 GB). ROS2 Humble middleware; Python 3.10 primary. No cloud
APIs — all inference is on-device.

- **Cognitive core**: Moondream 1.6B VLM run **in-process via `llama.cpp`**
  (default; GPU `mtmd` vision + flash attention). The Ollama HTTP server remains
  available as `cognitive_backend:=ollama` and as an automatic fallback.
- **Perception**: YOLOv11n + Depth Anything V2 Small, TensorRT FP16.
- **Audio**: openWakeWord → Silero VAD → faster-whisper (`tiny.en`), Piper TTS.
- **Behavior**: reactive — `command_router_node` (regex + cognitive forward) plus
  a `visual_verification_node` goal-completion loop.
- **Actuation**: Wave Rover differential drive over UART (`/dev/ttyTHS1`).

## 2. Current State & Scope (do not regress)

The v3.1 → v4.0 migration is complete (`Plan.md` is the record). Two things are
**out of scope for the MVP and must not be reintroduced**:

- **SLAM / localization** — no RTAB-Map, no `robot_localization`/EKF, no
  `localization_nodes` package. The reactive "go to visible object" path needs no
  map. (The package and SLAM scripts were deleted in the `cleanup/remove-slam`
  branch.)
- **BehaviorTree.CPP** — no `behaviortree_cpp_v3`, no `py-trees`, no behavior-tree
  executor/blackboard. Use `command_router_node` + `visual_verification_node`.

**IMU** is published on `/imu/data` by `uart_motor_controller` itself (it owns the
serial port and parses continuous feedback / `T=126` responses). There is no
standalone IMU node.

## 3. Environment

- **Hardware**: NVIDIA Jetson Orin Nano 8 GB, JetPack 6.2 (L4T 36.4.7).
- **Power mode**: `MAXN_SUPER` with `jetson_clocks`. **Do not change the power
  mode.**
- **ROS**: ROS2 Humble, installed at `/opt/ros/humble`.
- **Python**: a venv at `.venv` (used for model libraries). Source helpers:
  - `source ros2_venv.sh` — activates `.venv` **and** sources `install/setup.bash`,
    then adds venv site-packages to `PYTHONPATH`.
  - `./launch_node.sh <package> <node>` — runs a single ROS2 Python node with the
    venv interpreter (bypasses the system-Python shebang).
- **Models** live under `models/` (git-ignored, downloaded separately). Ollama
  runs as a systemd service on `localhost:11434` with `moondream` pulled.

## 4. Commands

### Build (canonical)
```bash
source /opt/ros/humble/setup.bash
colcon --log-base log build --install-base install --build-base build --symlink-install
```
- The canonical install tree is the **top-level `install/`**.
- `.colcon/defaults.yaml` sets `install-base: src/install` / `build-base: src/build`,
  which does **not** match how this project is actually built. Always pass the
  explicit `--install-base install --build-base build` flags (or a full clean
  rebuild) so tests/helpers that source `install/setup.bash` keep working.
- `--symlink-install` means Python source edits are live; **but** setup.py entry
  points, `package.xml`, and launch data files require a rebuild.
- Stale console scripts survive rebuilds in `install/<pkg>/lib/<pkg>/`. If you
  remove an entry point, delete its generated script and rebuild.

### Test (canonical — run before and after changes)
```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python -m pytest tests/ integration_tests/test_visual_verification_loop.py -q -o pythonpath=""
```
- Expected baseline: **68 passed, 2 failed**.
- The 2 failures are pre-existing and unrelated: `tests/test_wake_word.py`
  (`AudioEvent` has no `.details`; mock lacks `use_sim_time`). Do not let them mask
  regressions — the passing count must never drop.
- `-o pythonpath=""` is intentional: `pytest.ini` sets `pythonpath = src`, and
  `conftest.py` manages the source/installed `robot_interfaces` merge. Don't
  remove the conftest shim: the source `robot_interfaces` package would otherwise
  shadow the ROS2-generated `robot_interfaces.msg`/`srv`.

### Run
```bash
ros2 launch launch/full_system_launch.py            # full system
./launch_node.sh <package> <node>                    # one node
ros2 launch behavioral_nodes behavioral_launch.py    # command router + verification
```

### Format / lint (pre-commit runs on commit)
```bash
black --line-length=100 <files>
isort --profile black --line-length=100 <files>
flake8 --max-line-length=100 --ignore=E203,W503 <files>
```
Pre-commit hooks: trailing-whitespace, end-of-file-fixer, check-yaml/json,
check-merge-conflict, check-added-large-files (>1 MB), black, flake8, isort, mypy
(`--ignore-missing-imports`). Committing triggers them.

## 5. Packages, Nodes & Topics

Packages live in `src/` (7 total: `actuation_nodes`, `audio_interface_nodes`,
`behavioral_nodes`, `cognitive_core_nodes`, `perception_nodes`,
`robot_interfaces`, `web_interface_nodes`).

| Package | Entry points | Key topics |
|---|---|---|
| `actuation_nodes` | `uart_motor_controller` | sub `/cmd_vel`, `/motor_command`; pub `/motor_status`, `/chassis_state`, `/odom_raw`, `/imu/data`; srv `/emergency_stop` |
| `perception_nodes` | `camera_driver`, `image_undistort_node`, `object_detector`, `depth_estimator`/`depth_estimation_node`, `pointcloud_generator` | `/camera/raw`, `/camera/undistorted`, `/perception/objects`, `/perception/depth`, `/perception/obstacles`, `/perception/pointcloud` |
| `audio_interface_nodes` | `audio_capture_node` (self-contained wake→VAD→Whisper), `audio_playback_node` (Piper TTS + notifications) | `/audio/events`, `/audio/transcription`, `/audio/tts_request` |
| `cognitive_core_nodes` | `cognitive_client_node` (in-process llama.cpp default; Ollama fallback) | sub `/cognitive/multimodal_query`; pub `/cognitive/command`, `/cognitive/status` |
| `behavioral_nodes` | `command_router_node`, `visual_verification_node` | `/cognitive/command`, `/cmd_vel`, `/verification/status` |
| `web_interface_nodes` | `web_server` (minimal FastAPI `/health`, `/status`) | HTTP |
| `robot_interfaces` | (messages/services only) | `msg/`, `srv/` |

Backend selection: `cognitive_backend:=llamacpp|ollama` (default `llamacpp`).
`llamacpp` is promoted (flash attention + GPU `mtmd` clip; faster honest vision
e2e and lower RSS). `ollama` is the fallback; the node auto-switches if the
in-process model cannot load. When benchmarking, use a unique frame per run —
Ollama caches the image KV prefix and reports a misleadingly low prompt-eval.

## 6. Coding Conventions

- Python: PEP 8, type hints, Google-style docstrings, **max line length 100**.
- ROS2: single-responsibility nodes; declare parameters; graceful `destroy_node`;
  use the ROS logger; no blocking I/O in callbacks.
- Hardware: validate connections, retry UART, handle model-load failures, never
  swallow errors silently.
- Do **not** add comments unless they clarify non-obvious intent (the codebase is
  sparsely commented).
- Do not commit secrets or model/engine binaries (`.gitignore` covers them).

## 7. Testing Conventions

- Unit tests in each package's `test/`; cross-cutting tests in `tests/`;
  hardware/multi-node tests in `integration_tests/` and `hardware_tests/`.
- Mock hardware (serial, camera, audio, models) in unit tests.
- `pytest.ini` marks: `unit`, `integration`, `hardware`, `slow`, `uart`.
- Hardware tests must skip (not fail) when the device/port is absent.

## 8. Gotchas / Lessons Learned

- **Serial port**: only `uart_motor_controller` may open `/dev/ttyTHS1`. It
  disables RTS/DTR and enables continuous feedback (`{"T":131,"cmd":1}`).
- **IMU**: do not add a second serial reader; extend the motor controller.
- **Depth topic**: the undistorted image topic is `/camera/undistorted` (not
  `/camera/image_undistorted`).
- **Install layout**: a clean rebuild installs generated `robot_interfaces`
  Python modules under `install/robot_interfaces/lib/python3.10/site-packages`.
  `conftest.py` probes multiple layouts — keep it layout-agnostic.
- **Entry-point ghosts**: old generated scripts (e.g. `behavior_tree_executor`,
  `dialogue_manager`) linger in `install/` after source changes; delete them.
- **`src/build`, `src/install`, `src/log`** are stale artifacts from
  `.colcon/defaults.yaml`; the active trees are top-level `build/`, `install/`,
  `log/`. Don't source or rely on the nested ones.
- **Memory**: 8 GB shared. Moondream `ollama` RSS peaks ~3 GB. Don't load all
  models at once; lazy-load where possible.

## 9. Source-of-Truth Docs

- `STATUS.md` — implementation status, known issues, next milestones.
- `docs/architecture.md` — v4.0 architecture (SLAM removed).
- `Plan.md` — completed v3.1 → v4.0 migration record.
- `.github/copilot-instructions.md` — additional Copilot-oriented guidance.

## 10. Next Milestones (from STATUS.md)

1. **Full-system 60-min integration soak**: object detection + depth + audio +
   cognitive core + visual verification concurrently, logging `tegrastats` RSS and
   thermals.
2. **Audio real-time validation**: end-to-end wake-word → transcription latency
   and resource usage on hardware.
3. **Optional**: promote the `llamacpp` backend after GPU-accelerating the clip
   encoder (currently slower than Ollama).
4. **Web interface**: expand beyond `/health` + `/status` once the system is
   stable.

## 11. Ask for Human Review When

- Changing hardware communication protocols or the serial/IMU path.
- Changing command-routing or verification-loop logic.
- Implementing safety-critical features (e.g. emergency stop).
- Making significant architectural changes or adding a new AI model integration.
- Reintroducing anything from §2 (SLAM, BehaviorTree) — confirm first.
