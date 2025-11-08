# AI Models Directory

This directory contains all AI models used by the robot assistant. Models are downloaded automatically using the provided scripts.

## Directory Structure

- `wake_word/` - openWakeWord models for wake word detection
- `whisper_tiny_trt/` - Whisper Tiny model (TensorRT optimized or faster-whisper)
- `piper_voice/` - Piper TTS voice models and configuration
- `yolo_trt/` - YOLOv11n object detection model (TensorRT optimized)
- `depth_trt/` - FastDepth depth estimation model (TensorRT optimized)
- `nanollm_quantized/` - Quantized LLM for cognitive processing

## Model Download

To download all required models:

```bash
python scripts/setup/download_models.py
```

To download specific models:

```bash
python scripts/setup/download_models.py --models yolo whisper
```

To verify model integrity:

```bash
python scripts/setup/download_models.py --verify
```

## Storage Requirements

- Total size: ~8-12 GB
- Ensure sufficient storage on Jetson before downloading
- Models are automatically validated using checksums

## Performance Targets

- YOLO inference: <50ms on Jetson Orin Nano
- Depth estimation: <70ms on Jetson Orin Nano
- Whisper transcription: Real-time factor <0.3x
- TTS synthesis: <500ms for 20 words
- LLM inference: <3s for typical response

See `docs/model_credits.md` for licensing and attribution information.
