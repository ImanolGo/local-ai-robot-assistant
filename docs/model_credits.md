# Model Credits and Licensing

This document provides attribution and licensing information for all AI models used in the Local AI Robot Assistant project.

## Overview

All models are used in compliance with their respective licenses. This project is for educational and research purposes, and all commercial use must comply with individual model licenses.

## Vision Models

### YOLOv8n Object Detection

- **Source**: Ultralytics
- **Repository**: https://github.com/ultralytics/ultralytics
- **License**: GPL-3.0
- **Citation**:
  ```
  @software{Jocher_Ultralytics_YOLO_2023,
    author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
    license = {GPL-3.0},
    month = jan,
    title = {{Ultralytics YOLO}},
    url = {https://github.com/ultralytics/ultralytics},
    version = {8.0.0},
    year = {2023}
  }
  ```
- **Usage**: Object detection for robot perception
- **Modifications**: Converted to TensorRT format for inference optimization

### FastDepth Monocular Depth Estimation

- **Source**: MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
- **Repository**: https://github.com/dwofk/fast-depth
- **License**: MIT
- **Paper**: "FastDepth: Fast Monocular Depth Estimation on Embedded Systems" (ICRA 2019)
- **Citation**:
  ```
  @inproceedings{wofk2019fastdepth,
    title={FastDepth: Fast Monocular Depth Estimation on Embedded Systems},
    author={Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
    year={2019}
  }
  ```
- **Usage**: Depth estimation from monocular camera input
- **Modifications**: Converted to TensorRT format for inference optimization

## Audio Models

### Whisper Speech Recognition

- **Source**: OpenAI
- **Repository**: https://github.com/openai/whisper
- **License**: MIT
- **Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision"
- **Citation**:
  ```
  @article{radford2022whisper,
    title={Robust Speech Recognition via Large-Scale Weak Supervision},
    author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
    journal={arXiv preprint arXiv:2212.04356},
    year={2022}
  }
  ```
- **Model Used**: Whisper Tiny (39M parameters)
- **Usage**: Speech-to-text conversion for voice commands
- **Modifications**: Used with faster-whisper for optimized inference

### Piper Text-to-Speech

- **Source**: Rhasspy Project
- **Repository**: https://github.com/rhasspy/piper
- **License**: MIT
- **Voice Model**: en_US-lessac-medium
- **Citation**:
  ```
  @software{Piper_TTS_2023,
    author = {Hansen, Michael},
    license = {MIT},
    title = {Piper},
    url = {https://github.com/rhasspy/piper},
    year = {2023}
  }
  ```
- **Usage**: Text-to-speech synthesis for robot responses
- **Voice**: Lessac medium quality English (US) voice

### openWakeWord

- **Source**: David Scripka
- **Repository**: https://github.com/dscripka/openWakeWord
- **License**: Apache 2.0
- **Citation**:
  ```
  @software{openWakeWord_2023,
    author = {Scripka, David},
    license = {Apache-2.0},
    title = {openWakeWord},
    url = {https://github.com/dscripka/openWakeWord},
    year = {2023}
  }
  ```
- **Usage**: Wake word detection for hands-free activation
- **Models**: Pre-trained models for common wake words

## Language Models

### Base LLM Options

The system supports multiple LLM options depending on requirements:

#### Option 1: Microsoft Phi-3 Mini

- **Source**: Microsoft
- **Repository**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- **License**: MIT
- **Model Size**: 3.8B parameters
- **Citation**:
  ```
  @misc{phi3_2024,
    title={Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone},
    author={Microsoft},
    year={2024},
    url={https://huggingface.co/microsoft/Phi-3-mini-4k-instruct}
  }
  ```
- **Usage**: Intent understanding and conversational AI
- **Modifications**: Quantized using NVIDIA NanoLLM for efficient inference

#### Option 2: Meta LLaMA 2 7B

- **Source**: Meta AI
- **Repository**: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- **License**: Custom LLaMA 2 License (Commercial use allowed with restrictions)
- **Model Size**: 7B parameters
- **Citation**:
  ```
  @misc{touvron2023llama,
    title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
    author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and others},
    journal={arXiv preprint arXiv:2307.09288},
    year={2023}
  }
  ```
- **Usage**: Advanced reasoning and conversation capabilities
- **Modifications**: Quantized to INT4 using AWQ for memory efficiency

#### Option 3: Google Gemma 2-7B

- **Source**: Google
- **Repository**: https://huggingface.co/google/gemma-2-7b-it
- **License**: Gemma Terms of Use
- **Model Size**: 7B parameters
- **Citation**:
  ```
  @misc{gemma_2024,
    title={Gemma: Open Models Based on Gemini Research and Technology},
    author={Google DeepMind},
    year={2024},
    url={https://huggingface.co/google/gemma-2-7b-it}
  }
  ```
- **Usage**: Balanced performance and efficiency for cognitive tasks
- **Modifications**: Quantized using NVIDIA NanoLLM

## Model Optimization Tools

### TensorRT

- **Source**: NVIDIA
- **License**: NVIDIA Software License
- **Usage**: Accelerating inference for vision models on Jetson hardware
- **Documentation**: https://developer.nvidia.com/tensorrt

### NanoLLM

- **Source**: NVIDIA Jetson Community
- **Repository**: https://github.com/dusty-nv/NanoLLM
- **License**: MIT
- **Usage**: Optimizing and running LLMs on Jetson devices
- **Citation**:
  ```
  @software{NanoLLM_2024,
    author = {Walsh, Dustin},
    title = {NanoLLM},
    url = {https://github.com/dusty-nv/NanoLLM},
    year = {2024}
  }
  ```

### faster-whisper

- **Source**: Guillaume Klein
- **Repository**: https://github.com/guillaumekln/faster-whisper
- **License**: MIT
- **Usage**: Optimized Whisper inference using CTranslate2
- **Citation**:
  ```
  @software{faster_whisper_2023,
    author = {Klein, Guillaume},
    title = {faster-whisper},
    url = {https://github.com/guillaumekln/faster-whisper},
    year = {2023}
  }
  ```

## Compliance Notes

### Commercial Use

- **YOLOv8**: GPL-3.0 license requires derivative works to be open source
- **FastDepth**: MIT license allows commercial use with attribution
- **Whisper**: MIT license allows commercial use with attribution
- **Piper**: MIT license allows commercial use with attribution
- **openWakeWord**: Apache 2.0 allows commercial use with attribution
- **Phi-3**: MIT license allows commercial use with attribution
- **LLaMA 2**: Custom license allows commercial use under 700M monthly active users
- **Gemma**: Custom license with specific terms for commercial use

### Attribution Requirements

When distributing this software or derivatives:

1. Include this credits file in any distribution
2. Maintain original copyright notices for all models
3. Include license texts for GPL-licensed components (YOLOv8)
4. Provide clear attribution for all models used

### Research and Educational Use

All models are approved for research and educational purposes. This project serves as an educational platform for learning about:

- Computer vision and object detection
- Speech recognition and synthesis
- Natural language processing
- Robotics and autonomous systems
- Edge AI and model optimization

## Model Performance Targets

### Inference Benchmarks on NVIDIA Jetson Orin Nano

- **YOLOv8n**: <50ms per frame (640x480 input)
- **FastDepth**: <70ms per frame (640x480 input)
- **Whisper Tiny**: Real-time factor <0.3x (faster than real-time)
- **Piper TTS**: <500ms for 20-word synthesis
- **Wake Word**: <100ms detection latency, <5% CPU usage
- **LLM**: <3s for typical response (10-50 tokens)

### Memory Usage Targets

- **Total system**: <7.5GB RAM usage
- **Vision models**: ~1.5GB VRAM when loaded
- **Audio models**: ~500MB RAM when loaded
- **LLM**: ~2.5GB RAM when loaded (quantized)
- **Lazy loading**: Models loaded/unloaded based on memory pressure

## Contact and Support

For questions about model usage, licensing, or optimization:

1. Review original model documentation and licenses
2. Consult project documentation in `/docs`
3. Open GitHub issues for project-specific questions
4. Contact original model authors for model-specific questions

## Disclaimer

This project uses pre-trained models from various sources. Users are responsible for ensuring compliance with all applicable licenses and terms of use. The project maintainers provide this information for reference but do not provide legal advice regarding model usage.

**Last Updated**: November 2024
**Project Version**: 1.0
