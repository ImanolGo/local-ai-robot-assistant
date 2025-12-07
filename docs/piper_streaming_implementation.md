# Piper TTS Streaming Implementation

## New Features Added

The Piper TTS test script now includes real-time audio streaming capabilities, similar to the audio devices test framework.

### New Command Line Options

```bash
--stream              # Stream audio to speakers in real-time
--stream-benchmark    # Run streaming benchmark tests
```

### Usage Examples

#### Basic Streaming
```bash
# Stream text directly to speakers
python scripts/test_piper_tts.py --stream --text "Hello, this is streaming TTS!"
```

#### Streaming vs File-based Comparison
```bash
# Regular synthesis (saves to file)
python scripts/test_piper_tts.py --text "Test sentence"

# Streaming synthesis (plays through speakers)
python scripts/test_piper_tts.py --stream --text "Test sentence"
```

#### Performance Benchmarking
```bash
# Regular synthesis benchmark
python scripts/test_piper_tts.py --benchmark

# Streaming synthesis benchmark (includes playback time)
python scripts/test_piper_tts.py --stream-benchmark
```

#### Combined Testing
```bash
# Run all tests including streaming
python scripts/test_piper_tts.py --benchmark --stream-benchmark --quality --text "Comprehensive test"
```

### Technical Implementation

#### Streaming Method
```python
def synthesize_stream(self, text: str) -> float:
    """Synthesize text to speech with real-time audio streaming playback."""
    # Setup sounddevice OutputStream
    stream = sd.OutputStream(
        samplerate=self.voice.config.sample_rate,
        channels=1,
        dtype='int16'
    )
    stream.start()

    try:
        # Stream audio chunks as they're generated
        for audio_chunk in self.voice.synthesize(text):
            audio_bytes = audio_chunk.audio_int16_bytes
            int_data = np.frombuffer(audio_bytes, dtype=np.int16)
            stream.write(int_data)
    finally:
        stream.stop()
        stream.close()
```

#### Dependencies
- `sounddevice` - For real-time audio output
- `numpy` - For audio data processing
- `piper-tts` - Core TTS functionality

### Performance Comparison

#### Regular Synthesis (File Output)
- **2 words**: 0.101s (0.050s/word) ✓ PASS
- **9 words**: 0.295s (0.033s/word) ✓ PASS
- **14 words**: 0.380s (0.027s/word) ✓ PASS

#### Streaming Synthesis (Real-time Playback)
- **2 words**: 0.998s (0.499s/word) [STREAMING]
- **9 words**: 2.833s (0.315s/word) [STREAMING]
- **14 words**: 4.698s (0.336s/word) [STREAMING]

**Note**: Streaming times include both synthesis and real-time audio playback duration.

### Audio Configuration

- **Sample Rate**: 22,050 Hz (matches Piper voice model)
- **Channels**: 1 (Mono)
- **Format**: 16-bit PCM
- **Chunk-based**: Streams audio as it's synthesized for minimal latency

### Use Cases

1. **Interactive Robot Responses**: Real-time speech feedback
2. **Voice Assistants**: Immediate audio output
3. **Live Demonstrations**: Streaming TTS for presentations
4. **Development Testing**: Quick audio verification without files

### Error Handling

- Graceful fallback if `sounddevice` not available
- Proper audio stream cleanup on interruption
- ALSA underrun warnings are normal and expected

This implementation provides the same streaming capabilities as shown in the audio devices test, allowing for real-time text-to-speech playback through connected speakers.
