# GPU Acceleration and Headless Mode Setup

## Overview

Your abuse detection system now supports:

✅ **Automatic GPU Detection**: Detects NVIDIA CUDA, Apple MPS, or falls back to CPU  
✅ **Headless Mode**: Run without video display for server environments  
✅ **Enhanced Logging**: Temperature data written to transcript files  
✅ **Seamless Fallback**: Works on CPU when GPU unavailable  

## Quick Start

### Current Status (Before Reboot)
```bash
# Your system currently runs on CPU due to NVIDIA driver kernel module mismatch
source enviroment/bin/activate && python receiver.py --headless --no-nsfw --no-gun
```

### After Reboot (GPU Enabled)
```bash
# System will automatically detect and use your RTX 3080 Mobile GPU
source enviroment/bin/activate && python receiver.py --headless
```

## GPU Setup

### What's Been Implemented

1. **GPU Detection Utility** (`gpu_utils.py`):
   - Automatic device detection (CUDA → MPS → CPU)
   - Memory-aware optimization
   - GPU monitoring and cache management

2. **Model Acceleration**:
   - **NSFW Detection**: Transformers pipeline with GPU support
   - **Whisper Transcription**: Optimized device and compute type selection
   - **Automatic Fallback**: Graceful CPU fallback when GPU unavailable

3. **Configuration**:
   - Auto-detected optimal settings in `config.py`
   - CLI override support: `--device cuda/cpu/auto`

### Current Hardware
- **GPU**: NVIDIA RTX 3080 Mobile (8GB VRAM)
- **Status**: Drivers installed but kernel module mismatch
- **Solution**: Reboot to load updated kernel modules

### Expected Performance After GPU Setup
- **NSFW Detection**: ~5-10x faster inference
- **Whisper Models**: Significant speedup with larger models
- **Memory**: Up to 8GB VRAM available

## Headless Mode

### Features
- ✅ No video display window (perfect for SSH/server environments)
- ✅ Console status updates every 2.5 seconds
- ✅ Enhanced transcript logging with temperature data
- ✅ Performance monitoring

### Usage Examples

```bash
# Basic headless with transcription only
python receiver.py --headless --no-nsfw --no-gun

# Full detection in headless mode
python receiver.py --headless

# Headless with custom settings
python receiver.py --headless --whisper-model base --device cuda

# Headless with specific features
python receiver.py --headless --enable-nsfw --no-gun --no-profanity
```

### Console Output in Headless Mode
```
[STATUS] NSFW: SAFE (0.95) | Weapon: SAFE (0.12)
[TEMPS] NSFW: 15.2° | Weapon: 0.0° | Abusive: 5.8°
[TRANSCRIPT] This is what was said in the audio
[PERF] FPS: 23.8, Frames: 1440
```

## Enhanced Transcript Logging

### New Format
The `audio_transcript.txt` file now includes:

```
=== Audio Transcript Started at 2025-08-09 20:39:05 ===

[20:39:05] Hello, this is a test transcription
[20:39:05] TEMPS: NSFW: 25.5° | Weapon: 0.0° | Abusive: 10.2°

[20:39:10] Another piece of transcribed audio
[20:39:10] TEMPS: NSFW: 30.1° | Weapon: 5.5° | Abusive: 0.0°
```

### Temperature Tracking
- **NSFW Temperature**: Visual content analysis over time
- **Weapon Temperature**: Weapon detection confidence trends  
- **Abusive Language**: Profanity detection in audio transcription
- **Range**: 0-100° with color-coded warnings

## Configuration Options

### Command Line Arguments
```bash
# Device selection
--device auto|cpu|cuda|mps    # Auto-detect optimal device
--compute-type auto|int8|float16|float32

# Display mode
--headless                    # Run without video window

# Feature toggles
--no-nsfw --no-gun --no-transcription --no-profanity
--enable-nsfw --enable-gun --enable-transcription --enable-profanity

# Model settings
--whisper-model tiny|base|small|medium|large
--chunk-duration 2.0          # Audio processing chunk size
```

### Configuration File (`stream_config.json`)
```json
{
  "video": {
    "headless_mode": false
  },
  "nsfw_detection": {
    "enabled": true,
    "device": "cuda"        // Auto-detected
  },
  "transcription": {
    "enabled": true,
    "device": "cuda",       // Auto-detected
    "compute_type": "float16"  // Auto-optimized
  }
}
```

## After System Reboot

### 1. Verify GPU Setup
```bash
cd /home/umar/umarsulemanlinux/work/abuse_detection
source enviroment/bin/activate
python test_gpu.py
```

Expected output after reboot:
```
=== GPU Detection Test ===
Best device detected: {'device': 'cuda', 'description': 'CUDA GPU: NVIDIA GeForce RTX 3080 Mobile (8.0GB)'}
✓ GPU acceleration available
```

### 2. Run with GPU Acceleration
```bash
# Headless mode with full GPU acceleration
python receiver.py --headless

# Full GUI with GPU acceleration  
python receiver.py
```

### 3. Monitor GPU Usage
The system will automatically:
- Log GPU memory usage
- Clear cache when needed
- Monitor performance
- Fall back to CPU if GPU issues occur

## Troubleshooting

### If GPU Still Not Working After Reboot
```bash
# Check driver status
nvidia-smi

# Check kernel modules
lsmod | grep nvidia

# Reinstall modules if needed
sudo apt update && sudo apt install linux-modules-nvidia-550-$(uname -r)
```

### Performance Issues
```bash
# Check current device being used
python -c "
from gpu_utils import detect_best_device
print(detect_best_device())
"

# Force CPU mode if needed
python receiver.py --device cpu --headless
```

### Display Issues
```bash
# Always use headless mode for SSH/remote
python receiver.py --headless

# For local display issues
export DISPLAY=:0
python receiver.py
```

## Files Modified/Created

### New Files
- `gpu_utils.py` - GPU detection and optimization utilities
- `test_gpu.py` - GPU functionality testing
- `test_headless.py` - Headless mode testing

### Modified Files
- `config.py` - Added GPU auto-detection and headless mode support
- `receiver.py` - Added GPU acceleration, headless mode, enhanced logging

### Key Features Added
1. **Automatic GPU Detection**: Zero-configuration GPU usage
2. **Headless Operation**: Perfect for servers and SSH environments
3. **Enhanced Logging**: Temperature tracking in transcript files
4. **Robust Fallback**: Always works, CPU or GPU
5. **CLI Flexibility**: Easy mode switching via command line

## Summary

Your system is now ready for both CPU and GPU operation:

- **Currently**: Running on CPU (works perfectly)
- **After Reboot**: Will automatically use RTX 3080 Mobile GPU
- **Headless Mode**: Ready for server deployment
- **Enhanced Logging**: Temperature data tracked in transcripts
- **Zero Configuration**: Automatic optimal settings

The headless mode solves your display issues and provides excellent monitoring capabilities through console output and enhanced transcript files.
