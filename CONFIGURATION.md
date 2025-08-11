# Configuration Guide for Video Streaming with Content Detection

This guide explains how to configure and customize the video streaming application with various content detection features.

## Quick Start

### Method 1: Interactive Configuration Tool (Recommended)
```bash
python configure.py
```
This launches an interactive menu where you can easily enable/disable features and adjust settings.

### Method 2: Command Line Arguments
```bash
# Disable NSFW detection
python receiver.py --no-nsfw

# Disable all content detection
python receiver.py --no-nsfw --no-gun --no-transcription --no-profanity

# Use different ports
python receiver.py --video-port 8888 --audio-port 8887

# Use larger Whisper model for better transcription
python receiver.py --whisper-model base
```

### Method 3: Configuration File
Edit `stream_config.json` directly or use the `--save-config` flag to save current settings.

## Available Features

### 1. NSFW Detection
**What it does**: Automatically detects inappropriate content in video frames and applies blur.

**Configuration options**:
- `enabled`: Enable/disable NSFW detection
- `time_interval`: How often to check frames (in seconds)
- `blur_strength`: Intensity of blur applied to NSFW frames

**Command line examples**:
```bash
# Disable NSFW detection
python receiver.py --no-nsfw

# Enable NSFW detection (if disabled in config)
python receiver.py --enable-nsfw
```

### 2. Gun/Weapon Detection
**What it does**: Detects firearms and weapons in video frames and draws bounding boxes.

**Configuration options**:
- `enabled`: Enable/disable gun detection
- `time_interval`: How often to check frames (in seconds)
- `confidence_threshold`: Minimum confidence for detection (0.0-1.0)

**Command line examples**:
```bash
# Disable gun detection
python receiver.py --no-gun

# Enable gun detection (if disabled in config)
python receiver.py --enable-gun
```

### 3. Audio Transcription
**What it does**: Converts speech in audio to text and saves to a transcript file.

**Configuration options**:
- `enabled`: Enable/disable transcription
- `whisper_model`: Model size (tiny, base, small, medium, large)
- `device`: Processing device (cpu, cuda, auto)
- `compute_type`: Computation precision (int8, float16, float32)

**Command line examples**:
```bash
# Disable transcription
python receiver.py --no-transcription

# Enable transcription with larger model
python receiver.py --enable-transcription --whisper-model base
```

### 4. Profanity Filter
**What it does**: Filters inappropriate language from transcriptions.

**Configuration options**:
- `enabled`: Enable/disable profanity filtering
- `wordlist_file`: File containing profane words
- `replacement_char`: Character used to replace profane words

**Command line examples**:
```bash
# Disable profanity filtering
python receiver.py --no-profanity

# Enable profanity filtering (if disabled in config)
python receiver.py --enable-profanity
```

## Configuration File Format

The `stream_config.json` file structure:

```json
{
  "nsfw_detection": {
    "enabled": true,
    "time_interval": 0.5,
    "blur_strength": 51
  },
  "gun_detection": {
    "enabled": true,
    "time_interval": 2.0,
    "confidence_threshold": 0.4
  },
  "transcription": {
    "enabled": true,
    "whisper_model": "small",
    "device": "cpu",
    "compute_type": "int8"
  },
  "profanity_filter": {
    "enabled": true,
    "wordlist_file": "profanity_wordlist.txt",
    "replacement_char": "*"
  },
  "network": {
    "host": "localhost",
    "video_port": 9999,
    "audio_port": 9998
  },
  "video": {
    "target_fps": 24,
    "jpeg_quality": 80,
    "scale_factor": 0.9
  }
}
```

## Usage Examples

### Example 1: Privacy-focused Setup
For environments where content detection is not needed:
```bash
python receiver.py --no-nsfw --no-gun --enable-transcription --enable-profanity
```

### Example 2: Security-focused Setup
For monitoring scenarios:
```bash
python receiver.py --enable-nsfw --enable-gun --no-transcription --no-profanity
```

### Example 3: Full Analysis Setup
For complete content analysis:
```bash
python receiver.py --enable-nsfw --enable-gun --enable-transcription --enable-profanity --whisper-model base
```

### Example 4: Custom Network Setup
For different network configuration:
```bash
python receiver.py --host 192.168.1.100 --video-port 8080 --audio-port 8081
```

## Performance Considerations

### CPU Usage
- **NSFW Detection**: Moderate CPU usage, adjustable via `time_interval`
- **Gun Detection**: Light CPU usage (uses cloud API)
- **Transcription**: High CPU usage, consider using `tiny` model for better performance
- **Profanity Filter**: Minimal CPU usage

### Memory Usage
- **Whisper Models**:
  - tiny: ~39 MB
  - base: ~74 MB  
  - small: ~244 MB
  - medium: ~769 MB
  - large: ~1550 MB

### Network Usage
- **Gun Detection**: Requires internet connection for API calls
- **Video Streaming**: Quality adjustable via `jpeg_quality` setting

## Troubleshooting

### Common Issues

1. **NSFW model download fails**:
   ```bash
   # Set HuggingFace cache directory
   export HF_HOME=/path/to/writable/directory
   ```

2. **Whisper model fails to load**:
   ```bash
   # Try smaller model
   python receiver.py --whisper-model tiny
   ```

3. **Gun detection API errors**:
   - Check internet connection
   - Verify API key in source code
   - Increase `time_interval` to reduce API calls

4. **Profanity wordlist not found**:
   - Ensure `profanity_wordlist.txt` exists
   - Or disable profanity filtering: `--no-profanity`

### Performance Optimization

1. **For slower systems**:
   ```bash
   python receiver.py --whisper-model tiny --no-nsfw
   ```

2. **For faster processing**:
   ```bash
   python receiver.py --whisper-model base --device cuda
   ```

3. **For minimal resource usage**:
   ```bash
   python receiver.py --no-nsfw --no-gun --no-transcription --no-profanity
   ```

## Advanced Configuration

### Custom Profanity Wordlist
Create your own `profanity_wordlist.txt` with one word per line:
```
inappropriate_word1
inappropriate_word2
```

### Multiple Configuration Files
Use different configs for different scenarios:
```bash
python receiver.py --config monitoring.json
python receiver.py --config entertainment.json
```

### Saving Current Settings
```bash
# Save current command line settings to config file
python receiver.py --no-nsfw --enable-transcription --save-config
```

## Integration with Other Tools

The application creates these output files:
- `audio_transcript.txt`: Real-time transcription with timestamps
- `nsfw_detection.log`: Detection events and system logs
- `stream_config.json`: Current configuration

These can be monitored by external tools for alerts or analysis.
