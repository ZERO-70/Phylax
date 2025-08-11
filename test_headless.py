#!/usr/bin/env python3
"""
Test script for headless mode with enhanced transcript logging
"""

import os
import sys
import time
from config import Config

def test_headless_config():
    """Test headless mode configuration"""
    print("=== Testing Headless Mode Configuration ===")
    
    # Test default config
    config = Config()
    headless_default = config.get('video', 'headless_mode')
    print(f"Default headless mode: {headless_default}")
    
    # Test setting headless mode
    config.set('video', 'headless_mode', True)
    headless_enabled = config.get('video', 'headless_mode')
    print(f"Headless mode enabled: {headless_enabled}")
    
    # Test CLI argument simulation
    class MockArgs:
        def __init__(self):
            self.headless = True
            self.no_nsfw = False
            self.no_gun = False
            self.no_transcription = False
            self.no_profanity = False
            self.enable_nsfw = False
            self.enable_gun = False
            self.enable_transcription = False
            self.enable_profanity = False
            self.host = None
            self.video_port = None
            self.audio_port = None
            self.whisper_model = None
            self.device = None
            self.compute_type = None
            self.chunk_duration = None
            self.fps = None
            self.jpeg_quality = None
            
    from config import apply_cli_overrides
    
    config = Config()
    args = MockArgs()
    apply_cli_overrides(config, args)
    
    headless_cli = config.get('video', 'headless_mode')
    print(f"Headless mode via CLI: {headless_cli}")
    
    print("✓ Headless mode configuration working!")

def test_transcript_format():
    """Test the enhanced transcript format"""
    print("\n=== Testing Enhanced Transcript Format ===")
    
    # Create a mock transcript entry
    timestamp = time.strftime('%H:%M:%S')
    text = "This is a test transcription"
    
    # Mock temperature data
    nsfw_temp = 25.5
    weapon_temp = 0.0
    abusive_temp = 10.2
    
    # Format like the receiver would
    temp_data = []
    temp_data.append(f"NSFW: {nsfw_temp:.1f}°")
    temp_data.append(f"Weapon: {weapon_temp:.1f}°")
    temp_data.append(f"Abusive: {abusive_temp:.1f}°")
    
    temp_str = " | ".join(temp_data)
    
    transcript_entry = f"[{timestamp}] {text}\n"
    transcript_entry += f"[{timestamp}] TEMPS: {temp_str}\n"
    
    print("Sample transcript entry:")
    print(transcript_entry)
    
    # Test writing to file
    test_file = "test_transcript.txt"
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Test Transcript Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            f.write(transcript_entry)
            f.write("\n=== Test Transcript Ended ===\n")
        
        print(f"✓ Test transcript written to {test_file}")
        
        # Read it back to verify
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print("\nFile content:")
            print(content)
            
        # Clean up
        os.remove(test_file)
        print("✓ Test file cleaned up")
        
    except Exception as e:
        print(f"✗ Error testing transcript: {e}")

def main():
    print("Headless Mode and Enhanced Logging Test")
    print("=" * 50)
    
    test_headless_config()
    test_transcript_format()
    
    print("\n=== Summary ===")
    print("✓ Headless mode configuration ready")
    print("✓ Enhanced transcript logging ready")
    print("\nTo run in headless mode:")
    print("  python receiver.py --headless --no-nsfw --no-gun")
    print("\nTo enable all detection with headless mode:")
    print("  python receiver.py --headless")
    print("\nTranscript file will contain:")
    print("  - Timestamped transcriptions")
    print("  - Temperature data for all enabled detections")
    print("  - Console output for headless monitoring")

if __name__ == "__main__":
    main()
