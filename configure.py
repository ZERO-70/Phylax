#!/usr/bin/env python3
"""
Interactive Configuration Tool for Video Streaming with Content Detection

This script provides an easy way to configure the video streaming application
without having to edit configuration files or remember command line arguments.
"""

import json
import os
from config import Config, DEFAULT_CONFIG

def print_banner():
    print("=" * 60)
    print("     Video Streaming Configuration Tool")
    print("=" * 60)
    print()

def display_current_config(config):
    print("Current Configuration:")
    print("-" * 30)
    
    # NSFW Detection
    nsfw_enabled = config.is_enabled('nsfw_detection')
    print(f"1. NSFW Detection: {'✓ Enabled' if nsfw_enabled else '✗ Disabled'}")
    if nsfw_enabled:
        print(f"   - Check interval: {config.get('nsfw_detection', 'time_interval')}s")
        print(f"   - Blur strength: {config.get('nsfw_detection', 'blur_strength')}")
    
    # Gun Detection
    gun_enabled = config.is_enabled('gun_detection')
    print(f"2. Gun Detection: {'✓ Enabled' if gun_enabled else '✗ Disabled'}")
    if gun_enabled:
        print(f"   - Check interval: {config.get('gun_detection', 'time_interval')}s")
        print(f"   - Confidence threshold: {config.get('gun_detection', 'confidence_threshold')}")
    
    # Transcription
    trans_enabled = config.is_enabled('transcription')
    print(f"3. Audio Transcription: {'✓ Enabled' if trans_enabled else '✗ Disabled'}")
    if trans_enabled:
        print(f"   - Whisper model: {config.get('transcription', 'whisper_model')}")
        print(f"   - Device: {config.get('transcription', 'device')}")
    
    # Profanity Filter
    prof_enabled = config.is_enabled('profanity_filter')
    print(f"4. Profanity Filter: {'✓ Enabled' if prof_enabled else '✗ Disabled'}")
    if prof_enabled:
        print(f"   - Replacement char: '{config.get('profanity_filter', 'replacement_char')}'")
        print(f"   - Wordlist file: {config.get('profanity_filter', 'wordlist_file')}")
    
    # Network
    print(f"5. Network Settings:")
    print(f"   - Host: {config.get('network', 'host')}")
    print(f"   - Video port: {config.get('network', 'video_port')}")
    print(f"   - Audio port: {config.get('network', 'audio_port')}")
    
    # Video
    print(f"6. Video Settings:")
    print(f"   - Target FPS: {config.get('video', 'target_fps')}")
    print(f"   - JPEG quality: {config.get('video', 'jpeg_quality')}%")
    print(f"   - Scale factor: {config.get('video', 'scale_factor')}")
    
    print()

def toggle_feature(config, feature_name, feature_display_name):
    """Toggle a feature on/off"""
    current_state = config.is_enabled(feature_name)
    new_state = not current_state
    config.set(feature_name, 'enabled', new_state)
    
    status = "enabled" if new_state else "disabled"
    print(f"✓ {feature_display_name} {status}")

def configure_nsfw_settings(config):
    """Configure NSFW detection settings"""
    print("\nNSFW Detection Settings:")
    print("1. Toggle on/off")
    print("2. Change check interval")
    print("3. Change blur strength")
    print("4. Back to main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        toggle_feature(config, 'nsfw_detection', 'NSFW Detection')
    elif choice == '2':
        current = config.get('nsfw_detection', 'time_interval')
        print(f"Current interval: {current}s")
        try:
            new_interval = float(input("Enter new interval (seconds): "))
            config.set('nsfw_detection', 'time_interval', new_interval)
            print(f"✓ Check interval set to {new_interval}s")
        except ValueError:
            print("✗ Invalid number")
    elif choice == '3':
        current = config.get('nsfw_detection', 'blur_strength')
        print(f"Current blur strength: {current}")
        try:
            new_strength = int(input("Enter new blur strength (odd number, 1-101): "))
            if new_strength % 2 == 0:
                new_strength += 1
            config.set('nsfw_detection', 'blur_strength', new_strength)
            print(f"✓ Blur strength set to {new_strength}")
        except ValueError:
            print("✗ Invalid number")

def configure_gun_settings(config):
    """Configure gun detection settings"""
    print("\nGun Detection Settings:")
    print("1. Toggle on/off")
    print("2. Change check interval")
    print("3. Change confidence threshold")
    print("4. Back to main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        toggle_feature(config, 'gun_detection', 'Gun Detection')
    elif choice == '2':
        current = config.get('gun_detection', 'time_interval')
        print(f"Current interval: {current}s")
        try:
            new_interval = float(input("Enter new interval (seconds): "))
            config.set('gun_detection', 'time_interval', new_interval)
            print(f"✓ Check interval set to {new_interval}s")
        except ValueError:
            print("✗ Invalid number")
    elif choice == '3':
        current = config.get('gun_detection', 'confidence_threshold')
        print(f"Current threshold: {current}")
        try:
            new_threshold = float(input("Enter new threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                config.set('gun_detection', 'confidence_threshold', new_threshold)
                print(f"✓ Confidence threshold set to {new_threshold}")
            else:
                print("✗ Threshold must be between 0.0 and 1.0")
        except ValueError:
            print("✗ Invalid number")

def configure_transcription_settings(config):
    """Configure transcription settings"""
    print("\nTranscription Settings:")
    print("1. Toggle on/off")
    print("2. Change Whisper model")
    print("3. Change device")
    print("4. Change chunk duration (processing speed)")
    print("5. Back to main menu")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        toggle_feature(config, 'transcription', 'Audio Transcription')
    elif choice == '2':
        current = config.get('transcription', 'whisper_model')
        print(f"Current model: {current}")
        print("Available models: tiny, base, small, medium, large")
        new_model = input("Enter model name: ").strip()
        if new_model in ['tiny', 'base', 'small', 'medium', 'large']:
            config.set('transcription', 'whisper_model', new_model)
            print(f"✓ Whisper model set to {new_model}")
        else:
            print("✗ Invalid model name")
    elif choice == '3':
        current = config.get('transcription', 'device')
        print(f"Current device: {current}")
        print("Common devices: cpu, cuda, auto")
        new_device = input("Enter device: ").strip()
        config.set('transcription', 'device', new_device)
        print(f"✓ Device set to {new_device}")
    elif choice == '4':
        current = config.get('transcription', 'chunk_duration')
        print(f"Current chunk duration: {current}s")
        print("Smaller values = faster response, larger values = better accuracy")
        print("Recommended: 1.0-3.0 seconds")
        try:
            new_duration = float(input("Enter chunk duration (seconds): "))
            if 0.5 <= new_duration <= 10.0:
                config.set('transcription', 'chunk_duration', new_duration)
                print(f"✓ Chunk duration set to {new_duration}s")
            else:
                print("✗ Duration must be between 0.5 and 10.0 seconds")
        except ValueError:
            print("✗ Invalid number")

def configure_profanity_settings(config):
    """Configure profanity filter settings"""
    print("\nProfanity Filter Settings:")
    print("1. Toggle on/off")
    print("2. Change replacement character")
    print("3. Change wordlist file")
    print("4. Back to main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        toggle_feature(config, 'profanity_filter', 'Profanity Filter')
    elif choice == '2':
        current = config.get('profanity_filter', 'replacement_char')
        print(f"Current replacement char: '{current}'")
        new_char = input("Enter new replacement character: ").strip()
        if len(new_char) == 1:
            config.set('profanity_filter', 'replacement_char', new_char)
            print(f"✓ Replacement character set to '{new_char}'")
        else:
            print("✗ Please enter a single character")
    elif choice == '3':
        current = config.get('profanity_filter', 'wordlist_file')
        print(f"Current wordlist file: {current}")
        new_file = input("Enter wordlist filename: ").strip()
        config.set('profanity_filter', 'wordlist_file', new_file)
        print(f"✓ Wordlist file set to {new_file}")

def configure_network_settings(config):
    """Configure network settings"""
    print("\nNetwork Settings:")
    print("1. Change host")
    print("2. Change video port")
    print("3. Change audio port")
    print("4. Back to main menu")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        current = config.get('network', 'host')
        print(f"Current host: {current}")
        new_host = input("Enter new host: ").strip()
        config.set('network', 'host', new_host)
        print(f"✓ Host set to {new_host}")
    elif choice == '2':
        current = config.get('network', 'video_port')
        print(f"Current video port: {current}")
        try:
            new_port = int(input("Enter new video port: "))
            config.set('network', 'video_port', new_port)
            print(f"✓ Video port set to {new_port}")
        except ValueError:
            print("✗ Invalid port number")
    elif choice == '3':
        current = config.get('network', 'audio_port')
        print(f"Current audio port: {current}")
        try:
            new_port = int(input("Enter new audio port: "))
            config.set('network', 'audio_port', new_port)
            print(f"✓ Audio port set to {new_port}")
        except ValueError:
            print("✗ Invalid port number")

def quick_presets(config):
    """Apply quick configuration presets"""
    print("\nQuick Presets:")
    print("1. All features enabled (default)")
    print("2. Content detection only (no transcription)")
    print("3. Transcription only (no content detection)")
    print("4. Minimal (video streaming only)")
    print("5. Back to main menu")
    
    choice = input("\nSelect preset (1-5): ").strip()
    
    if choice == '1':
        # All features enabled
        config.set('nsfw_detection', 'enabled', True)
        config.set('gun_detection', 'enabled', True)
        config.set('transcription', 'enabled', True)
        config.set('profanity_filter', 'enabled', True)
        print("✓ All features enabled")
    elif choice == '2':
        # Content detection only
        config.set('nsfw_detection', 'enabled', True)
        config.set('gun_detection', 'enabled', True)
        config.set('transcription', 'enabled', False)
        config.set('profanity_filter', 'enabled', False)
        print("✓ Content detection enabled, transcription disabled")
    elif choice == '3':
        # Transcription only
        config.set('nsfw_detection', 'enabled', False)
        config.set('gun_detection', 'enabled', False)
        config.set('transcription', 'enabled', True)
        config.set('profanity_filter', 'enabled', True)
        print("✓ Transcription enabled, content detection disabled")
    elif choice == '4':
        # Minimal
        config.set('nsfw_detection', 'enabled', False)
        config.set('gun_detection', 'enabled', False)
        config.set('transcription', 'enabled', False)
        config.set('profanity_filter', 'enabled', False)
        print("✓ All features disabled (video streaming only)")

def main():
    config_file = "stream_config.json"
    
    print_banner()
    
    # Load or create configuration
    config = Config(config_file)
    
    while True:
        display_current_config(config)
        
        print("Options:")
        print("1. Configure NSFW Detection")
        print("2. Configure Gun Detection") 
        print("3. Configure Audio Transcription")
        print("4. Configure Profanity Filter")
        print("5. Configure Network Settings")
        print("6. Quick Presets")
        print("7. Save & Exit")
        print("8. Exit without saving")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            configure_nsfw_settings(config)
        elif choice == '2':
            configure_gun_settings(config)
        elif choice == '3':
            configure_transcription_settings(config)
        elif choice == '4':
            configure_profanity_settings(config)
        elif choice == '5':
            configure_network_settings(config)
        elif choice == '6':
            quick_presets(config)
        elif choice == '7':
            config.save_config()
            print(f"\n✓ Configuration saved to {config_file}")
            print("You can now run the application with your settings!")
            break
        elif choice == '8':
            print("\nExiting without saving...")
            break
        else:
            print("✗ Invalid option. Please try again.")
        
        print()

if __name__ == "__main__":
    main()
