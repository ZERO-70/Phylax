#!/usr/bin/env python3
"""
Example Usage Scenarios for Video Streaming with Content Detection

This script demonstrates different ways to run the video streaming application
with various configuration options.
"""

import subprocess
import sys
import time

def run_example(name, description, sender_args=None, receiver_args=None):
    """Run an example configuration"""
    print("=" * 60)
    print(f"EXAMPLE: {name}")
    print("=" * 60)
    print(f"Description: {description}")
    print()
    
    if sender_args is None:
        sender_args = []
    if receiver_args is None:
        receiver_args = []
    
    sender_cmd = ["python", "sender.py"] + sender_args
    receiver_cmd = ["python", "receiver.py"] + receiver_args
    
    print("Commands to run:")
    print(f"Terminal 1: {' '.join(sender_cmd)}")
    print(f"Terminal 2: {' '.join(receiver_cmd)}")
    print()
    
    response = input("Run this example? (y/N): ").strip().lower()
    if response == 'y':
        print("Starting sender...")
        print("Note: You'll need to run the receiver command in another terminal")
        print(f"Receiver command: {' '.join(receiver_cmd)}")
        print()
        try:
            subprocess.run(sender_cmd)
        except KeyboardInterrupt:
            print("\\nExample stopped by user")
    
    print("\\n")

def show_menu():
    """Show the main menu"""
    print("Video Streaming Configuration Examples")
    print("=" * 40)
    print()
    print("Choose an example to try:")
    print()
    print("1. Default (All Features)")
    print("2. Privacy Mode (No Content Detection)")  
    print("3. Security Mode (Detection Only)")
    print("4. Transcription Only")
    print("5. Minimal (Video Only)")
    print("6. High Quality Transcription")
    print("7. Custom Network Settings")
    print("8. Performance Optimized")
    print("9. Show Command Reference")
    print("0. Exit")
    print()

def show_command_reference():
    """Show all available command line options"""
    print("Command Line Reference")
    print("=" * 30)
    print()
    print("Feature Control:")
    print("  --no-nsfw              Disable NSFW detection")
    print("  --no-gun               Disable gun detection")  
    print("  --no-transcription     Disable audio transcription")
    print("  --no-profanity         Disable profanity filtering")
    print("  --enable-nsfw          Force enable NSFW detection")
    print("  --enable-gun           Force enable gun detection")
    print("  --enable-transcription Force enable transcription") 
    print("  --enable-profanity     Force enable profanity filtering")
    print()
    print("Model Settings:")
    print("  --whisper-model MODEL  Whisper model (tiny|base|small|medium|large)")
    print()
    print("Network Settings:")
    print("  --host HOST            Server host address")
    print("  --video-port PORT      Video streaming port")
    print("  --audio-port PORT      Audio streaming port")
    print()
    print("Configuration:")
    print("  --config FILE          Use specific config file")
    print("  --save-config          Save current settings to config file")
    print()
    print("Help:")
    print("  --help                 Show detailed help")
    print()

def main():
    examples = {
        '1': {
            'name': 'Default Configuration',
            'description': 'All features enabled - NSFW detection, gun detection, transcription, and profanity filtering',
            'receiver_args': []
        },
        '2': {
            'name': 'Privacy Mode',
            'description': 'No content detection, only basic video streaming with transcription',
            'receiver_args': ['--no-nsfw', '--no-gun', '--enable-transcription', '--enable-profanity']
        },
        '3': {
            'name': 'Security Mode', 
            'description': 'Content detection only - NSFW and gun detection without transcription',
            'receiver_args': ['--enable-nsfw', '--enable-gun', '--no-transcription', '--no-profanity']
        },
        '4': {
            'name': 'Transcription Only',
            'description': 'Audio transcription and profanity filtering only, no visual content detection',
            'receiver_args': ['--no-nsfw', '--no-gun', '--enable-transcription', '--enable-profanity']
        },
        '5': {
            'name': 'Minimal Configuration',
            'description': 'Just video streaming, all detection features disabled for maximum performance',
            'receiver_args': ['--no-nsfw', '--no-gun', '--no-transcription', '--no-profanity']
        },
        '6': {
            'name': 'High Quality Transcription',
            'description': 'Better transcription quality using larger Whisper model (slower)',
            'receiver_args': ['--whisper-model', 'base', '--no-nsfw', '--no-gun']
        },
        '7': {
            'name': 'Custom Network Settings',
            'description': 'Example with custom host and ports',
            'receiver_args': ['--host', '192.168.1.100', '--video-port', '8080', '--audio-port', '8081']
        },
        '8': {
            'name': 'Performance Optimized',
            'description': 'Fastest settings - tiny model, reduced detection frequency',
            'receiver_args': ['--whisper-model', 'tiny', '--no-gun']
        }
    }
    
    while True:
        show_menu()
        choice = input("Enter your choice (0-9): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '9':
            show_command_reference()
            input("\\nPress Enter to continue...")
        elif choice in examples:
            example = examples[choice]
            run_example(
                example['name'],
                example['description'],
                receiver_args=example['receiver_args']
            )
        else:
            print("Invalid choice. Please try again.\\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nExiting...")
        sys.exit(0)
