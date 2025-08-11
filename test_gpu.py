#!/usr/bin/env python3
"""
Test script to verify GPU detection and model acceleration
"""

import sys
import torch
from gpu_utils import detect_best_device, log_gpu_info, optimize_whisper_settings, optimize_transformers_device

def test_pytorch():
    """Test PyTorch GPU availability"""
    print("=== PyTorch GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Simple tensor test
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    else:
        print("No CUDA GPUs available")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ Apple Silicon MPS available")
        try:
            x = torch.randn(3, 3).to('mps')
            y = torch.randn(3, 3).to('mps')
            z = torch.mm(x, y)
            print("✓ MPS tensor operations working")
        except Exception as e:
            print(f"✗ MPS tensor operations failed: {e}")

def test_transformers():
    """Test Transformers with GPU"""
    print("\n=== Transformers GPU Test ===")
    try:
        from transformers import pipeline
        
        device_info = detect_best_device()
        device = optimize_transformers_device(device_info)
        
        print(f"Testing with device: {device}")
        
        if device != 'cpu':
            # Test a simple text classification pipeline
            classifier = pipeline("sentiment-analysis", device=device)
            result = classifier("This is a test sentence")
            print(f"✓ Transformers GPU pipeline working: {result}")
        else:
            print("Using CPU for transformers")
            
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")

def test_faster_whisper():
    """Test Faster Whisper with GPU"""
    print("\n=== Faster Whisper GPU Test ===")
    try:
        from faster_whisper import WhisperModel
        
        device_info = detect_best_device()
        device, compute_type = optimize_whisper_settings(device_info)
        
        print(f"Testing Whisper with device: {device}, compute_type: {compute_type}")
        
        # Test loading tiny model
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        print(f"✓ Whisper model loaded successfully on {device}")
        
        # Clean up
        del model
        
    except Exception as e:
        print(f"✗ Faster Whisper test failed: {e}")

def main():
    print("GPU Acceleration Test Script")
    print("=" * 40)
    
    # Test device detection
    print("\n=== Device Detection ===")
    device_info = detect_best_device()
    print(f"Best device detected: {device_info}")
    
    # Log detailed GPU info
    print("\n=== Detailed GPU Information ===")
    log_gpu_info()
    
    # Test PyTorch
    test_pytorch()
    
    # Test Transformers
    test_transformers()
    
    # Test Faster Whisper
    test_faster_whisper()
    
    print("\n=== Summary ===")
    if device_info['device'] != 'cpu':
        print(f"✓ GPU acceleration available: {device_info['description']}")
        print("Your code will automatically use GPU for model inference!")
    else:
        print("GPU not available, using CPU")
        print("To enable GPU acceleration:")
        print("1. Install NVIDIA drivers")
        print("2. Install CUDA toolkit")
        print("3. Reinstall PyTorch with CUDA support")

if __name__ == "__main__":
    main()
