#!/usr/bin/env python3
"""
GPU utility functions for automatic device detection and optimization
"""

import torch
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

def detect_best_device():
    """
    Detect the best available device for computation.
    Returns tuple: (device_string, device_type, compute_type)
    
    Priority:
    1. CUDA GPU (if available)
    2. MPS (Apple Silicon Mac)
    3. CPU
    """
    device_info = {
        'device': 'cpu',
        'device_type': 'cpu',
        'compute_type': 'int8',
        'description': 'CPU',
        'memory_gb': None
    }
    
    try:
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            
            # Get GPU memory
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            memory_gb = total_memory / (1024**3)
            
            device_info.update({
                'device': 'cuda',
                'device_type': 'gpu',
                'compute_type': 'float16' if memory_gb > 4 else 'int8',
                'description': f'CUDA GPU: {gpu_name} ({memory_gb:.1f}GB)',
                'memory_gb': memory_gb,
                'gpu_count': gpu_count
            })
            
            logger.info(f"CUDA GPU detected: {gpu_name}")
            logger.info(f"GPU Memory: {memory_gb:.1f}GB")
            logger.info(f"Available GPUs: {gpu_count}")
            
            return device_info
            
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_info.update({
                'device': 'mps',
                'device_type': 'gpu',
                'compute_type': 'float16',
                'description': 'Apple Silicon GPU (MPS)'
            })
            
            logger.info("Apple Silicon GPU (MPS) detected")
            return device_info
            
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")
    
    # Fallback to CPU
    logger.info("Using CPU for computation")
    return device_info

def optimize_whisper_settings(device_info):
    """
    Optimize Whisper model settings based on available device
    """
    device = device_info['device']
    compute_type = device_info['compute_type']
    
    # Optimize compute type based on device and memory
    if device == 'cuda':
        memory_gb = device_info.get('memory_gb', 0)
        if memory_gb >= 8:
            compute_type = 'float16'
        elif memory_gb >= 4:
            compute_type = 'int8'
        else:
            compute_type = 'int8'
    elif device == 'mps':
        compute_type = 'float16'
    else:
        compute_type = 'int8'
    
    logger.info(f"Optimized Whisper settings: device={device}, compute_type={compute_type}")
    return device, compute_type

def optimize_transformers_device(device_info):
    """
    Get optimal device for transformers models (HuggingFace)
    """
    if device_info['device'] in ['cuda', 'mps']:
        return device_info['device']
    return 'cpu'

def log_gpu_info():
    """
    Log detailed GPU information for debugging
    """
    try:
        if torch.cuda.is_available():
            logger.info("=== GPU Information ===")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {memory_gb:.1f}GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
                logger.info(f"  Multiprocessors: {props.multi_processor_count}")
                
                # Current memory usage
                if i == torch.cuda.current_device():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    logger.info(f"  Memory Usage: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
        else:
            logger.info("CUDA not available, using CPU")
            
        # Check for other acceleration
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS available")
            
    except Exception as e:
        logger.error(f"Error logging GPU info: {e}")

def clear_gpu_cache():
    """
    Clear GPU cache to free memory
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Error clearing GPU cache: {e}")

def get_optimal_batch_size(device_info, model_type='nsfw'):
    """
    Get optimal batch size based on available GPU memory
    """
    if device_info['device'] == 'cpu':
        return 1
    
    memory_gb = device_info.get('memory_gb', 0)
    
    if model_type == 'nsfw':
        if memory_gb >= 8:
            return 4
        elif memory_gb >= 4:
            return 2
        else:
            return 1
    elif model_type == 'whisper':
        # Whisper models process one audio chunk at a time
        return 1
    
    return 1

def monitor_gpu_usage():
    """
    Monitor GPU usage and log warnings if memory is high
    """
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            usage_percent = (allocated / total) * 100
            
            if usage_percent > 80:
                logger.warning(f"High GPU memory usage: {usage_percent:.1f}% ({allocated:.1f}GB/{total:.1f}GB)")
            elif usage_percent > 60:
                logger.info(f"GPU memory usage: {usage_percent:.1f}% ({allocated:.1f}GB/{total:.1f}GB)")
                
            return usage_percent
    except Exception as e:
        logger.warning(f"Error monitoring GPU usage: {e}")
    
    return 0

if __name__ == "__main__":
    # Test the GPU detection
    print("=== GPU Detection Test ===")
    device_info = detect_best_device()
    print(f"Best device: {device_info}")
    
    log_gpu_info()
    
    # Test optimizations
    whisper_device, whisper_compute = optimize_whisper_settings(device_info)
    transformers_device = optimize_transformers_device(device_info)
    
    print(f"\nOptimized settings:")
    print(f"  Whisper: device={whisper_device}, compute_type={whisper_compute}")
    print(f"  Transformers: device={transformers_device}")
    print(f"  Batch size (NSFW): {get_optimal_batch_size(device_info, 'nsfw')}")
