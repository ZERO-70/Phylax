#!/usr/bin/env python3
"""
Utility script to check and manage the NSFW detection model cache.
This script helps you understand where models are cached and manage the cache.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Set up logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Use stdout for better encoding
        logging.FileHandler('cache_check.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def get_cache_directories():
    """Get the cache directories used by transformers"""
    cache_dirs = []
    
    # Check common cache locations
    possible_paths = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.path.expanduser("~/AppData/Local/huggingface"),  # Windows
        os.path.expanduser("~/Library/Caches/huggingface"),  # macOS
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            cache_dirs.append(path)
            logger.info(f"Found cache directory: {path}")
    
    return cache_dirs

def check_model_cache():
    """Check if the NSFW model is cached"""
    logger.info("=== Checking NSFW Model Cache ===")
    
    cache_dirs = get_cache_directories()
    
    if not cache_dirs:
        logger.warning("No cache directories found!")
        return
    
    model_name = "Falconsai/nsfw_image_detection"
    model_found = False
    
    for cache_dir in cache_dirs:
        logger.info(f"Searching in: {cache_dir}")
        
        # Look for the model in the cache
        for root, dirs, files in os.walk(cache_dir):
            if model_name.replace("/", "_") in root or "nsfw" in root.lower():
                logger.info(f"Found potential model cache: {root}")
                
                # Check for model files
                model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.json'))]
                if model_files:
                    logger.info(f"Model files found: {len(model_files)} files")
                    model_found = True
                    
                    # Show file sizes
                    total_size = 0
                    for file in model_files:
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        total_size += size
                        logger.info(f"  - {file}: {size / (1024*1024):.1f} MB")
                    
                    logger.info(f"Total model size: {total_size / (1024*1024):.1f} MB")
                    break
    
    if not model_found:
        logger.info("Model not found in cache. It will be downloaded on first use.")
    
    return model_found

def clear_model_cache():
    """Clear the NSFW model cache"""
    logger.info("=== Clearing NSFW Model Cache ===")
    
    cache_dirs = get_cache_directories()
    
    if not cache_dirs:
        logger.warning("No cache directories found!")
        return
    
    model_name = "Falconsai/nsfw_image_detection"
    cleared = False
    
    for cache_dir in cache_dirs:
        logger.info(f"Searching in: {cache_dir}")
        
        # Look for the model in the cache
        for root, dirs, files in os.walk(cache_dir):
            if model_name.replace("/", "_") in root or "nsfw" in root.lower():
                logger.info(f"Found model cache to clear: {root}")
                
                try:
                    shutil.rmtree(root)
                    logger.info(f"Cleared cache directory: {root}")
                    cleared = True
                except Exception as e:
                    logger.error(f"Error clearing cache: {e}")
    
    if not cleared:
        logger.info("No model cache found to clear.")
    else:
        logger.info("Model cache cleared successfully!")

def show_cache_info():
    """Show detailed cache information"""
    logger.info("=== Cache Information ===")
    
    cache_dirs = get_cache_directories()
    
    if not cache_dirs:
        logger.warning("No cache directories found!")
        return
    
    total_size = 0
    
    for cache_dir in cache_dirs:
        logger.info(f"\nCache directory: {cache_dir}")
        
        try:
            for root, dirs, files in os.walk(cache_dir):
                dir_size = sum(os.path.getsize(os.path.join(root, file)) for file in files)
                total_size += dir_size
                
                if dir_size > 1024*1024:  # Show directories larger than 1MB
                    logger.info(f"  {root}: {dir_size / (1024*1024):.1f} MB")
        except Exception as e:
            logger.error(f"Error reading cache directory: {e}")
    
    logger.info(f"\nTotal cache size: {total_size / (1024*1024):.1f} MB")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            check_model_cache()
        elif command == "clear":
            clear_model_cache()
        elif command == "info":
            show_cache_info()
        else:
            print("Usage: python check_model_cache.py [check|clear|info]")
            print("  check - Check if NSFW model is cached")
            print("  clear - Clear the NSFW model cache")
            print("  info  - Show detailed cache information")
    else:
        # Default: check cache
        check_model_cache()
        print("\nUse 'python check_model_cache.py clear' to clear the cache")
        print("Use 'python check_model_cache.py info' to see cache details")

if __name__ == "__main__":
    main() 