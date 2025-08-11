#!/usr/bin/env python3
"""
Test script to verify NSFW detection functionality.
"""

import os
import cv2
import numpy as np
from huggingface_hub import InferenceClient
from PIL import Image
import io

# Hugging Face token must be provided via environment variable `HF_TOKEN`.
# Do not hardcode secrets in source code. If missing, the test will fail gracefully.
HF_TOKEN = os.getenv("HF_TOKEN")

def test_nsfw_detection():
    """Test NSFW detection with a simple test image"""
    print("Testing NSFW detection...")
    
    try:
        # Initialize Hugging Face client
        if not HF_TOKEN:
            raise RuntimeError(
                "Missing HF_TOKEN environment variable. Set your Hugging Face token securely, e.g.: export HF_TOKEN=..."
            )

        client = InferenceClient(
            model="Falconsai/nsfw_image_detection",
            token=HF_TOKEN,
        )
        print("✓ Hugging Face client initialized")
        
        # Create a simple test image (SFW content)
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to PIL Image and then to bytes
        test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(test_image_rgb)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Test NSFW detection
        print("Calling NSFW detection API...")
        output = client.post(data=img_byte_arr)
        
        print("✓ API call successful!")
        print("Results:")
        for result in output:
            print(f"  - {result['label']}: {result['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ NSFW detection test failed: {e}")
        return False

def main():
    """Run NSFW detection test"""
    print("=== NSFW Detection Test ===\n")
    
    if test_nsfw_detection():
        print("\n✓ NSFW detection is working correctly!")
        print("You can now run the video streaming system with NSFW detection:")
        print("1. python sender.py")
        print("2. python receiver.py (in another terminal)")
    else:
        print("\n✗ NSFW detection test failed.")
        print("The video streaming will still work, but without NSFW detection.")

if __name__ == "__main__":
    main() 