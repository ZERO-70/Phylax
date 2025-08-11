#!/usr/bin/env python3
"""
Test script to verify the video streaming setup.
This script checks dependencies and creates a test video if needed.
"""

import os
import sys
import cv2
import numpy as np

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import cv2
        print("✓ OpenCV installed")
    except ImportError:
        print("✗ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        import socket
        print("✓ Socket module available")
    except ImportError:
        print("✗ Socket module not available")
        return False
    
    try:
        import pickle
        print("✓ Pickle module available")
    except ImportError:
        print("✗ Pickle module not available")
        return False
    
    try:
        import struct
        print("✓ Struct module available")
    except ImportError:
        print("✗ Struct module not available")
        return False
    
    return True

def create_test_video():
    """Create a simple test video if video.mp4 doesn't exist"""
    if os.path.exists('video.mp4'):
        print("✓ Test video (video.mp4) already exists")
        return True
    
    print("Creating test video...")
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 10  # 10 seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("✗ Could not create video file")
        return False
    
    # Create frames
    for i in range(fps * duration):
        # Create a frame with moving text
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some color
        frame[:, :, 0] = (i * 2) % 255  # Blue channel
        frame[:, :, 1] = (i * 3) % 255  # Green channel
        frame[:, :, 2] = (i * 4) % 255  # Red channel
        
        # Add text
        text = f"Test Video - Frame {i}"
        cv2.putText(frame, text, (50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = f"Time: {i/fps:.1f}s"
        cv2.putText(frame, timestamp, (50, height//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✓ Test video created successfully")
    return True

def test_opencv_display():
    """Test if OpenCV can display windows"""
    print("Testing OpenCV display...")
    
    # Create a simple test image
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(test_image, "OpenCV Test", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Try to display the image
    try:
        cv2.imshow('Test Window', test_image)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()
        print("✓ OpenCV display test passed")
        return True
    except Exception as e:
        print(f"✗ OpenCV display test failed: {e}")
        print("Note: This might be due to missing display server on headless systems")
        return False

def main():
    """Run all tests"""
    print("=== Video Streaming Setup Test ===\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    print("\n✓ All dependencies are installed!")
    
    # Create test video
    if not create_test_video():
        print("\n✗ Could not create test video.")
        sys.exit(1)
    
    # Test OpenCV display
    test_opencv_display()
    
    print("\n=== Setup Complete ===")
    print("You can now run the video streaming system:")
    print("1. python sender.py")
    print("2. python receiver.py (in another terminal)")
    print("\nNote: If you're on a headless system, the display test may fail,")
    print("but the streaming should still work if you have a video.mp4 file.")

if __name__ == "__main__":
    main() 