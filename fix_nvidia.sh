#!/bin/bash
# NVIDIA Driver Fix Script
# Run this to resolve the kernel module version mismatch

echo "=== NVIDIA Driver Fix for Kernel Updates ==="
echo "Current kernel: $(uname -r)"
echo "Checking NVIDIA drivers..."

# Method 1: Try installing the specific kernel module
echo "Attempting to install NVIDIA modules for current kernel..."
sudo apt update
sudo apt install -y linux-modules-nvidia-550-$(uname -r) 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ NVIDIA modules installed successfully"
    echo "Loading NVIDIA modules..."
    sudo modprobe nvidia
    nvidia-smi
else
    echo "Module installation failed. Trying alternative methods..."
    
    # Method 2: Update all NVIDIA packages
    echo "Updating all NVIDIA packages..."
    sudo apt update
    sudo apt upgrade -y nvidia-*
    
    # Method 3: Reinstall NVIDIA driver
    echo "If issues persist, reinstalling NVIDIA driver..."
    echo "sudo apt remove --purge nvidia-* libnvidia-*"
    echo "sudo apt autoremove"
    echo "sudo apt install nvidia-driver-550"
    echo "sudo reboot"
    
    # Method 4: Use DKMS to rebuild modules
    echo "Or try rebuilding with DKMS:"
    echo "sudo dkms autoinstall"
fi

echo ""
echo "After successful installation, test with:"
echo "nvidia-smi"
echo "python test_gpu.py"
