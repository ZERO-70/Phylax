#!/bin/bash
"""
Post-reboot GPU setup script
Run this after rebooting to kernel 6.8.0-65 to enable GPU acceleration
"""

echo "=== GPU Setup Script ==="
echo "Setting up NVIDIA GPU support for abuse detection system"

# Check current kernel
CURRENT_KERNEL=$(uname -r)
echo "Current kernel: $CURRENT_KERNEL"

# Check if we're on the right kernel
if [[ "$CURRENT_KERNEL" != "6.8.0-65-generic" ]]; then
    echo "WARNING: You're not on kernel 6.8.0-65-generic"
    echo "Please reboot and select the latest kernel from GRUB menu"
    echo "Current kernel: $CURRENT_KERNEL"
    read -p "Continue anyway? (y/N): " -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update package list
echo "Updating package list..."
sudo apt update

# Upgrade NVIDIA drivers to latest version
echo "Upgrading NVIDIA drivers..."
sudo apt upgrade -y nvidia-driver-550

# Install NVIDIA modules for current kernel
echo "Installing NVIDIA modules for kernel $CURRENT_KERNEL..."
sudo apt install -y linux-modules-nvidia-550-$CURRENT_KERNEL

# Update the generic package to track latest modules
echo "Updating NVIDIA modules meta-package..."
sudo apt upgrade -y linux-modules-nvidia-550-generic-hwe-22.04

# Load NVIDIA modules
echo "Loading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm

# Check if NVIDIA is working
echo "Testing NVIDIA driver..."
if nvidia-smi; then
    echo "✓ NVIDIA GPU is working!"
else
    echo "✗ NVIDIA GPU not working. You may need to reboot again."
fi

# Test PyTorch CUDA
echo "Testing PyTorch CUDA support..."
cd /home/umar/umarsulemanlinux/work/abuse_detection
source enviroment/bin/activate
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== GPU Setup Complete ==="
echo "Your abuse detection system is now configured to use GPU acceleration!"
echo "You can run: python receiver.py --device auto"
echo "Or test GPU detection with: python test_gpu.py"
