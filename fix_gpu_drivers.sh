#!/usr/bin/env bash
set -euo pipefail

echo "=== NVIDIA GPU Driver Repair (Jammy 22.04, HWE kernel) ==="
echo "Kernel: $(uname -r)"

# 0) Secure Boot check (NVIDIA kernel module won't load if Secure Boot is enabled and unsigned)
if command -v mokutil >/dev/null 2>&1; then
  echo "--- Secure Boot status ---"
  mokutil --sb-state || true
else
  echo "mokutil not installed (cannot check Secure Boot status)"
fi

# 1) Ensure repositories are complete (restricted/universe/multiverse)
echo "--- Enabling required repositories (restricted, universe, multiverse) ---"
sudo add-apt-repository -y restricted || true
sudo add-apt-repository -y universe || true
sudo add-apt-repository -y multiverse || true

# 2) Basic tooling and headers
echo "--- Installing build tools and headers ---"
sudo apt-get update -y
sudo apt-get install -y dkms build-essential linux-headers-"$(uname -r)" ubuntu-drivers-common ppa-purge

# 3) If graphics-drivers PPA exists, purge it to avoid version skew
if grep -R "graphics-drivers/ppa" /etc/apt/sources.list.d 2>/dev/null | grep -q ppa; then
  echo "--- Purging graphics-drivers PPA to return to Ubuntu official packages ---"
  sudo ppa-purge -y ppa:graphics-drivers/ppa || true
  sudo apt-get update -y
fi

# 4) Remove all existing NVIDIA packages (clean slate)
echo "--- Purging existing NVIDIA packages ---"
sudo apt-get remove --purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' 'libcuda*' 'xserver-xorg-video-nvidia*' || true
sudo apt-get autoremove -y || true
sudo apt-get autoclean -y || true

# 5) Install a consistent driver set using DKMS (avoids linux-modules-nvidia version mismatch)
echo "--- Installing NVIDIA driver (DKMS-based) ---"
sudo apt-get update -y
# Let Ubuntu choose the best tested driver for this release
if ubuntu-drivers devices 2>/dev/null | grep -q recommended; then
  echo "Using ubuntu-drivers autoinstall (recommended)"
  sudo ubuntu-drivers autoinstall -g || sudo ubuntu-drivers autoinstall || true
fi

# Also explicitly install the 550 DKMS stack if autoinstall did not install anything
if ! dpkg -l | grep -q "nvidia-driver-"; then
  echo "Falling back to explicit 550 install via DKMS"
  sudo apt-get install -y nvidia-driver-550 nvidia-dkms-550
fi

# 6) Rebuild modules via DKMS (should build for current kernel)
echo "--- DKMS build ---"
sudo dkms autoinstall || true
sudo dkms status || true

# 7) Load the module now (may fail if Secure Boot is enabled) and report
echo "--- Loading module ---"
sudo modprobe nvidia || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi ---"
  nvidia-smi || true
else
  echo "nvidia-smi not found yet"
fi

echo "--- lsmod (nvidia) ---"
lsmod | grep -i nvidia || echo "nvidia kernel module not loaded"

cat << 'EONOTE'

Next steps if still not working:
1) Secure Boot: If Secure Boot is enabled, either disable it in BIOS or enroll the DKMS module signature (MOK) on next reboot.
2) Reboot: A reboot is often required after a clean driver reinstall. Run: sudo reboot
3) After reboot, validate:
   - nvidia-smi
   - python test_gpu.py

EONOTE
