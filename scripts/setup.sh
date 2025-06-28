#!/bin/bash

# Macha  Setup Script
# Comprehensive setup for Raspberry Pi with dual camera support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up Macha ..."
echo "============================================="

# Check if running on Raspberry Pi
if [ ! -f /proc/cpuinfo ] || ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "Warning: This script is designed for Raspberry Pi systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running from correct directory
if [ ! -f "$PROJECT_DIR/pyproject.toml" ] || [ ! -f "$PROJECT_DIR/config.yaml" ]; then
    echo "Error: This script must be run from the macha project directory"
    exit 1
fi

# Set locale
echo "Setting up locale..."
if ! grep -q "export LANG=en_US.UTF-8" ~/.bashrc; then
    echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
fi
if ! grep -q "export LC_ALL=en_US.UTF-8" ~/.bashrc; then
    echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
fi

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    libcap-dev \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    libcamera-tools \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    ffmpeg \
    python3-pycoral

# Add user to required groups
echo "Adding user to video and gpio groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G gpio $USER

# Set camera permissions
echo "Setting up camera permissions..."
sudo bash -c 'cat > /etc/udev/rules.d/99-camera.rules << EOF
SUBSYSTEM=="vchiq", GROUP="video", MODE="0666"
SUBSYSTEM=="bcm2835-gpiomem", GROUP="gpio", MODE="0660"
KERNEL=="vchiq", GROUP="video", MODE="0666"
EOF'

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Enable camera in boot config
echo "Enabling camera in boot config..."
BOOT_CONFIG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CONFIG" ]; then
    BOOT_CONFIG="/boot/config.txt"
fi

if [ -f "$BOOT_CONFIG" ]; then
    if ! grep -q "camera_auto_detect=1" "$BOOT_CONFIG"; then
        echo "camera_auto_detect=1" | sudo tee -a "$BOOT_CONFIG"
        echo "Added camera_auto_detect=1 to boot config"
    fi

    if ! grep -q "dtoverlay=camera" "$BOOT_CONFIG"; then
        echo "dtoverlay=camera" | sudo tee -a "$BOOT_CONFIG"
        echo "Added dtoverlay=camera to boot config"
    fi
else
    echo "Warning: Boot config file not found"
fi

# Change to project directory
cd "$PROJECT_DIR"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment with system site-packages access
echo "Creating virtual environment with system site-packages access..."
python3 -m venv .venv --system-site-packages

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install pyyaml>=6.0.2 colorlog>=6.8.2 sqlalchemy>=2.0.0 aiosqlite>=0.20.0 psutil>=6.1.0 pydantic>=2.0.0


# Test camera setup
echo "Testing camera hardware..."
if command -v libcamera-hello &> /dev/null; then
    libcamera-hello --list-cameras | head -n 10
else
    echo "Warning: libcamera-hello not available"
fi

# Test Python camera access
echo "Testing Python camera access..."
python -c "
try:
    from picamera2 import Picamera2
    cameras = Picamera2.global_camera_info()
    print(f'✓ picamera2 working - {len(cameras)} cameras detected')
    for i, cam in enumerate(cameras):
        print(f'  Camera {i}: {cam[\"Model\"]}')

    # Quick test of camera initialization
    if cameras:
        print('Testing camera initialization...')
        picam = Picamera2(camera_num=0)
        picam.start()
        print('✓ Camera 0 started successfully')
        picam.stop()
        picam.close()
        print('✓ Camera 0 closed successfully')

except Exception as e:
    print(f'✗ Camera test failed: {e}')
    print('This may be resolved after a reboot.')
"

# Test configuration loading
echo "Testing Macha configuration..."
python -c "
try:
    from src.config import load_config
    config = load_config()
    print(f'✓ Configuration valid with {len(config.tasks)} tasks')

    camera_tasks = [t for t in config.tasks if t.class_name == 'CameraTask']
    if camera_tasks:
        print(f'✓ Found {len(camera_tasks)} camera task(s)')
        for task in camera_tasks:
            print(f'  - {task.name}: {len(task.parameters.cameras)} cameras configured')

except Exception as e:
    print(f'✗ Configuration test failed: {e}')
    exit(1)
"

echo ""
echo "Setup completed successfully!"
echo ""
echo "IMPORTANT: Reboot required for all changes to take effect:"
echo "  sudo reboot"
echo ""
echo "Note: You may need to log out and back in for group changes to take effect."
