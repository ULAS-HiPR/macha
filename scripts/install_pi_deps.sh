#!/bin/bash

# Pi Hardware Dependencies Installation Script
# This script installs Raspberry Pi specific dependencies for macha

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Macha Pi Hardware Dependencies Installation ===${NC}"

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null && ! grep -q "BCM" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    echo "This script is designed for Raspberry Pi hardware"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled"
        exit 1
    fi
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo -e "${GREEN}Installing system dependencies...${NC}"

# Update package list
sudo apt-get update

# Install system dependencies for I2C and GPIO
echo "Installing I2C and GPIO libraries..."
sudo apt-get install -y \
    python3-dev \
    libcap-dev \
    i2c-tools \
    python3-smbus \
    libgpiod-dev \
    python3-lgpio \
    python3-rpi.gpio

# Install camera dependencies
echo "Installing camera dependencies..."
sudo apt-get install -y \
    python3-picamera2 \
    libcamera-apps \
    libcamera-dev

# Enable I2C interface
echo -e "${GREEN}Configuring I2C interface...${NC}"
if ! grep -q "dtparam=i2c_arm=on" /boot/firmware/config.txt; then
    echo "Enabling I2C interface..."
    echo "dtparam=i2c_arm=on" | sudo tee -a /boot/firmware/config.txt
    I2C_ENABLED=true
fi

# Add user to i2c group
if ! groups $USER | grep -q i2c; then
    echo "Adding user $USER to i2c group..."
    sudo usermod -a -G i2c $USER
    GROUP_CHANGED=true
fi

# Add user to gpio group
if ! groups $USER | grep -q gpio; then
    echo "Adding user $USER to gpio group..."
    sudo usermod -a -G gpio $USER
    GROUP_CHANGED=true
fi

echo -e "${GREEN}Installing Python hardware dependencies...${NC}"

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Install Pi hardware dependencies using uv
echo "Installing Pi-specific Python packages..."
uv sync --extra pi-hardware

echo -e "${GREEN}Testing I2C interface...${NC}"

# Test I2C interface
if command -v i2cdetect &> /dev/null; then
    echo "I2C devices detected:"
    i2cdetect -y 1 || echo "No I2C devices found (this is normal if sensors aren't connected)"
else
    echo -e "${YELLOW}Warning: i2cdetect not available${NC}"
fi

echo -e "${GREEN}Testing camera interface...${NC}"

# Test camera (non-blocking)
if command -v libcamera-hello &> /dev/null; then
    echo "Testing camera (5 second test)..."
    timeout 5s libcamera-hello --nopreview --timeout 1000 || echo "Camera test completed (may have timed out normally)"
else
    echo -e "${YELLOW}Warning: libcamera-hello not available${NC}"
fi

echo -e "${GREEN}Creating hardware test script...${NC}"

# Create hardware test script
cat > "$PROJECT_DIR/scripts/test_hardware.py" << 'EOF'
#!/usr/bin/env python3
"""
Hardware test script for macha sensors
Tests I2C connectivity and sensor availability
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_i2c():
    """Test I2C bus availability."""
    print("Testing I2C bus...")
    try:
        import board
        import busio
        i2c = busio.I2C(board.SCL, board.SDA)
        print("✓ I2C bus initialized successfully")
        i2c.deinit()
        return True
    except Exception as e:
        print(f"✗ I2C bus test failed: {e}")
        return False

def test_bmp390():
    """Test BMP390 barometer sensor."""
    print("Testing BMP390 barometer...")
    try:
        import board
        import busio
        import adafruit_bmp3xx
        
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bmp3xx.BMP3XX_I2C(i2c, address=0x77)
        
        pressure = sensor.pressure
        temperature = sensor.temperature
        
        print(f"✓ BMP390 connected: {pressure:.2f} hPa, {temperature:.2f}°C")
        i2c.deinit()
        return True
    except Exception as e:
        print(f"✗ BMP390 test failed: {e}")
        try:
            # Try alternative address
            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_bmp3xx.BMP3XX_I2C(i2c, address=0x76)
            pressure = sensor.pressure
            temperature = sensor.temperature
            print(f"✓ BMP390 connected at 0x76: {pressure:.2f} hPa, {temperature:.2f}°C")
            i2c.deinit()
            return True
        except Exception as e2:
            print(f"✗ BMP390 test failed at both addresses: {e2}")
            return False

def test_lsm6dsox():
    """Test LSM6DSOX IMU sensor."""
    print("Testing LSM6DSOX IMU...")
    try:
        import board
        import busio
        import adafruit_lsm6ds
        
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_lsm6ds.LSM6DSOX(i2c, address=0x6A)
        
        accel = sensor.acceleration
        gyro = sensor.gyro
        temp = sensor.temperature
        
        print(f"✓ LSM6DSOX connected: accel=({accel[0]:.2f},{accel[1]:.2f},{accel[2]:.2f}), temp={temp:.2f}°C")
        i2c.deinit()
        return True
    except Exception as e:
        print(f"✗ LSM6DSOX test failed: {e}")
        try:
            # Try alternative address
            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_lsm6ds.LSM6DSOX(i2c, address=0x6B)
            accel = sensor.acceleration
            gyro = sensor.gyro
            temp = sensor.temperature
            print(f"✓ LSM6DSOX connected at 0x6B: accel=({accel[0]:.2f},{accel[1]:.2f},{accel[2]:.2f}), temp={temp:.2f}°C")
            i2c.deinit()
            return True
        except Exception as e2:
            print(f"✗ LSM6DSOX test failed at both addresses: {e2}")
            return False

def test_camera():
    """Test camera functionality."""
    print("Testing camera...")
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        picam2.stop()
        
        print("✓ Camera initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all hardware tests."""
    print("=== Macha Hardware Test ===")
    print()
    
    tests = [
        ("I2C Bus", test_i2c),
        ("BMP390 Barometer", test_bmp390),
        ("LSM6DSOX IMU", test_lsm6dsox),
        ("Camera", test_camera),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"--- {name} ---")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
        print()
    
    print("=== Test Summary ===")
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All hardware tests passed!")
        return 0
    else:
        print("\n⚠️  Some hardware tests failed. Check connections and configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x "$PROJECT_DIR/scripts/test_hardware.py"

echo -e "${GREEN}Installation complete!${NC}"
echo
echo "=== Next Steps ==="

if [[ "$I2C_ENABLED" == "true" ]] || [[ "$GROUP_CHANGED" == "true" ]]; then
    echo -e "${YELLOW}⚠️  REBOOT REQUIRED${NC}"
    echo "Some changes require a system reboot to take effect:"
    if [[ "$I2C_ENABLED" == "true" ]]; then
        echo "  - I2C interface enabled"
    fi
    if [[ "$GROUP_CHANGED" == "true" ]]; then
        echo "  - User groups updated"
    fi
    echo
    echo "Run: sudo reboot"
    echo
fi

echo "After reboot, test your hardware:"
echo "  cd $PROJECT_DIR"
echo "  python3 scripts/test_hardware.py"
echo
echo "To run macha with hardware support:"
echo "  uv run src/main.py"
echo
echo "For development without Pi hardware:"
echo "  uv sync --extra dev"