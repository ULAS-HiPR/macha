#!/bin/bash

# Development Setup Script for MacBook/Non-Pi Systems
# Installs dependencies for mock camera functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🖥️  Setting up Macha for Development (MacBook/Non-Pi)"
echo "============================================="

# Check if running from correct directory
if [ ! -f "$PROJECT_DIR/pyproject.toml" ] || [ ! -f "$PROJECT_DIR/config.yaml" ]; then
    echo "❌ Error: This script must be run from the macha project directory"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

echo "Installing development dependencies..."

# Install Python dependencies for mock camera functionality
echo "📦 Installing mock camera dependencies..."
pip install opencv-python pillow numpy

# Install core dependencies if not already installed
echo "📦 Installing core dependencies..."
pip install pyyaml>=6.0.2 colorlog>=6.8.2 sqlalchemy>=2.0.0 aiosqlite>=0.20.0 psutil>=6.1.0 pydantic>=2.0.0

# Test installations
echo "🧪 Testing installations..."

# Test OpenCV
python3 -c "
import cv2
print('✅ OpenCV installed successfully')
print(f'   Version: {cv2.__version__}')
"

# Test PIL/Pillow
python3 -c "
from PIL import Image, ImageDraw, ImageFont
print('✅ PIL/Pillow installed successfully')
print(f'   Version: {Image.__version__}')
"

# Test NumPy
python3 -c "
import numpy as np
print('✅ NumPy installed successfully')
print(f'   Version: {np.__version__}')
"

# Create dev directories
echo "📁 Creating development directories..."
mkdir -p dev_images/cam0
mkdir -p dev_images/cam1
mkdir -p mock_assets
mkdir -p test_images
mkdir -p logs

echo "🔍 Testing camera access..."

# Check if camera is available
python3 -c "
import cv2
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('✅ Camera 0 (built-in) is accessible')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'   Default resolution: {width}x{height}')
        cap.release()
    else:
        print('⚠️  Camera 0 not accessible (may need permission)')
except Exception as e:
    print(f'⚠️  Camera test failed: {e}')
"

# Test mock camera functionality
echo "🧪 Testing mock camera task..."
python3 tests/test_mock_camera.py

echo ""
echo "✅ Development setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Use the development config: python3 src/main.py --config config_dev.yaml"
echo "2. Or test mock camera directly: python3 tests/test_mock_camera.py"
echo "3. Check camera permissions if OpenCV camera fails"
echo ""
echo "🎯 Development features enabled:"
echo "   • OpenCV camera access (MacBook built-in)"
echo "   • Synthetic image generation"
echo "   • Static test image cycling"
echo "   • Automatic fallback strategies"
echo ""
echo "💡 Tips:"
echo "   • Grant camera permissions to Terminal if prompted"
echo "   • Use config_dev.yaml for development"
echo "   • Hardware sensors are disabled in dev config"
echo "" 