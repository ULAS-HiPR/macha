#!/usr/bin/env python3
"""
Test script for mock camera functionality on MacBook/development systems.
"""

import sys
import os
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_opencv_camera():
    """Test if OpenCV can access the MacBook camera."""
    print("Testing OpenCV camera access...")
    try:
        import cv2
        
        # Try to open camera 0 (usually built-in camera)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera 0 not accessible")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ Camera 0 accessible!")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Try to capture a frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully captured frame: {frame.shape}")
            
            # Save test image
            test_dir = Path("test_images")
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "opencv_test.jpg"
            cv2.imwrite(str(test_file), frame)
            print(f"✅ Test image saved to: {test_file}")
        else:
            print("❌ Failed to capture frame")
        
        cap.release()
        return True
        
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_pil_generation():
    """Test PIL image generation capabilities."""
    print("\nTesting PIL synthetic image generation...")
    try:
        from PIL import Image, ImageDraw, ImageFont
        import random
        from datetime import datetime
        
        # Create test image
        img = Image.new('RGB', (640, 480), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        
        # Add some shapes
        for i in range(3):
            x1, y1 = random.randint(0, 320), random.randint(0, 240)
            x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        
        # Add text
        text = f"Test Image\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        draw.text((10, 10), text, fill=(255, 255, 255))
        
        # Save test image
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "pil_test.jpg"
        img.save(test_file, quality=95)
        
        print(f"✅ PIL synthetic image created: {test_file}")
        return True
        
    except ImportError as e:
        print(f"❌ PIL not available: {e}")
        return False
    except Exception as e:
        print(f"❌ PIL test failed: {e}")
        return False

def test_mock_camera_task():
    """Test the mock camera task functionality."""
    print("\nTesting MockCameraTask...")
    try:
        from config import load_config
        from mock_camera_task import MockCameraTask
        import asyncio
        from database import DatabaseManager
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("test")
        
        # Load config
        config = load_config()
        
        # Create mock camera task
        mock_task = MockCameraTask(config, mock_strategy="auto")
        print(f"✅ MockCameraTask created with strategy: {mock_task.mock_strategy}")
        
        # Set up database
        db_manager = DatabaseManager(config.db.connection_string)
        
        async def run_test():
            await db_manager.initialize()
            engine = db_manager.engine
            
            # Execute mock capture
            result = await mock_task.execute(engine, logger)
            
            print(f"✅ Mock capture result:")
            print(f"   Captured: {result['captured']}")
            print(f"   Failed: {result['failed']}")
            print(f"   Strategy: {result['mock_strategy']}")
            
            if result['images']:
                for img_info in result['images']:
                    print(f"   📸 {img_info['camera']}: {img_info['file']} ({img_info['size']} bytes)")
        
        # Run the async test
        asyncio.run(run_test())
        
        print("✅ MockCameraTask test completed!")
        return True
        
    except Exception as e:
        print(f"❌ MockCameraTask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all camera tests."""
    print("🔍 Testing Camera Options for MacBook/Development")
    print("=" * 50)
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print()
    
    results = []
    
    # Test 1: OpenCV Camera Access
    results.append(("OpenCV Camera", test_opencv_camera()))
    
    # Test 2: PIL Synthetic Generation
    results.append(("PIL Generation", test_pil_generation()))
    
    # Test 3: Mock Camera Task
    results.append(("Mock Camera Task", test_mock_camera_task()))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Your system is ready for mock camera development.")
        print("\nNext steps:")
        print("1. Update your config.yaml to use MockCameraTask")
        print("2. Install missing dependencies if needed:")
        print("   pip install opencv-python pillow")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nTo install missing dependencies:")
        print("   pip install opencv-python pillow")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 