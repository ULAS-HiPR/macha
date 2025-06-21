import asyncio
import json
import logging
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from typing import List, Optional

from config import MachaConfig, CameraParameters
from task import Task

# Check for optional dependencies
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageDraw = None
    ImageFont = None


class MockCameraTask(Task):
    """
    Mock camera task for development and testing on non-Pi systems.

    Supports multiple mock strategies:
    1. OpenCV webcam (MacBook built-in camera)
    2. Static test images
    3. Generated synthetic images
    4. Cycling through image sets
    """

    def __init__(self, config: MachaConfig, mock_strategy: str = "auto"):
        super().__init__(config)

        # Find camera task config
        camera_params = None
        for task in config.tasks:
            if task.class_name in ["CameraTask", "MockCameraTask"] and task.parameters:
                if isinstance(task.parameters, CameraParameters):
                    camera_params = task.parameters
                    break

        if not camera_params:
            raise ValueError("No valid CameraTask or MockCameraTask configuration found")

        self.cameras = camera_params.cameras
        self.image_format = camera_params.image_format
        self.resolution = camera_params.resolution
        self.quality = camera_params.quality
        self.rotation = camera_params.rotation
        self.capture_timeout = camera_params.capture_timeout

        # Mock-specific settings
        self.mock_strategy = self._determine_strategy(mock_strategy)
        self.opencv_cameras = {}  # Cache for OpenCV cameras
        self.image_counter = 0

        # Create test image set
        self._prepare_mock_assets()

    def _determine_strategy(self, strategy: str) -> str:
        """Determine the best mock strategy based on system and preference."""
        if strategy == "auto":
            # Try to detect best strategy
            if platform.system() == "Darwin":  # macOS
                return "opencv"  # Try MacBook camera first
            elif self._has_opencv_camera():
                return "opencv"
            else:
                return "synthetic"  # Fallback to generated images
        return strategy

    def _has_opencv_camera(self) -> bool:
        """Check if OpenCV can access any camera."""
        if not OPENCV_AVAILABLE:
            return False
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                return True
        except Exception:
            pass
        return False

    def _prepare_mock_assets(self):
        """Prepare mock assets directory and sample images."""
        self.mock_assets_dir = Path("mock_assets")
        self.mock_assets_dir.mkdir(exist_ok=True)

        # Create some sample static images if they don't exist
        sample_images = ["test_landing_1.jpg", "test_landing_2.jpg", "test_aerial_view.jpg"]
        for img_name in sample_images:
            img_path = self.mock_assets_dir / img_name
            if not img_path.exists():
                self._create_sample_image(img_path, img_name)

    def _create_sample_image(self, path: Path, name: str):
        """Create a sample test image."""
        if not PIL_AVAILABLE:
            # Create a minimal placeholder file
            with open(path, 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01' + b'\x00' * 100 + b'\xff\xd9')
            return

        img = Image.new('RGB', (self.resolution.width, self.resolution.height),
                       color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
        draw = ImageDraw.Draw(img)

        # Add some shapes to make it recognizable
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw some geometric shapes to simulate terrain/features
        for i in range(5):
            x1, y1 = random.randint(0, img.width//2), random.randint(0, img.height//2)
            x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))

        # Add timestamp and camera info
        text = f"MOCK: {name}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if font:
            draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((10, 10), text, fill=(255, 255, 255))

        img.save(path, quality=self.quality)

    async def _capture_opencv(self, cam_config, filepath: str) -> bool:
        """Capture image using OpenCV (for MacBook camera, etc.)."""
        if not OPENCV_AVAILABLE:
            return False

        try:
            # Initialize camera if not cached
            if cam_config.port not in self.opencv_cameras:
                cap = cv2.VideoCapture(cam_config.port)
                if not cap.isOpened():
                    return False

                # Set resolution if possible
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution.height)

                self.opencv_cameras[cam_config.port] = cap

            cap = self.opencv_cameras[cam_config.port]

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                return False

            # Apply rotation if needed
            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Save image
            if self.image_format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:
                cv2.imwrite(filepath, frame)

            return True

        except Exception as e:
            print(f"OpenCV capture error: {e}")
            return False

    async def _capture_static(self, cam_config, filepath: str) -> bool:
        """Capture using static test images."""
        if not PIL_AVAILABLE:
            # Create a minimal placeholder file
            with open(filepath, 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01' + b'\x00' * 100 + b'\xff\xd9')
            self.image_counter += 1
            return True

        try:
            # Get list of available mock images
            mock_images = list(self.mock_assets_dir.glob("*.jpg"))
            if not mock_images:
                return False

            # Cycle through images based on counter
            source_img = mock_images[self.image_counter % len(mock_images)]

            # Copy and potentially modify the image
            img = Image.open(source_img)

            # Resize if needed
            if img.size != (self.resolution.width, self.resolution.height):
                img = img.resize((self.resolution.width, self.resolution.height))

            # Add timestamp overlay
            draw = ImageDraw.Draw(img)
            timestamp_text = f"{cam_config.name} - {datetime.now().strftime('%H:%M:%S')}"
            try:
                font = ImageFont.load_default()
                draw.text((10, img.height - 30), timestamp_text, fill=(255, 255, 0), font=font)
            except:
                draw.text((10, img.height - 30), timestamp_text, fill=(255, 255, 0))

            # Apply rotation
            if self.rotation != 0:
                img = img.rotate(-self.rotation, expand=True)

            # Save
            img.save(filepath, quality=self.quality if self.image_format.lower() in ['jpg', 'jpeg'] else None)

            self.image_counter += 1
            return True

        except Exception as e:
            print(f"Static capture error: {e}")
            return False

    async def _capture_synthetic(self, cam_config, filepath: str) -> bool:
        """Generate synthetic test images."""
        if not PIL_AVAILABLE:
            # Create a minimal placeholder file
            with open(filepath, 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01' + b'\x00' * 100 + b'\xff\xd9')
            self.image_counter += 1
            return True

        try:
            # Create synthetic image
            img = Image.new('RGB', (self.resolution.width, self.resolution.height))
            draw = ImageDraw.Draw(img)

            # Create a gradient background
            for y in range(self.resolution.height):
                color_intensity = int(255 * (y / self.resolution.height))
                for x in range(self.resolution.width):
                    blue_intensity = int(255 * (x / self.resolution.width))
                    draw.point((x, y), (color_intensity//2, color_intensity, blue_intensity))

            # Add some "terrain" features
            num_features = random.randint(3, 8)
            for _ in range(num_features):
                shape_type = random.choice(['circle', 'rectangle', 'line'])
                x = random.randint(0, self.resolution.width)
                y = random.randint(0, self.resolution.height)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                if shape_type == 'circle':
                    radius = random.randint(10, 50)
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
                elif shape_type == 'rectangle':
                    w, h = random.randint(20, 100), random.randint(20, 100)
                    draw.rectangle([x, y, x+w, y+h], fill=color)
                elif shape_type == 'line':
                    x2, y2 = random.randint(0, self.resolution.width), random.randint(0, self.resolution.height)
                    draw.line([x, y, x2, y2], fill=color, width=random.randint(2, 5))

            # Add camera info and timestamp
            info_text = [
                f"SYNTHETIC CAMERA: {cam_config.name}",
                f"Port: {cam_config.port}",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Resolution: {self.resolution.width}x{self.resolution.height}",
                f"Counter: {self.image_counter}"
            ]

            try:
                font = ImageFont.load_default()
            except:
                font = None

            y_offset = 10
            for line in info_text:
                if font:
                    draw.text((10, y_offset), line, fill=(255, 255, 255), font=font)
                else:
                    draw.text((10, y_offset), line, fill=(255, 255, 255))
                y_offset += 20

            # Apply rotation if needed
            if self.rotation != 0:
                img = img.rotate(-self.rotation, expand=True)

            # Save image
            img.save(filepath, quality=self.quality if self.image_format.lower() in ['jpg', 'jpeg'] else None)

            self.image_counter += 1
            return True

        except Exception as e:
            print(f"Synthetic capture error: {e}")
            return False

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute mock camera capture."""
        logger.info(f"Starting mock camera capture (strategy: {self.mock_strategy})")

        results = {"captured": 0, "failed": 0, "images": [], "mock_strategy": self.mock_strategy}

        for cam in self.cameras:
            try:
                # Create output directory
                Path(cam.output_folder).mkdir(parents=True, exist_ok=True)

                # Generate filename
                timestamp = datetime.now()
                filename = f"{cam.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{self.image_format}"
                filepath = os.path.join(cam.output_folder, filename)

                # Capture based on strategy
                success = False
                if self.mock_strategy == "opencv":
                    success = await self._capture_opencv(cam, filepath)
                    if not success:
                        logger.warning(f"OpenCV capture failed for {cam.name}, falling back to synthetic")
                        success = await self._capture_synthetic(cam, filepath)
                elif self.mock_strategy == "static":
                    success = await self._capture_static(cam, filepath)
                else:  # synthetic
                    success = await self._capture_synthetic(cam, filepath)

                if success and os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    resolution_str = f"{self.resolution.width}x{self.resolution.height}"

                    # Store image metadata in database
                    try:
                        async with engine.connect() as conn:
                            await conn.execute(
                                text("""
                                INSERT INTO images
                                (camera_name, camera_port, filepath, filename, file_size_bytes,
                                 resolution, format, quality, metadata)
                                VALUES (:camera_name, :camera_port, :filepath, :filename, :file_size,
                                        :resolution, :format, :quality, :metadata)
                                """),
                                {
                                    "camera_name": cam.name,
                                    "camera_port": cam.port,
                                    "filepath": filepath,
                                    "filename": filename,
                                    "file_size": file_size,
                                    "resolution": resolution_str,
                                    "format": self.image_format,
                                    "quality": self.quality,
                                    "metadata": json.dumps({
                                        "rotation": self.rotation,
                                        "capture_timeout": self.capture_timeout,
                                        "timestamp": timestamp.isoformat(),
                                        "mock_strategy": self.mock_strategy,
                                        "mock_counter": self.image_counter,
                                        "system": platform.system()
                                    })
                                }
                            )
                            await conn.commit()

                        logger.info(f"Stored mock image metadata for {cam.name}: {filename}")
                    except Exception as db_error:
                        logger.error(f"Failed to store image metadata: {db_error}")

                    results["captured"] += 1
                    results["images"].append({
                        "camera": cam.name,
                        "file": filename,
                        "path": filepath,
                        "size": file_size,
                        "mock_strategy": self.mock_strategy
                    })
                    logger.info(f"Mock captured {cam.name}: {filename} ({self.mock_strategy})")
                else:
                    results["failed"] += 1
                    logger.error(f"Failed to mock capture {cam.name}")

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Error in mock capture {cam.name}: {e}")

        logger.info(f"Mock capture complete: {results['captured']} success, {results['failed']} failed")
        return results

    def __del__(self):
        """Clean up OpenCV cameras."""
        if hasattr(self, 'opencv_cameras'):
            for cap in self.opencv_cameras.values():
                try:
                    cap.release()
                except:
                    pass
