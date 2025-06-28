from task import Task
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from config import MachaConfig, CameraParameters

class Picamera2Task(Task):
    """Camera task using picamera2 for Raspberry Pi Camera V2."""

    def __init__(self, config: MachaConfig):
        super().__init__(config)
        # Find camera task config for this class
        camera_params = None
        for task in config.tasks:
            if task.class_name == "Picamera2Task" and task.parameters:
                if isinstance(task.parameters, CameraParameters):
                    camera_params = task.parameters
                    break

        if not camera_params:
            raise ValueError("No valid Picamera2Task configuration found")

        self.cameras = camera_params.cameras
        self.image_format = camera_params.image_format
        self.resolution = camera_params.resolution
        self.quality = camera_params.quality
        self.rotation = camera_params.rotation
        self.capture_timeout = camera_params.capture_timeout

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Capture images from all cameras using picamera2."""
        logger.info("Starting picamera2 capture")

        results = {"captured": 0, "failed": 0, "images": []}

        try:
            from picamera2 import Picamera2
        except ImportError:
            logger.error("picamera2 library not installed")
            results["failed"] = len(self.cameras)
            return results

        for cam in self.cameras:
            try:
                Path(cam.output_folder).mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now()
                filename = f"{cam.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{self.image_format}"
                filepath = os.path.join(cam.output_folder, filename)

                picam2 = Picamera2()
                config = picam2.create_still_configuration(
                    main={
                        "size": (self.resolution.width, self.resolution.height),
                        "format": self.image_format.upper()
                    }
                )
                picam2.configure(config)
                picam2.start()
                await asyncio.sleep(2)  # Camera warm-up

                picam2.capture_file(filepath)
                picam2.stop()

                if os.path.exists(filepath):
                    file_size = os.path.getsize(filepath)
                    resolution_str = f"{self.resolution.width}x{self.resolution.height}"
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
                                    "camera_port": getattr(cam, "port", 0),
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
                                        "library": "picamera2"
                                    })
                                }
                            )
                            await conn.commit()
                        logger.info(f"Stored image metadata for {cam.name}: {filename}")
                    except Exception as db_error:
                        logger.error(f"Failed to store image metadata: {db_error}")

                    results["captured"] += 1
                    results["images"].append(
                        {
                            "camera": cam.name,
                            "file": filename,
                            "path": filepath,
                            "size": file_size,
                        }
                    )
                    logger.info(f"Captured {cam.name}: {filename}")
                else:
                    results["failed"] += 1
                    logger.error(f"Failed to capture {cam.name} - File not created")
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Error capturing {cam.name} with picamera2: {e}")

        logger.info(
            f"Picamera2 capture complete: {results['captured']} success, {results['failed']} failed"
        )
        return results
