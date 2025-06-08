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


class CameraTask(Task):
    """Fast camera task using libcamera-still for cam0 and cam1."""

    def __init__(self, config: MachaConfig):
        super().__init__(config)
        # Find camera task config
        camera_params = None
        for task in config.tasks:
            if task.class_name == "CameraTask" and task.parameters:
                if isinstance(task.parameters, CameraParameters):
                    camera_params = task.parameters
                    break

        if not camera_params:
            raise ValueError("No valid CameraTask configuration found")

        self.cameras = camera_params.cameras
        self.image_format = camera_params.image_format
        self.resolution = camera_params.resolution
        self.quality = camera_params.quality
        self.rotation = camera_params.rotation
        self.capture_timeout = camera_params.capture_timeout

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Capture images from all cameras quickly."""
        logger.info("Starting camera capture")

        # Create images table if it doesn't exist
        await self._create_table(engine)

        results = {"captured": 0, "failed": 0, "images": []}

        for cam in self.cameras:
            try:
                # Create output dir
                Path(cam.output_folder).mkdir(parents=True, exist_ok=True)

                # Generate filename
                timestamp = datetime.now()
                filename = f"{cam.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.{self.image_format}"
                filepath = os.path.join(cam.output_folder, filename)

                # Capture with libcamera-still (fast)
                cmd = [
                    "libcamera-still",
                    "--camera",
                    str(cam.port),
                    "--output",
                    filepath,
                    "--width",
                    str(self.resolution.width),
                    "--height",
                    str(self.resolution.height),
                    "--timeout",
                    "500",
                    "--nopreview",
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3)

                if proc.returncode == 0 and os.path.exists(filepath):
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
                                        "command": cmd
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
                    error_msg = (
                        f"Failed to capture {cam.name} - Return code: {proc.returncode}"
                    )
                    if stderr:
                        error_msg += f" - Error: {stderr.decode().strip()}"
                    if stdout:
                        error_msg += f" - Output: {stdout.decode().strip()}"
                    if not os.path.exists(filepath):
                        error_msg += " - File not created"
                    logger.error(error_msg)

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Error capturing {cam.name}: {e}")

        logger.info(
            f"Capture complete: {results['captured']} success, {results['failed']} failed"
        )
        return results

    async def _create_table(self, engine: AsyncEngine):
        """Create the images table if it doesn't exist."""
        async with engine.connect() as conn:
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    camera_name TEXT NOT NULL,
                    camera_port INTEGER NOT NULL,
                    filepath TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    resolution TEXT,
                    format TEXT,
                    quality INTEGER,
                    metadata TEXT
                )
                """)
            )
            await conn.commit()
