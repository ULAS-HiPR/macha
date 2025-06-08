from task import Task
from sqlalchemy.ext.asyncio import AsyncEngine
import logging
import asyncio
import os
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

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Capture images from all cameras quickly."""
        logger.info("Starting camera capture")

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
                    results["captured"] += 1
                    results["images"].append(
                        {
                            "camera": cam.name,
                            "file": filename,
                            "size": os.path.getsize(filepath),
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
