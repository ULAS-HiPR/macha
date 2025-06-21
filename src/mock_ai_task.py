import json
import math
import random
import time
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

from task import Task
from config import MachaConfig, AiParameters

# Try to import image processing libraries
try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    IMAGE_LIBS_AVAILABLE = True
except ImportError:
    IMAGE_LIBS_AVAILABLE = False
    np = None
    Image = None


class MockAiTask(Task):
    """
    Mock AI task for development and testing without actual AI models.

    Applies various color filters and effects to simulate AI processing:
    1. "sepia" - Apply sepia tone filter
    2. "blue_tint" - Apply blue color tint
    3. "red_tint" - Apply red color tint
    4. "edge_enhance" - Enhance edges in the image
    5. "vintage" - Apply vintage photo effect
    6. "green_tint" - Apply green color tint for "safe landing" simulation
    7. "auto" - Randomly select from available filters
    """

    def __init__(self, config: MachaConfig, mock_strategy: str = "auto"):
        super().__init__(config)

        if not IMAGE_LIBS_AVAILABLE:
            raise ImportError("PIL and numpy are required for MockAiTask. Install with: pip install Pillow numpy")

        # Find AI task config
        ai_params = None
        for task_config in config.tasks:
            if task_config.class_name in ["AiTask", "MockAiTask"] and task_config.name == getattr(self, 'task_name', 'ai_segmentation'):
                ai_params = task_config.parameters
                break

        if ai_params is None:
            # Use default parameters if not found in config
            ai_params = AiParameters(
                model_path="mock_model.tflite",
                model_name="mock_segmentation",
                output_folder="mock_ai_outputs"
            )

        self.ai_params = ai_params
        self.mock_strategy = self._determine_strategy(mock_strategy)

        # Create output directory
        Path(self.ai_params.output_folder).mkdir(parents=True, exist_ok=True)

        # Mock model state
        self.model_loaded = False
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.system_disabled = False

        # Filter strategies
        self.filter_strategies = [
            "sepia", "blue_tint", "red_tint", "edge_enhance",
            "vintage", "green_tint", "purple_tint"
        ]

    def _determine_strategy(self, strategy: str) -> str:
        """Determine the best mock strategy."""
        if strategy == "auto":
            # Randomly select a strategy for variety
            return random.choice(["sepia", "blue_tint", "red_tint", "green_tint", "vintage"])
        return strategy

    def _apply_sepia_filter(self, image: Image.Image) -> Image.Image:
        """Apply sepia tone filter."""
        # Convert to numpy array
        img_array = np.array(image)

        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])

        # Apply sepia filter
        sepia_img = img_array.dot(sepia_filter.T)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

        return Image.fromarray(sepia_img)

    def _apply_color_tint(self, image: Image.Image, color: str) -> Image.Image:
        """Apply color tint to image."""
        img_array = np.array(image)

        if color == "blue":
            # Enhance blue channel
            img_array[:,:,0] = img_array[:,:,0] * 0.8  # Reduce red
            img_array[:,:,1] = img_array[:,:,1] * 0.9  # Reduce green slightly
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.3, 0, 255)  # Enhance blue
        elif color == "red":
            # Enhance red channel
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.3, 0, 255)  # Enhance red
            img_array[:,:,1] = img_array[:,:,1] * 0.8  # Reduce green
            img_array[:,:,2] = img_array[:,:,2] * 0.8  # Reduce blue
        elif color == "green":
            # Enhance green channel (simulate "safe landing" areas)
            img_array[:,:,0] = img_array[:,:,0] * 0.8  # Reduce red
            img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.4, 0, 255)  # Enhance green
            img_array[:,:,2] = img_array[:,:,2] * 0.7  # Reduce blue
        elif color == "purple":
            # Purple tint
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.2, 0, 255)  # Enhance red
            img_array[:,:,1] = img_array[:,:,1] * 0.8  # Reduce green
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.3, 0, 255)  # Enhance blue

        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_vintage_filter(self, image: Image.Image) -> Image.Image:
        """Apply vintage photo effect."""
        # Reduce saturation
        enhancer = ImageEnhance.Color(image)
        vintage_img = enhancer.enhance(0.7)

        # Slightly reduce brightness and increase contrast
        enhancer = ImageEnhance.Brightness(vintage_img)
        vintage_img = enhancer.enhance(0.9)

        enhancer = ImageEnhance.Contrast(vintage_img)
        vintage_img = enhancer.enhance(1.2)

        # Add slight sepia tone
        img_array = np.array(vintage_img)
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.1, 0, 255)  # Slight red boost
        img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.05, 0, 255)  # Slight yellow

        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        """Apply edge enhancement filter."""
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    def _process_image_with_filter(self, image_path: str, strategy: str) -> Image.Image:
        """Process image with the specified filter strategy."""
        # Load image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Apply filter based on strategy
            if strategy == "sepia":
                processed_img = self._apply_sepia_filter(img)
            elif strategy == "blue_tint":
                processed_img = self._apply_color_tint(img, "blue")
            elif strategy == "red_tint":
                processed_img = self._apply_color_tint(img, "red")
            elif strategy == "green_tint":
                processed_img = self._apply_color_tint(img, "green")
            elif strategy == "purple_tint":
                processed_img = self._apply_color_tint(img, "purple")
            elif strategy == "vintage":
                processed_img = self._apply_vintage_filter(img)
            elif strategy == "edge_enhance":
                processed_img = self._apply_edge_enhance(img)
            else:
                # Default to sepia
                processed_img = self._apply_sepia_filter(img)

            return processed_img

    async def _get_unprocessed_images(self, engine: AsyncEngine, logger: logging.Logger) -> List[Dict[str, Any]]:
        """Get list of images that haven't been processed yet."""
        try:
            async with engine.connect() as conn:
                # Query for images that haven't been processed or failed processing
                result = await conn.execute(
                    text("""
                    SELECT i.id, i.filepath, i.filename, i.timestamp
                    FROM images i
                    LEFT JOIN ai_processing_queue apq ON i.id = apq.image_id
                    WHERE apq.id IS NULL
                       OR (apq.status = 'failed' AND apq.attempts < :max_retries)
                    ORDER BY i.timestamp ASC
                    LIMIT :limit
                    """),
                    {
                        "max_retries": self.ai_params.max_retries,
                        "limit": self.ai_params.max_queue_size
                    }
                )

                rows = result.fetchall()
                images = []
                for row in rows:
                    images.append({
                        "id": row[0],
                        "filepath": row[1],
                        "filename": row[2],
                        "timestamp": row[3]
                    })

                logger.debug(f"Found {len(images)} unprocessed images")
                return images

        except Exception as e:
            logger.error(f"Failed to query unprocessed images: {e}")
            return []

    async def _update_processing_status(self, engine: AsyncEngine, image_id: int,
                                      status: str, error_message: Optional[str] = None):
        """Update processing status in queue table."""
        try:
            async with engine.connect() as conn:
                # Check if entry exists
                result = await conn.execute(
                    text("SELECT id, attempts FROM ai_processing_queue WHERE image_id = :image_id"),
                    {"image_id": image_id}
                )
                existing = result.fetchone()

                if existing:
                    # Update existing entry
                    await conn.execute(
                        text("""
                        UPDATE ai_processing_queue
                        SET status = :status, processed_at = CURRENT_TIMESTAMP,
                            error_message = :error_message, attempts = attempts + 1
                        WHERE image_id = :image_id
                        """),
                        {
                            "image_id": image_id,
                            "status": status,
                            "error_message": error_message
                        }
                    )
                else:
                    # Insert new entry
                    await conn.execute(
                        text("""
                        INSERT INTO ai_processing_queue
                        (image_id, status, processed_at, error_message, attempts)
                        VALUES (:image_id, :status, CURRENT_TIMESTAMP, :error_message, 1)
                        """),
                        {
                            "image_id": image_id,
                            "status": status,
                            "error_message": error_message
                        }
                    )

                await conn.commit()

        except Exception as e:
            logger.debug(f"Failed to update processing status: {e}")

    async def _store_segmentation_result(self, engine: AsyncEngine, logger: logging.Logger,
                                       image_id: int, output_path: str, processing_time_ms: int) -> bool:
        """Store segmentation result in database."""
        try:
            output_file = Path(output_path)
            file_size = output_file.stat().st_size

            # Mock confidence scores based on filter type
            mock_scores = {
                "sepia": {"background": 0.2, "safe_landing": 0.6, "unsafe_landing": 0.2},
                "green_tint": {"background": 0.1, "safe_landing": 0.8, "unsafe_landing": 0.1},
                "red_tint": {"background": 0.1, "safe_landing": 0.2, "unsafe_landing": 0.7},
                "blue_tint": {"background": 0.7, "safe_landing": 0.2, "unsafe_landing": 0.1},
                "vintage": {"background": 0.4, "safe_landing": 0.4, "unsafe_landing": 0.2},
                "purple_tint": {"background": 0.3, "safe_landing": 0.5, "unsafe_landing": 0.2},
                "edge_enhance": {"background": 0.5, "safe_landing": 0.3, "unsafe_landing": 0.2}
            }

            confidence_scores = mock_scores.get(self.mock_strategy, mock_scores["sepia"])

            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO segmentation_results
                    (source_image_id, ai_model_id, output_filepath, output_filename,
                     file_size_bytes, resolution, format, processing_time_ms,
                     confidence_scores, metadata)
                    VALUES (NULL, NULL, :filepath, :filename, :file_size, :resolution,
                            :format, :processing_time, :confidence_scores, :metadata)
                    """),
                    {
                        "filepath": str(output_path),
                        "filename": output_file.name,
                        "file_size": file_size,
                        "resolution": f"unknown",  # Would need to read from image
                        "format": self.ai_params.output_format,
                        "processing_time": processing_time_ms,
                        "confidence_scores": json.dumps(confidence_scores),
                        "metadata": json.dumps({
                            "mock_strategy": self.mock_strategy,
                            "filter_applied": self.mock_strategy,
                            "is_mock": True,
                            "source_image_id": image_id
                        })
                    }
                )
                await conn.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to store segmentation result: {e}")
            return False

    async def _process_single_image(self, engine: AsyncEngine, logger: logging.Logger,
                                  image_info: Dict[str, Any]) -> bool:
        """Process a single image with mock AI."""
        image_id = image_info["id"]
        image_path = image_info["filepath"]

        start_time = time.time()

        try:
            # Check if source image exists
            if not os.path.exists(image_path):
                logger.warning(f"Source image not found: {image_path}")
                return False

            # Simulate AI inference time (1 second + random variation)
            inference_time = 1.0 + random.uniform(-0.2, 0.5)
            await asyncio.sleep(inference_time)

            # Randomly select strategy if auto
            if self.mock_strategy == "auto":
                current_strategy = random.choice(self.filter_strategies)
            else:
                current_strategy = self.mock_strategy

            # Process image with filter
            try:
                processed_img = self._process_image_with_filter(image_path, current_strategy)
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                return False

            # Generate output filename
            input_filename = Path(image_path).stem
            output_filename = f"{input_filename}_processed_{current_strategy}.{self.ai_params.output_format}"
            output_path = Path(self.ai_params.output_folder) / output_filename

            # Save processed image
            try:
                processed_img.save(output_path, format=self.ai_params.output_format.upper())
            except Exception as e:
                logger.error(f"Failed to save processed image to {output_path}: {e}")
                return False

            # Calculate processing time
            end_time = time.time()
            processing_time_ms = int((end_time - start_time) * 1000)

            # Store result in database
            success = await self._store_segmentation_result(
                engine, logger, image_id, str(output_path), processing_time_ms
            )

            if success:
                logger.info(f"Mock AI processed {image_info['filename']} -> {output_filename} "
                           f"(filter: {current_strategy}, time: {processing_time_ms}ms)")
                return True
            else:
                logger.error(f"Failed to store result for {image_info['filename']}")
                return False

        except Exception as e:
            logger.error(f"Error in mock AI processing: {e}")
            return False

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the mock AI segmentation task."""
        if self.system_disabled:
            logger.warning("Mock AI task is disabled due to previous failures")
            return {"status": "disabled", "processed": 0}

        logger.info(f"Executing mock AI task (strategy: {self.mock_strategy})")

        try:
            # Get unprocessed images
            unprocessed_images = await self._get_unprocessed_images(engine, logger)

            if not unprocessed_images:
                logger.debug("No unprocessed images found")
                return {
                    "status": "no_work",
                    "processed": 0,
                    "failed": 0,
                    "message": "No unprocessed images"
                }

            logger.info(f"Found {len(unprocessed_images)} images to process")

            # Process images
            results = {
                "status": "completed",
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": []
            }

            for image_info in unprocessed_images:
                image_id = image_info["id"]

                try:
                    # Mark as processing
                    await self._update_processing_status(engine, image_id, "processing")

                    # Process image
                    success = await self._process_single_image(engine, logger, image_info)

                    if success:
                        await self._update_processing_status(engine, image_id, "completed")
                        results["processed"] += 1
                        self.consecutive_failures = 0  # Reset failure counter
                    else:
                        await self._update_processing_status(engine, image_id, "failed", "Mock processing failed")
                        results["failed"] += 1
                        self.consecutive_failures += 1

                except Exception as e:
                    error_msg = f"Error processing image {image_id}: {e}"
                    logger.error(error_msg)
                    await self._update_processing_status(engine, image_id, "failed", error_msg)
                    results["failed"] += 1
                    results["errors"].append(error_msg)
                    self.consecutive_failures += 1

            # Check if we should disable the task
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.critical("Mock AI task disabled due to repeated failures")
                self.system_disabled = True
                results["status"] = "system_disabled"

            logger.info(f"Mock AI processing complete: {results['processed']} processed, "
                       f"{results['failed']} failed")

            return results

        except Exception as e:
            logger.error(f"Critical error in mock AI task: {e}")
            self.consecutive_failures += 1

            if self.consecutive_failures >= self.max_consecutive_failures:
                self.system_disabled = True

            return {
                "status": "system_error",
                "processed": 0,
                "failed": 1,
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
                "system_disabled": self.system_disabled
            }

    def reset_simulation(self):
        """Reset simulation state for testing."""
        self.consecutive_failures = 0
        self.system_disabled = False
        self.model_loaded = False

    def set_mock_strategy(self, strategy: str):
        """Change mock strategy at runtime."""
        if strategy in self.filter_strategies + ["auto"]:
            self.mock_strategy = strategy
        else:
            raise ValueError(f"Invalid mock strategy: {strategy}")

    def get_simulation_state(self) -> Dict:
        """Get current simulation state for debugging."""
        return {
            "mock_strategy": self.mock_strategy,
            "available_strategies": self.filter_strategies,
            "consecutive_failures": self.consecutive_failures,
            "system_disabled": self.system_disabled,
            "model_loaded": self.model_loaded,
            "output_folder": self.ai_params.output_folder
        }
