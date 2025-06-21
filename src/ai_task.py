import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

# Import numpy (now a core dependency, always available)
import numpy as np

from task import Task
from config import MachaConfig, AiParameters

# Check for optional AI dependencies (excluding numpy)
try:
    import tflite_runtime.interpreter as tflite
    from PIL import Image, ImageDraw, ImageFont
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    tflite = None
    Image = None
    ImageDraw = None
    ImageFont = None

# Check for Coral TPU support
try:
    from pycoral.utils import edgetpu
    CORAL_TPU_AVAILABLE = True
except ImportError:
    CORAL_TPU_AVAILABLE = False
    edgetpu = None



class ModelManager:
    """Manages model loading and inference operations with TPU fallback."""

    def __init__(self, ai_params: AiParameters, logger: logging.Logger):
        self.ai_params = ai_params
        self.logger = logger
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.using_tpu = False
        self.model_loaded = False

    async def load_model(self) -> bool:
        """Load model with TPU fallback strategy."""
        if not MODEL_AVAILABLE:
            self.logger.error("TensorFlow Lite runtime not available")
            return False

        model_path = Path(self.ai_params.model_path)
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return False

        # Try Coral TPU first if requested and available
        if self.ai_params.use_coral_tpu and CORAL_TPU_AVAILABLE:
            if await self._try_load_tpu_model():
                self.using_tpu = True
                self.logger.info("Model loaded with Coral TPU acceleration")
                return True
            else:
                self.logger.warning("Coral TPU loading failed, falling back to CPU")

        # Fallback to CPU
        if await self._try_load_cpu_model():
            self.logger.info("Model loaded for CPU inference")
            return True

        self.logger.error("Failed to load model on both TPU and CPU")
        return False

    async def _try_load_tpu_model(self) -> bool:
        """Attempt to load TPU-optimized model."""
        try:
            # Look for EdgeTPU model first
            model_path = Path(self.ai_params.model_path)
            tpu_model_path = model_path.parent / (model_path.stem + '_edgetpu.tflite')

            if tpu_model_path.exists():
                model_file = str(tpu_model_path)
                self.logger.debug(f"Using EdgeTPU model: {model_file}")
            else:
                model_file = str(model_path)
                self.logger.debug(f"Using standard model with TPU delegate: {model_file}")

            self.interpreter = tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[edgetpu.make_edgetpu_delegate()]
            )
            self.interpreter.allocate_tensors()
            self._setup_io_details()
            self.model_loaded = True
            return True

        except Exception as e:
            self.logger.debug(f"TPU loading failed: {e}")
            return False

    async def _try_load_cpu_model(self) -> bool:
        """Load model for CPU inference."""
        try:
            self.interpreter = tflite.Interpreter(model_path=self.ai_params.model_path)
            self.interpreter.allocate_tensors()
            self._setup_io_details()
            self.model_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"CPU model loading failed: {e}")
            return False

    def _setup_io_details(self):
        """Set up input and output tensor details."""
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if self.logger:
            self.logger.debug(f"Model input shape: {self.input_details[0]['shape']}")
            self.logger.debug(f"Model output shape: {self.output_details[0]['shape']}")

    async def run_inference(self, preprocessed_image: np.ndarray) -> Optional[np.ndarray]:
        """Run inference on preprocessed image."""
        if not self.model_loaded or not self.interpreter:
            self.logger.error("Model not loaded")
            return None

        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)

            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            self.logger.debug(f"Inference completed in {inference_time:.1f}ms")
            return output_data

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model_loaded and self.interpreter is not None

    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        """Get model input shape."""
        if self.input_details:
            return tuple(self.input_details[0]['shape'])
        return None


class ImageProcessor:
    """Handles image preprocessing for inference."""

    def __init__(self, ai_params: AiParameters):
        self.ai_params = ai_params

    async def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess single image for inference."""
        if not MODEL_AVAILABLE:
            raise RuntimeError("PIL not available for image processing")

        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get target size from model or use default
                target_size = (224, 224)  # Default size

                # Resize image
                img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32)

                # Normalize to [0, 1]
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(img_array, axis=0)

                return img_array

        except Exception as e:
            raise RuntimeError(f"Failed to preprocess image {image_path}: {e}")

    def set_input_shape(self, input_shape: Tuple[int, ...]):
        """Set target input shape from model."""
        if len(input_shape) >= 3:
            # Extract height and width from shape (batch, height, width, channels)
            self.target_size = (input_shape[1], input_shape[2])


class InferenceResultHandler:
    """Processes and stores inference results."""

    def __init__(self, ai_params: AiParameters):
        self.ai_params = ai_params

    async def process_segmentation_result(
        self,
        output_data: np.ndarray,
        original_image_path: str,
        model_id: int,
        processing_time_ms: float
    ) -> Optional[Dict[str, Any]]:
        """Process segmentation output and save result image."""
        if not MODEL_AVAILABLE:
            return None

        try:
            # Remove batch dimension if present
            if len(output_data.shape) == 4:
                output_data = output_data[0]

            # Get segmentation mask (argmax over classes)
            segmentation_mask = np.argmax(output_data, axis=-1)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(output_data)

            # Create colored segmentation image
            colored_mask = self._create_colored_mask(segmentation_mask)

            # Generate output filename
            timestamp = datetime.now()
            original_name = Path(original_image_path).stem
            output_filename = f"{original_name}_seg_{timestamp.strftime('%Y%m%d_%H%M%S')}.{self.ai_params.output_format}"
            output_path = Path(self.ai_params.output_folder) / output_filename

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save segmentation result
            result_image = Image.fromarray(colored_mask.astype(np.uint8))
            result_image.save(str(output_path))

            # Get file size
            file_size = output_path.stat().st_size

            # Prepare result data
            result_data = {
                "output_filepath": str(output_path),
                "output_filename": output_filename,
                "file_size_bytes": file_size,
                "confidence_scores": confidence_scores,
                "processing_time_ms": int(processing_time_ms),
                "resolution": f"{colored_mask.shape[1]}x{colored_mask.shape[0]}",
                "model_id": model_id,
                "class_distribution": self._calculate_class_distribution(segmentation_mask),
                "metadata": {
                    "timestamp": timestamp.isoformat(),
                    "original_image": original_image_path,
                    "segmentation_shape": output_data.shape,
                    "confidence_threshold": self.ai_params.confidence_threshold
                }
            }

            return result_data

        except Exception as e:
            raise RuntimeError(f"Failed to process segmentation result: {e}")

    def _calculate_confidence_scores(self, output_data: np.ndarray) -> List[float]:
        """Calculate per-class confidence scores."""
        # Apply softmax to get probabilities
        exp_data = np.exp(output_data - np.max(output_data, axis=-1, keepdims=True))
        probabilities = exp_data / np.sum(exp_data, axis=-1, keepdims=True)

        # Calculate mean confidence per class
        mean_confidences = np.mean(probabilities, axis=(0, 1))
        return mean_confidences.tolist()

    def _create_colored_mask(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """Create colored segmentation mask."""
        height, width = segmentation_mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx, class_name in enumerate(self.ai_params.class_names):
            if class_name in self.ai_params.class_colors:
                color = self.ai_params.class_colors[class_name]
                colored_mask[segmentation_mask == class_idx] = color

        return colored_mask

    def _calculate_class_distribution(self, segmentation_mask: np.ndarray) -> Dict[str, float]:
        """Calculate percentage distribution of classes in the mask."""
        total_pixels = segmentation_mask.size
        distribution = {}

        for class_idx, class_name in enumerate(self.ai_params.class_names):
            class_pixels = np.sum(segmentation_mask == class_idx)
            percentage = (class_pixels / total_pixels) * 100
            distribution[class_name] = round(percentage, 2)

        return distribution

    async def store_result(self, engine: AsyncEngine, image_id: int, result_data: Dict[str, Any]) -> bool:
        """Store inference result in database."""
        try:
            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO segmentation_results
                    (source_image_id, ai_model_id, output_filepath, output_filename,
                     file_size_bytes, resolution, format, processing_time_ms,
                     confidence_scores, metadata)
                    VALUES (:source_image_id, :ai_model_id, :output_filepath, :output_filename,
                            :file_size_bytes, :resolution, :format, :processing_time_ms,
                            :confidence_scores, :metadata)
                    """),
                    {
                        "source_image_id": image_id,
                        "ai_model_id": result_data["model_id"],
                        "output_filepath": result_data["output_filepath"],
                        "output_filename": result_data["output_filename"],
                        "file_size_bytes": result_data["file_size_bytes"],
                        "resolution": result_data["resolution"],
                        "format": self.ai_params.output_format,
                        "processing_time_ms": result_data["processing_time_ms"],
                        "confidence_scores": json.dumps(result_data["confidence_scores"]),
                        "metadata": json.dumps(result_data["metadata"])
                    }
                )
                await conn.commit()
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to store result in database: {e}")


class AiTask(Task):
    """AI segmentation task for landing safety classification."""

    def __init__(self, config: MachaConfig):
        super().__init__(config)
        self.ai_params = self._get_ai_parameters(config)

        # Initialize components (lazy loading)
        self.model_manager = None
        self.image_processor = None
        self.result_handler = None
        self.initialized = False
        self.model_id = None

        # Critical system state for reliability
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.system_disabled = False

    def _get_ai_parameters(self, config: MachaConfig) -> AiParameters:
        """Extract AI parameters from config."""
        for task in config.tasks:
            if task.class_name == "AiTask" and isinstance(task.parameters, AiParameters):
                return task.parameters

        # Return default parameters if none found
        return AiParameters(
            model_path="model.tflite",
            model_name="default_model"
        )

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute segmentation on new images with bulletproof error handling."""

        # Never let this method crash the system
        try:
            return await self._safe_execute(engine, logger)
        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error in AI task: {e}")
            self.consecutive_failures += 1

            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.critical("AI task disabled due to repeated failures")
                self.system_disabled = True

            return {
                "status": "system_error",
                "processed": 0,
                "failed": 1,
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
                "system_disabled": self.system_disabled
            }

    async def _safe_execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Protected execution with comprehensive error handling."""

        if self.system_disabled:
            logger.warning("AI task is disabled due to previous failures")
            return {"status": "disabled", "processed": 0}

        # Check if AI libraries are available
        if not MODEL_AVAILABLE:
            logger.error("AI dependencies not available (PIL, numpy, tflite-runtime)")
            return {"status": "dependencies_unavailable", "processed": 0}

        # Lazy initialization
        if not await self._initialize_if_needed(engine, logger):
            return {"status": "initialization_failed", "processed": 0}

        # Get unprocessed images
        unprocessed_images = await self._get_unprocessed_images(engine, logger)
        if not unprocessed_images:
            logger.debug("No new images to process")
            return {"status": "no_new_images", "processed": 0}

        # Limit queue size
        if len(unprocessed_images) > self.ai_params.max_queue_size:
            logger.warning(f"Queue size ({len(unprocessed_images)}) exceeds maximum ({self.ai_params.max_queue_size}), processing subset")
            unprocessed_images = unprocessed_images[:self.ai_params.max_queue_size]

        # Process images with individual error handling
        results = await self._process_images_batch(engine, logger, unprocessed_images)

        # Reset failure counter on successful processing
        if results["processed"] > 0:
            self.consecutive_failures = 0

        return results

    async def _initialize_if_needed(self, engine: AsyncEngine, logger: logging.Logger) -> bool:
        """Initialize AI system following the sensor task pattern."""

        if self.initialized and self.model_manager and self.model_manager.is_loaded():
            return True

        try:
            logger.info("Initializing AI segmentation system...")

            # Create output directory
            Path(self.ai_params.output_folder).mkdir(parents=True, exist_ok=True)

            # Initialize components
            self.model_manager = ModelManager(self.ai_params, logger)
            self.image_processor = ImageProcessor(self.ai_params)
            self.result_handler = InferenceResultHandler(self.ai_params)

            # Load model (with Coral TPU fallback)
            if not await self.model_manager.load_model():
                logger.error("Failed to load AI model")
                return False

            # Set input shape for image processor
            input_shape = self.model_manager.get_input_shape()
            if input_shape:
                self.image_processor.set_input_shape(input_shape)

            # Register/update model in database
            self.model_id = await self._register_model(engine, logger)
            if not self.model_id:
                logger.error("Failed to register model in database")
                return False

            self.initialized = True
            logger.info(f"AI segmentation system initialized successfully (TPU: {self.model_manager.using_tpu})")
            return True

        except Exception as e:
            logger.error(f"AI initialization failed: {e}")
            self.initialized = False
            return False

    async def _register_model(self, engine: AsyncEngine, logger: logging.Logger) -> Optional[int]:
        """Register model in database and return model ID."""
        try:
            async with engine.connect() as conn:
                # Check if model already exists
                result = await conn.execute(
                    text("SELECT id FROM ai_models WHERE name = :name AND version = :version"),
                    {"name": self.ai_params.model_name, "version": self.ai_params.model_version}
                )
                existing = result.fetchone()

                if existing:
                    model_id = existing[0]
                    logger.debug(f"Using existing model registration: {model_id}")
                    return model_id

                # Register new model
                input_shape = self.model_manager.get_input_shape()
                await conn.execute(
                    text("""
                    INSERT INTO ai_models
                    (name, version, filepath, model_type, input_height, input_width,
                     input_channels, output_classes, is_active, metadata)
                    VALUES (:name, :version, :filepath, :model_type, :input_height,
                            :input_width, :input_channels, :output_classes, :is_active, :metadata)
                    """),
                    {
                        "name": self.ai_params.model_name,
                        "version": self.ai_params.model_version,
                        "filepath": self.ai_params.model_path,
                        "model_type": "tflite_edgetpu" if self.model_manager.using_tpu else "tflite",
                        "input_height": input_shape[1] if input_shape else 224,
                        "input_width": input_shape[2] if input_shape else 224,
                        "input_channels": input_shape[3] if input_shape else 3,
                        "output_classes": json.dumps(self.ai_params.class_names),
                        "is_active": True,
                        "metadata": json.dumps({
                            "confidence_threshold": self.ai_params.confidence_threshold,
                            "class_colors": self.ai_params.class_colors,
                            "coral_tpu_enabled": self.model_manager.using_tpu
                        })
                    }
                )

                # Get the inserted model ID
                result = await conn.execute(
                    text("SELECT id FROM ai_models WHERE name = :name AND version = :version"),
                    {"name": self.ai_params.model_name, "version": self.ai_params.model_version}
                )
                new_model = result.fetchone()

                await conn.commit()

                if new_model:
                    model_id = new_model[0]
                    logger.info(f"Registered new model: {self.ai_params.model_name} v{self.ai_params.model_version} (ID: {model_id})")
                    return model_id

        except Exception as e:
            logger.error(f"Failed to register model: {e}")

        return None

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

    async def _process_images_batch(self, engine: AsyncEngine, logger: logging.Logger,
                                   images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch of images with individual error handling."""

        results = {
            "status": "completed",
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": []
        }

        for image_info in images:
            image_id = image_info["id"]
            image_path = image_info["filepath"]

            try:
                # Mark as processing
                await self._update_processing_status(engine, image_id, "processing")

                # Process single image with timeout
                async with asyncio.timeout(self.ai_params.processing_timeout):
                    segmentation_result = await self._process_single_image(
                        engine, logger, image_info
                    )

                if segmentation_result:
                    await self._update_processing_status(engine, image_id, "completed")
                    results["processed"] += 1
                    logger.info(f"Processed image {image_id}: {image_info['filename']}")
                else:
                    await self._update_processing_status(engine, image_id, "failed", "Processing returned no result")
                    results["failed"] += 1

            except asyncio.TimeoutError:
                error_msg = f"Processing timeout for image {image_id}"
                logger.warning(error_msg)
                await self._update_processing_status(engine, image_id, "failed", error_msg)
                results["failed"] += 1
                results["errors"].append(error_msg)

            except FileNotFoundError:
                error_msg = f"Image file not found: {image_path}"
                logger.warning(error_msg)
                await self._update_processing_status(engine, image_id, "skipped", error_msg)
                results["skipped"] += 1

            except Exception as e:
                error_msg = f"Error processing image {image_id}: {e}"
                logger.error(error_msg)
                await self._update_processing_status(engine, image_id, "failed", error_msg)
                results["failed"] += 1
                results["errors"].append(error_msg)

                # Don't let individual image errors break the batch
                continue

        logger.info(f"Batch processing complete: {results['processed']} processed, {results['failed']} failed, {results['skipped']} skipped")
        return results

    async def _process_single_image(self, engine: AsyncEngine, logger: logging.Logger,
                                   image_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single image through the inference pipeline."""
        image_path = image_info["filepath"]
        image_id = image_info["id"]

        try:
            start_time = time.time()

            # Preprocess image
            preprocessed_image = await self.image_processor.preprocess_image(image_path)
            if preprocessed_image is None:
                logger.error(f"Failed to preprocess image: {image_path}")
                return None

            # Run inference
            output_data = await self.model_manager.run_inference(preprocessed_image)
            if output_data is None:
                logger.error(f"Inference failed for image: {image_path}")
                return None

            # Process results
            processing_time_ms = (time.time() - start_time) * 1000
            result_data = await self.result_handler.process_segmentation_result(
                output_data, image_path, self.model_id, processing_time_ms
            )

            if result_data is None:
                logger.error(f"Failed to process segmentation result for: {image_path}")
                return None

            # Store in database
            if await self.result_handler.store_result(engine, image_id, result_data):
                logger.debug(f"Stored segmentation result for image {image_id}")
                return result_data
            else:
                logger.error(f"Failed to store result for image {image_id}")
                return None

        except Exception as e:
            logger.error(f"Single image processing failed for {image_path}: {e}")
            return None

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
            # Don't let status update failures break the main processing
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update processing status: {e}")
