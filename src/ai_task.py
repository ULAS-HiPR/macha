from task import Task
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging
import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
from config import MachaConfig, AiParameters
import asyncio

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback for environments with full TF

class AiTask(Task):
    """
    AI Task for SZNet_mini TFLite segmentation inference.
    Processes images from input_folder, saves segmentation masks to output_folder.
    """

    def __init__(self, config: MachaConfig):
        super().__init__(config)
        ai_params = None
        for task in config.tasks:
            if task.class_name in ["AiTask", "MockAiTask"] and task.parameters:
                if isinstance(task.parameters, AiParameters):
                    ai_params = task.parameters
                    break

        if not ai_params:
            raise ValueError("No valid AiTask configuration found")

        self.model_path = ai_params.model_path
        self.input_folder = getattr(ai_params, "input_folder", "images/cam0")
        self.output_folder = ai_params.output_folder
        self.confidence_threshold = ai_params.confidence_threshold
        self.output_format = ai_params.output_format
        self.save_confidence_overlay = ai_params.save_confidence_overlay
        self.class_names = ai_params.class_names
        self.class_colors = ai_params.class_colors

        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']
        self.input_shape = input_details[0]['shape']

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """
        Process all images in input_folder, run segmentation, save masks to output_folder.
        """
        logger.info("Starting AI segmentation task")
        results = {"processed": 0, "failed": 0, "masks": []}

        # Find all images in input_folder
        image_files = sorted([
            f for f in Path(self.input_folder).glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ])

        for image_path in image_files:
            try:
                mask_path, overlay_path = await self.process_image(image_path, logger)
                results["processed"] += 1
                results["masks"].append({
                    "input": str(image_path),
                    "mask": str(mask_path),
                    "overlay": str(overlay_path) if overlay_path else None
                })
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results["failed"] += 1

        logger.info(f"AI segmentation complete: {results['processed']} processed, {results['failed']} failed")
        return results

    async def process_image(self, image_path, logger):
        """
        Preprocess image, run inference, save mask and overlay.
        Returns (mask_path, overlay_path)
        """
        # 1. Load image
        image = Image.open(image_path).convert("RGB")
        input_data = np.array(image).astype(np.float32)

        # 2. Resize to 500x500 if needed
        if input_data.shape[0] != 500 or input_data.shape[1] != 500:
            image = image.resize((500, 500))
            input_data = np.array(image).astype(np.float32)

        # 3. Convert HWC to NCHW
        if len(input_data.shape) == 3:
            input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW

        # 4. Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)  # CHW -> NCHW

        # Validation
        assert input_data.shape == (1, 3, 500, 500), f"Wrong shape: {input_data.shape}"
        assert input_data.dtype == np.float32, f"Wrong dtype: {input_data.dtype}"
        assert 0 <= input_data.min() and input_data.max() <= 255, f"Wrong range: [{input_data.min()}, {input_data.max()}]"

        # 5. Run inference
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)  # [1, 2, 500, 500]

        # 6. Postprocess output
        safe_probs = output[0, 0]  # [500, 500]
        unsafe_probs = output[0, 1]  # [500, 500]
        threshold = safe_probs.mean() if self.confidence_threshold is None else self.confidence_threshold
        binary_mask = (safe_probs > threshold).astype(np.uint8)  # 1=safe, 0=unsafe

        # 7. Save mask
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
        mask_filename = Path(image_path).stem + "_mask." + self.output_format
        mask_path = Path(self.output_folder) / mask_filename
        mask_img.save(mask_path)

        overlay_path = None
        if self.save_confidence_overlay:
            # Create a color overlay for visualization
            overlay = np.zeros((500, 500, 3), dtype=np.uint8)
            overlay[binary_mask == 1] = self.class_colors.get("safe_landing", [0, 255, 0])
            overlay[binary_mask == 0] = self.class_colors.get("unsafe_landing", [255, 0, 0])
            overlay_img = Image.fromarray(overlay, mode="RGB")
            overlay_filename = Path(image_path).stem + "_overlay." + self.output_format
            overlay_path = Path(self.output_folder) / overlay_filename
            overlay_img.save(overlay_path)

        # Optionally, store results in DB
        try:
            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO ai_results
                    (input_image, mask_path, overlay_path, timestamp)
                    VALUES (:input_image, :mask_path, :overlay_path, :timestamp)
                    """),
                    {
                        "input_image": str(image_path),
                        "mask_path": str(mask_path),
                        "overlay_path": str(overlay_path) if overlay_path else "",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                )
                await conn.commit()
        except Exception as db_error:
            logger.warning(f"Could not store AI result in DB: {db_error}")

        logger.info(f"Processed {image_path.name}: mask saved to {mask_path.name}")
        return mask_path, overlay_path
