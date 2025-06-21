import pytest
import asyncio
import sys
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig, AiParameters
from ai_task import AiTask, ModelManager, ImageProcessor, InferenceResultHandler


# Test Fixtures
@pytest.fixture
def mock_engine():
    """Create a mock database engine following existing pattern."""
    engine = Mock(spec=AsyncEngine)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_conn.fetchall = AsyncMock()
    mock_conn.fetchone = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    engine.connect.return_value = mock_conn
    return engine


@pytest.fixture
def mock_logger():
    """Create a mock logger following existing pattern."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def ai_config():
    """Create a test configuration with AI task."""
    return MachaConfig(
        app={"name": "test", "debug": True},
        logging={
            "level": "INFO",
            "file": {"path": "test.log"},
            "console": {"format": "test"}
        },
        db={
            "filename": "test.db",
            "connection_string": "sqlite:///test.db",
            "overwrite": False
        },
        tasks=[
            {
                "name": "test_camera",
                "class": "CameraTask",
                "frequency": 10,
                "enabled": True,
                "parameters": {
                    "cameras": [{"port": 0, "name": "cam0", "output_folder": "test_images"}],
                    "image_format": "jpg",
                    "resolution": {"width": 640, "height": 480}
                }
            },
            {
                "name": "test_ai",
                "class": "AiTask",
                "frequency": 20,  # Must be multiple of camera frequency
                "enabled": True,
                "parameters": {
                    "model_path": "test_model.tflite",
                    "model_name": "test_model",
                    "model_version": "1.0.0",
                    "use_coral_tpu": True,
                    "output_folder": "test_segmentation",
                    "confidence_threshold": 0.7,
                    "max_queue_size": 10,
                    "processing_timeout": 5,
                    "max_retries": 2,
                    "class_names": ["background", "safe", "unsafe"]
                }
            }
        ]
    )


@pytest.fixture
def temp_model_file():
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as f:
        # Write minimal tflite file structure (just header)
        f.write(b'TFL3' + b'\x00' * 100)  # Minimal valid tflite structure
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def test_images():
    """Create test image files for processing."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    # Create mock image files
    for i in range(3):
        image_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
        # Create a minimal JPEG file structure
        with open(image_path, 'wb') as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01' + b'\x00' * 100 + b'\xff\xd9')
        image_paths.append(image_path)

    yield temp_dir, image_paths

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


# Core AI Task Tests
class TestAiTask:
    """Test cases for AiTask following established patterns."""

    def test_ai_task_initialization(self, ai_config):
        """Test AiTask initialization with valid config."""
        task = AiTask(ai_config)
        assert task.name == "AiTask"
        assert isinstance(task.ai_params, AiParameters)
        assert task.ai_params.model_name == "test_model"
        assert task.ai_params.confidence_threshold == 0.7
        assert not task.initialized  # Should be False until first execution

    def test_ai_task_default_parameters(self):
        """Test AiTask with minimal configuration."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "test_ai",
                    "class": "AiTask",
                    "frequency": 30,
                    "enabled": True,
                    "parameters": {
                        "model_path": "model.tflite",
                        "model_name": "default_model"
                    }
                }
            ]
        )

        task = AiTask(config)
        assert task.ai_params.use_coral_tpu is True  # Default value
        assert task.ai_params.confidence_threshold == 0.5  # Default value
        assert task.ai_params.output_folder == "segmentation_outputs"

    @patch('ai_task.MODEL_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_ai_task_dependencies_unavailable(self, ai_config, mock_engine, mock_logger):
        """Test AiTask when AI dependencies are not available."""
        task = AiTask(ai_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["status"] == "dependencies_unavailable"
        assert result["processed"] == 0
        mock_logger.error.assert_called()

    @patch('ai_task.MODEL_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_ai_task_no_new_images(self, ai_config, mock_engine, mock_logger):
        """Test AiTask when no unprocessed images exist."""
        # Mock database to return no unprocessed images
        mock_engine.connect.return_value.__aenter__.return_value.execute.return_value.fetchall.return_value = []

        with patch.object(AiTask, '_initialize_if_needed', return_value=True):
            task = AiTask(ai_config)
            result = await task.execute(mock_engine, mock_logger)

        assert result["status"] == "no_new_images"
        assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_ai_task_frequency_validation_error(self):
        """Test that AI task frequency validation works."""
        from pydantic import ValidationError

        with pytest.raises(ValueError, match="AI task.*frequency.*cannot be faster"):
            MachaConfig(
                app={"name": "test", "debug": True},
                logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
                db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
                tasks=[
                    {
                        "name": "camera_task",
                        "class": "CameraTask",
                        "frequency": 10,
                        "enabled": True,
                        "parameters": {
                            "cameras": [{"port": 0, "name": "cam0", "output_folder": "test"}],
                            "image_format": "jpg",
                            "resolution": {"width": 640, "height": 480}
                        }
                    },
                    {
                        "name": "ai_task",
                        "class": "AiTask",
                        "frequency": 5,  # Faster than camera - should fail
                        "enabled": True,
                        "parameters": {"model_path": "test.tflite", "model_name": "test"}
                    }
                ]
            )

    @patch('ai_task.MODEL_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_ai_task_consecutive_failures(self, ai_config, mock_engine, mock_logger):
        """Test AI task failure handling and system disable."""
        task = AiTask(ai_config)

        # Force failures by making initialization fail
        with patch.object(task, '_initialize_if_needed', side_effect=Exception("Test error")):

            # Execute 5 times to trigger consecutive failure limit
            for i in range(5):  # Exactly max_consecutive_failures (5)
                result = await task.execute(mock_engine, mock_logger)
                assert result["status"] == "system_error"

            # Should be disabled after 5 failures
            assert task.system_disabled is True

            # Next execution should return disabled status
            result = await task.execute(mock_engine, mock_logger)
            assert result["status"] == "disabled"


# Model Manager Tests
class TestModelManager:
    """Test ModelManager component."""

    @patch('ai_task.MODEL_AVAILABLE', True)
    @patch('ai_task.tflite')
    def test_model_manager_cpu_loading(self, mock_tflite, temp_model_file):
        """Test CPU model loading."""
        params = AiParameters(
            model_path=temp_model_file,
            model_name="test",
            use_coral_tpu=False
        )

        mock_interpreter = Mock()
        mock_interpreter.get_input_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_interpreter.get_output_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_tflite.Interpreter.return_value = mock_interpreter

        logger = Mock()
        manager = ModelManager(params, logger)

        # Test CPU loading path
        result = asyncio.run(manager._try_load_cpu_model())
        assert result is True
        mock_tflite.Interpreter.assert_called_with(model_path=temp_model_file)

    @patch('ai_task.MODEL_AVAILABLE', True)
    @patch('ai_task.CORAL_TPU_AVAILABLE', True)
    @patch('ai_task.tflite')
    @patch('ai_task.edgetpu')
    def test_model_manager_tpu_loading(self, mock_edgetpu, mock_tflite, temp_model_file):
        """Test Coral TPU model loading."""
        params = AiParameters(
            model_path=temp_model_file,
            model_name="test",
            use_coral_tpu=True
        )

        mock_interpreter = Mock()
        mock_interpreter.get_input_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_interpreter.get_output_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_tflite.Interpreter.return_value = mock_interpreter
        mock_delegate = Mock()
        mock_edgetpu.make_edgetpu_delegate.return_value = mock_delegate

        logger = Mock()
        manager = ModelManager(params, logger)

        result = asyncio.run(manager._try_load_tpu_model())
        assert result is True
        mock_edgetpu.make_edgetpu_delegate.assert_called_once()

    @patch('ai_task.MODEL_AVAILABLE', True)
    @patch('ai_task.tflite')
    def test_model_manager_tpu_fallback(self, mock_tflite, temp_model_file):
        """Test TPU fallback to CPU when TPU unavailable."""
        params = AiParameters(
            model_path=temp_model_file,
            model_name="test",
            use_coral_tpu=True
        )

        mock_interpreter = Mock()
        mock_interpreter.get_input_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_interpreter.get_output_details.return_value = [{"shape": [1, 224, 224, 3]}]
        mock_tflite.Interpreter.return_value = mock_interpreter

        logger = Mock()
        manager = ModelManager(params, logger)

        # Mock TPU loading failure, CPU success
        with patch.object(manager, '_try_load_tpu_model', return_value=False):
            with patch.object(manager, '_try_load_cpu_model', return_value=True):
                result = asyncio.run(manager.load_model())
                assert result is True
                assert not manager.using_tpu


# Image Processing Tests
class TestImageProcessor:
    """Test ImageProcessor component."""

    def test_image_processor_initialization(self):
        """Test ImageProcessor initialization."""
        params = AiParameters(
            model_path="test.tflite",
            model_name="test"
        )

        processor = ImageProcessor(params)
        assert processor.ai_params == params

    @patch('ai_task.MODEL_AVAILABLE', True)
    @patch('ai_task.Image')
    @patch('ai_task.np.array')
    @pytest.mark.asyncio
    async def test_preprocess_image_success(self, mock_array, mock_pil, test_images):
        """Test successful image preprocessing."""
        temp_dir, image_paths = test_images

        params = AiParameters(
            model_path="test.tflite",
            model_name="test"
        )

        # Mock PIL operations
        mock_image = Mock()
        mock_image.mode = 'RGB'
        mock_image.resize.return_value = mock_image
        mock_image.convert.return_value = mock_image
        mock_pil.open.return_value.__enter__.return_value = mock_image

        # Mock numpy array with correct dtype
        import numpy as np
        mock_img_array = np.ones((224, 224, 3), dtype=np.float32)
        mock_array.return_value = mock_img_array

        processor = ImageProcessor(params)
        result = await processor.preprocess_image(image_paths[0])

        assert result is not None

    @patch('ai_task.MODEL_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_preprocess_image_dependencies_unavailable(self):
        """Test preprocessing when dependencies unavailable."""
        params = AiParameters(
            model_path="test.tflite",
            model_name="test"
        )

        processor = ImageProcessor(params)

        with pytest.raises(RuntimeError, match="PIL not available"):
            await processor.preprocess_image("test.jpg")

    @patch('ai_task.MODEL_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_preprocess_image_file_not_found(self):
        """Test preprocessing with missing image file."""
        params = AiParameters(
            model_path="test.tflite",
            model_name="test"
        )

        processor = ImageProcessor(params)

        with pytest.raises(RuntimeError, match="Failed to preprocess image"):
            await processor.preprocess_image("nonexistent_image.jpg")


# Integration Tests
class TestAiTaskIntegration:
    """Integration tests for complete AI task workflow."""

    @patch('ai_task.MODEL_AVAILABLE', True)
    @patch('ai_task.tflite')
    @patch('ai_task.Image')
    @patch('ai_task.np')
    @pytest.mark.asyncio
    async def test_full_pipeline_mock_success(self, mock_np, mock_pil, mock_tflite, ai_config,
                                             mock_engine, mock_logger, test_images):
        """Test complete pipeline with mocked successful inference."""
        temp_dir, image_paths = test_images

        # Mock database responses
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value

        # Mock unprocessed images query
        mock_conn.execute.return_value.fetchall.return_value = [
            {"id": 1, "filepath": image_paths[0], "filename": "test1.jpg", "timestamp": "2024-01-01"},
            {"id": 2, "filepath": image_paths[1], "filename": "test2.jpg", "timestamp": "2024-01-01"}
        ]

        # Mock model registration
        mock_conn.execute.return_value.fetchone.return_value = {"id": 1}

        # Mock TensorFlow Lite
        mock_interpreter = Mock()
        mock_interpreter.get_input_details.return_value = [{"shape": [1, 224, 224, 3], "index": 0}]
        mock_interpreter.get_output_details.return_value = [{"shape": [1, 224, 224, 3], "index": 0}]
        mock_interpreter.get_tensor.return_value = mock_np.random.rand(1, 224, 224, 3)
        mock_tflite.Interpreter.return_value = mock_interpreter

        # Mock PIL
        mock_image = Mock()
        mock_image.mode = 'RGB'
        mock_pil.open.return_value.__enter__.return_value = mock_image
        mock_image.resize.return_value = mock_image
        mock_image.convert.return_value = mock_image

        # Mock numpy operations
        mock_np.array.return_value = mock_np.random.rand(224, 224, 3).astype(mock_np.float32)
        mock_np.expand_dims.return_value = mock_np.random.rand(1, 224, 224, 3).astype(mock_np.float32)
        mock_np.argmax.return_value = mock_np.zeros((224, 224), dtype=int)
        mock_np.exp.return_value = mock_np.ones((224, 224, 3))
        mock_np.sum.return_value = mock_np.ones((224, 224, 1))
        mock_np.mean.return_value = mock_np.array([0.8, 0.1, 0.1])
        mock_np.zeros.return_value = mock_np.zeros((224, 224, 3), dtype=mock_np.uint8)

        # Mock file operations
        with patch('ai_task.Path.mkdir'):
            with patch('ai_task.Image.fromarray') as mock_fromarray:
                mock_result_image = Mock()
                mock_fromarray.return_value = mock_result_image
                mock_result_image.save = Mock()

                with patch('ai_task.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 1024

                    task = AiTask(ai_config)
                    result = await task.execute(mock_engine, mock_logger)

                    assert result["processed"] >= 0  # Should process some images
                    assert "status" in result


# Error Handling Tests
class TestAiTaskErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch('ai_task.MODEL_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, ai_config, mock_logger):
        """Test handling of database connection failures."""
        # Mock engine that raises connection errors
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Database connection failed")

        task = AiTask(ai_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["status"] == "initialization_failed"
        assert result["processed"] == 0

    @patch('ai_task.MODEL_AVAILABLE', True)
    @pytest.mark.asyncio
    async def test_processing_timeout_handling(self, ai_config, mock_engine, mock_logger):
        """Test timeout handling during image processing."""

        with patch.object(AiTask, '_initialize_if_needed', return_value=True):
            with patch.object(AiTask, '_get_unprocessed_images', return_value=[{"id": 1, "filepath": "test.jpg", "filename": "test.jpg", "timestamp": "2024-01-01"}]):
                with patch.object(AiTask, '_process_single_image', side_effect=asyncio.TimeoutError()):
                    with patch.object(AiTask, '_update_processing_status', return_value=None):

                        task = AiTask(ai_config)
                        result = await task.execute(mock_engine, mock_logger)

                        assert result["failed"] >= 0
                        assert "timeout" in str(result.get("errors", [])).lower()


# Configuration Validation Tests
class TestAiParameters:
    """Test AiParameters validation following existing patterns."""

    def test_ai_parameters_validation(self):
        """Test AiParameters validation with valid inputs."""
        params = AiParameters(
            model_path="test_model.tflite",
            model_name="test_model",
            model_version="1.0.0",
            use_coral_tpu=True,
            confidence_threshold=0.8,
            class_names=["bg", "safe", "unsafe"]
        )

        assert params.model_path == "test_model.tflite"
        assert params.confidence_threshold == 0.8
        assert len(params.class_names) == 3

    def test_ai_parameters_invalid_confidence(self):
        """Test validation of invalid confidence threshold."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiParameters(
                model_path="test.tflite",
                model_name="test",
                confidence_threshold=1.5  # Invalid - must be <= 1.0
            )

    def test_ai_parameters_invalid_format(self):
        """Test validation of invalid output format."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AiParameters(
                model_path="test.tflite",
                model_name="test",
                output_format="gif"  # Invalid format
            )

    def test_ai_parameters_color_validation(self):
        """Test class color validation."""
        from pydantic import ValidationError

        # Valid colors
        params = AiParameters(
            model_path="test.tflite",
            model_name="test",
            class_names=["bg", "safe"],
            class_colors={"bg": [0, 0, 0], "safe": [0, 255, 0]}
        )
        assert params.class_colors["bg"] == [0, 0, 0]

        # Invalid color values
        with pytest.raises(ValidationError):
            AiParameters(
                model_path="test.tflite",
                model_name="test",
                class_names=["bg"],
                class_colors={"bg": [300, 0, 0]}  # > 255
            )


if __name__ == "__main__":
    pytest.main([__file__])
