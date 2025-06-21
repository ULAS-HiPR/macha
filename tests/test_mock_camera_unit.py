import pytest
import sys
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig
from mock_camera_task import MockCameraTask


@pytest.fixture
def mock_config():
    """Create a test configuration with mock camera task."""
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()

    config = MachaConfig(
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
                "name": "test_mock_camera",
                "class": "MockCameraTask",
                "frequency": 10,
                "enabled": True,
                "parameters": {
                    "cameras": [
                        {"port": 0, "name": "test_cam0", "output_folder": temp_dir1},
                        {"port": 1, "name": "test_cam1", "output_folder": temp_dir2}
                    ],
                    "image_format": "jpg",
                    "resolution": {"width": 640, "height": 480},
                    "quality": 85,
                    "rotation": 0,
                    "capture_timeout": 5,
                    "retry_attempts": 2
                }
            }
        ]
    )

    yield config

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir1, ignore_errors=True)
    shutil.rmtree(temp_dir2, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMockCameraTask:
    """Unit tests for MockCameraTask."""

    def test_initialization(self, mock_config):
        """Test MockCameraTask initialization."""
        task = MockCameraTask(mock_config, mock_strategy="synthetic")

        assert task.mock_strategy == "synthetic"
        assert len(task.cameras) == 2
        assert task.cameras[0].name == "test_cam0"
        assert task.cameras[1].name == "test_cam1"
        assert task.image_format == "jpg"
        assert task.resolution.width == 640
        assert task.resolution.height == 480

    def test_strategy_determination_macos(self, mock_config):
        """Test strategy determination on macOS."""
        with patch('platform.system', return_value='Darwin'):
            task = MockCameraTask(mock_config, mock_strategy="auto")
            assert task.mock_strategy == "opencv"

    def test_strategy_determination_opencv_available(self, mock_config):
        """Test strategy determination when OpenCV camera is available."""
        with patch('platform.system', return_value='Linux'), \
             patch.object(MockCameraTask, '_has_opencv_camera', return_value=True):
            task = MockCameraTask(mock_config, mock_strategy="auto")
            assert task.mock_strategy == "opencv"

    def test_strategy_determination_fallback(self, mock_config):
        """Test strategy determination fallback to synthetic."""
        with patch('platform.system', return_value='Linux'), \
             patch.object(MockCameraTask, '_has_opencv_camera', return_value=False):
            task = MockCameraTask(mock_config, mock_strategy="auto")
            assert task.mock_strategy == "synthetic"

    @patch('cv2.VideoCapture')
    def test_has_opencv_camera_available(self, mock_cv2_capture, mock_config):
        """Test OpenCV camera detection when camera is available."""
        # Mock successful camera access
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cv2_capture.return_value = mock_cap

        task = MockCameraTask(mock_config, mock_strategy="synthetic")
        result = task._has_opencv_camera()

        assert result is True
        mock_cv2_capture.assert_called_with(0)
        mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_has_opencv_camera_not_available(self, mock_cv2_capture, mock_config):
        """Test OpenCV camera detection when camera is not available."""
        # Mock failed camera access
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap

        task = MockCameraTask(mock_config, mock_strategy="synthetic")
        result = task._has_opencv_camera()

        assert result is False

    @patch('cv2.VideoCapture', side_effect=Exception("Camera error"))
    def test_has_opencv_camera_exception(self, mock_cv2_capture, mock_config):
        """Test OpenCV camera detection when exception occurs."""
        task = MockCameraTask(mock_config, mock_strategy="synthetic")
        result = task._has_opencv_camera()

        assert result is False

    def test_prepare_mock_assets(self, mock_config, temp_dir):
        """Test mock assets preparation."""
        with patch.object(Path, 'mkdir'), \
             patch.object(MockCameraTask, '_create_sample_image') as mock_create:
            # Mock that image files don't exist initially
            with patch.object(Path, 'exists', return_value=False):
                task = MockCameraTask(mock_config, mock_strategy="synthetic")

                # Should create sample images if they don't exist
                assert mock_create.call_count == 3

                # Check that sample image names are correct
                calls = [call[0][1] for call in mock_create.call_args_list]
                expected_names = ["test_landing_1.jpg", "test_landing_2.jpg", "test_aerial_view.jpg"]
                assert all(name in calls for name in expected_names)

    @pytest.mark.asyncio
    async def test_capture_synthetic_success(self, mock_config, temp_dir):
        """Test successful synthetic image capture."""
        task = MockCameraTask(mock_config, mock_strategy="synthetic")

        # Mock PIL operations
        with patch('PIL.Image.new') as mock_image_new, \
             patch('PIL.ImageDraw.Draw') as mock_draw:

            mock_img = MagicMock()
            mock_image_new.return_value = mock_img
            mock_drawer = MagicMock()
            mock_draw.return_value = mock_drawer

            test_file = temp_dir / "test_synthetic.jpg"
            cam_config = task.cameras[0]

            result = await task._capture_synthetic(cam_config, str(test_file))

            assert result is True
            mock_img.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_synthetic_failure(self, mock_config, temp_dir):
        """Test synthetic image capture failure."""
        task = MockCameraTask(mock_config, mock_strategy="synthetic")

        # Mock PIL to raise exception
        with patch('PIL.Image.new', side_effect=Exception("PIL error")):
            test_file = temp_dir / "test_synthetic.jpg"
            cam_config = task.cameras[0]

            result = await task._capture_synthetic(cam_config, str(test_file))

            assert result is False

    @pytest.mark.asyncio
    async def test_capture_opencv_success(self, mock_config, temp_dir):
        """Test successful OpenCV capture."""
        task = MockCameraTask(mock_config, mock_strategy="opencv")

        with patch('cv2.VideoCapture') as mock_cv2_capture, \
             patch('cv2.imwrite', return_value=True) as mock_imwrite:

            # Mock successful camera operations
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, "mock_frame")
            mock_cv2_capture.return_value = mock_cap

            test_file = temp_dir / "test_opencv.jpg"
            cam_config = task.cameras[0]

            result = await task._capture_opencv(cam_config, str(test_file))

            assert result is True
            mock_imwrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_opencv_camera_not_opened(self, mock_config, temp_dir):
        """Test OpenCV capture when camera cannot be opened."""
        task = MockCameraTask(mock_config, mock_strategy="opencv")

        with patch('cv2.VideoCapture') as mock_cv2_capture:
            # Mock camera that fails to open
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2_capture.return_value = mock_cap

            test_file = temp_dir / "test_opencv.jpg"
            cam_config = task.cameras[0]

            result = await task._capture_opencv(cam_config, str(test_file))

            assert result is False

    @pytest.mark.asyncio
    async def test_capture_opencv_read_failure(self, mock_config, temp_dir):
        """Test OpenCV capture when frame read fails."""
        task = MockCameraTask(mock_config, mock_strategy="opencv")

        with patch('cv2.VideoCapture') as mock_cv2_capture:
            # Mock camera that opens but fails to read
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (False, None)
            mock_cv2_capture.return_value = mock_cap
            task.opencv_cameras[0] = mock_cap  # Add to cache

            test_file = temp_dir / "test_opencv.jpg"
            cam_config = task.cameras[0]

            result = await task._capture_opencv(cam_config, str(test_file))

            assert result is False

    def test_rotation_handling(self, mock_config):
        """Test that rotation values are handled correctly."""
        config_with_rotation = mock_config.model_copy()
        config_with_rotation.tasks[0].parameters.rotation = 90

        task = MockCameraTask(config_with_rotation, mock_strategy="synthetic")
        assert task.rotation == 90

    def test_image_counter_increment(self, mock_config):
        """Test that image counter increments correctly."""
        task = MockCameraTask(mock_config, mock_strategy="synthetic")
        initial_counter = task.image_counter

        # Simulate some captures
        task.image_counter += 1
        task.image_counter += 1

        assert task.image_counter == initial_counter + 2

    def test_cleanup_opencv_cameras(self, mock_config):
        """Test that OpenCV cameras are properly cleaned up."""
        task = MockCameraTask(mock_config, mock_strategy="opencv")

        # Add mock cameras to cache
        mock_cap1 = MagicMock()
        mock_cap2 = MagicMock()
        task.opencv_cameras[0] = mock_cap1
        task.opencv_cameras[1] = mock_cap2

        # Trigger cleanup
        task.__del__()

        # Verify all cameras were released
        mock_cap1.release.assert_called_once()
        mock_cap2.release.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
