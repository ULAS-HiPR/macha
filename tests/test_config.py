import pytest
import sys
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from pydantic import ValidationError

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    MachaConfig, AppConfig, LoggingConfig, DatabaseConfig, TaskConfig,
    CameraParameters, CameraConfig, CameraResolution,
    BarometerParameters, ImuParameters, AiParameters,
    LogLevel, LogColors, LogFileConfig, LogConsoleConfig,
    load_config, get_task_config, get_camera_tasks, get_enabled_tasks
)


@pytest.fixture
def valid_config_dict():
    """Create a valid configuration dictionary."""
    return {
        "app": {
            "name": "test_app",
            "debug": True
        },
        "logging": {
            "level": "INFO",
            "file": {
                "path": "logs/test.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "console": {
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white"
                }
            }
        },
        "db": {
            "filename": "test.db",
            "connection_string": "sqlite+aiosqlite:///test.db",
            "overwrite": False
        },
        "tasks": [
            {
                "name": "test_camera",
                "class": "CameraTask",
                "frequency": 10,
                "enabled": True,
                "parameters": {
                    "cameras": [
                        {
                            "port": 0,
                            "name": "cam0",
                            "output_folder": "images/cam0"
                        }
                    ],
                    "image_format": "jpg",
                    "resolution": {
                        "width": 640,
                        "height": 480
                    },
                    "quality": 95,
                    "rotation": 0,
                    "capture_timeout": 10,
                    "retry_attempts": 3
                }
            },
            {
                "name": "test_metrics",
                "class": "MetricsTask",
                "frequency": 60,
                "enabled": True
            }
        ]
    }


@pytest.fixture
def temp_config_file(valid_config_dict):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config_dict, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLogLevel:
    """Test LogLevel enum."""

    def test_valid_log_levels(self):
        """Test valid log level values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_log_level_enum_values(self):
        """Test that all expected log levels are present."""
        expected_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        actual_levels = {level.value for level in LogLevel}
        assert actual_levels == expected_levels


class TestAppConfig:
    """Test AppConfig model."""

    def test_valid_app_config(self):
        """Test valid app configuration."""
        config = AppConfig(name="test_app", debug=True)
        assert config.name == "test_app"
        assert config.debug is True

    def test_app_config_defaults(self):
        """Test app configuration defaults."""
        config = AppConfig(name="test_app")
        assert config.name == "test_app"
        assert config.debug is False

    def test_app_config_required_name(self):
        """Test that app name is required."""
        with pytest.raises(ValidationError):
            AppConfig()


class TestLoggingConfiguration:
    """Test logging-related configuration models."""

    def test_log_colors_defaults(self):
        """Test LogColors default values."""
        colors = LogColors()
        assert colors.DEBUG == "cyan"
        assert colors.INFO == "green"
        assert colors.WARNING == "yellow"
        assert colors.ERROR == "red"
        assert colors.CRITICAL == "red,bg_white"

    def test_log_colors_custom(self):
        """Test LogColors with custom values."""
        colors = LogColors(
            DEBUG="blue",
            INFO="white",
            WARNING="orange",
            ERROR="magenta",
            CRITICAL="red,bg_yellow"
        )
        assert colors.DEBUG == "blue"
        assert colors.INFO == "white"
        assert colors.WARNING == "orange"
        assert colors.ERROR == "magenta"
        assert colors.CRITICAL == "red,bg_yellow"

    def test_log_file_config(self):
        """Test LogFileConfig model."""
        config = LogFileConfig(
            path="logs/app.log",
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        assert config.path == "logs/app.log"
        assert config.format == "%(asctime)s - %(levelname)s - %(message)s"

    def test_log_file_config_defaults(self):
        """Test LogFileConfig defaults."""
        config = LogFileConfig(path="test.log")
        assert config.path == "test.log"
        assert "%(asctime)s" in config.format
        assert "%(levelname)s" in config.format

    def test_log_console_config_defaults(self):
        """Test LogConsoleConfig defaults."""
        config = LogConsoleConfig()
        assert "%(log_color)s" in config.format
        assert isinstance(config.colors, LogColors)

    def test_logging_config_complete(self):
        """Test complete LoggingConfig."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file=LogFileConfig(path="test.log"),
            console=LogConsoleConfig()
        )
        assert config.level == LogLevel.DEBUG
        assert config.file.path == "test.log"
        assert isinstance(config.console, LogConsoleConfig)


class TestDatabaseConfig:
    """Test DatabaseConfig model."""

    def test_valid_database_config(self):
        """Test valid database configuration."""
        config = DatabaseConfig(
            filename="app.db",
            connection_string="sqlite+aiosqlite:///app.db",
            overwrite=True
        )
        assert config.filename == "app.db"
        assert config.connection_string == "sqlite+aiosqlite:///app.db"
        assert config.overwrite is True

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        config = DatabaseConfig(
            filename="test.db",
            connection_string="sqlite:///test.db"
        )
        assert config.overwrite is False


class TestCameraConfiguration:
    """Test camera-related configuration models."""

    def test_camera_resolution(self):
        """Test CameraResolution model."""
        resolution = CameraResolution(width=1920, height=1080)
        assert resolution.width == 1920
        assert resolution.height == 1080

    def test_camera_resolution_validation(self):
        """Test CameraResolution validation."""
        # Valid ranges
        CameraResolution(width=320, height=240)
        CameraResolution(width=4096, height=3072)

        # Invalid ranges
        with pytest.raises(ValidationError):
            CameraResolution(width=100, height=240)  # Width too small

        with pytest.raises(ValidationError):
            CameraResolution(width=320, height=100)  # Height too small

        with pytest.raises(ValidationError):
            CameraResolution(width=5000, height=1080)  # Width too large

    def test_camera_config(self):
        """Test CameraConfig model."""
        config = CameraConfig(
            port=0,
            name="main_camera",
            output_folder="/path/to/images"
        )
        assert config.port == 0
        assert config.name == "main_camera"
        assert config.output_folder == "/path/to/images"

    def test_camera_config_validation(self):
        """Test CameraConfig validation."""
        # Valid port range
        CameraConfig(port=0, name="cam0", output_folder="images")
        CameraConfig(port=3, name="cam3", output_folder="images")

        # Invalid port range
        with pytest.raises(ValidationError):
            CameraConfig(port=-1, name="cam", output_folder="images")

        with pytest.raises(ValidationError):
            CameraConfig(port=4, name="cam", output_folder="images")

        # Empty name
        with pytest.raises(ValidationError):
            CameraConfig(port=0, name="", output_folder="images")

        # Empty output folder
        with pytest.raises(ValidationError):
            CameraConfig(port=0, name="cam", output_folder="")

    def test_camera_parameters_complete(self):
        """Test complete CameraParameters."""
        params = CameraParameters(
            cameras=[
                CameraConfig(port=0, name="cam0", output_folder="images/cam0"),
                CameraConfig(port=1, name="cam1", output_folder="images/cam1")
            ],
            image_format="png",
            quality=85,
            resolution=CameraResolution(width=1280, height=720),
            rotation=90,
            capture_timeout=15,
            retry_attempts=5
        )
        assert len(params.cameras) == 2
        assert params.image_format == "png"
        assert params.quality == 85
        assert params.rotation == 90

    def test_camera_parameters_defaults(self):
        """Test CameraParameters defaults."""
        params = CameraParameters(
            cameras=[CameraConfig(port=0, name="cam0", output_folder="images")],
            resolution=CameraResolution(width=640, height=480)
        )
        assert params.image_format == "jpg"
        assert params.quality == 95
        assert params.rotation == 0
        assert params.capture_timeout == 10
        assert params.retry_attempts == 3

    def test_camera_parameters_validation(self):
        """Test CameraParameters validation."""
        base_config = {
            "cameras": [CameraConfig(port=0, name="cam0", output_folder="images")],
            "resolution": CameraResolution(width=640, height=480)
        }

        # Invalid image format
        with pytest.raises(ValidationError):
            CameraParameters(**base_config, image_format="gif")

        # Invalid quality
        with pytest.raises(ValidationError):
            CameraParameters(**base_config, quality=0)

        with pytest.raises(ValidationError):
            CameraParameters(**base_config, quality=101)

        # Invalid rotation
        with pytest.raises(ValidationError):
            CameraParameters(**base_config, rotation=45)

        # Duplicate camera ports
        with pytest.raises(ValidationError):
            CameraParameters(
                cameras=[
                    CameraConfig(port=0, name="cam0", output_folder="images"),
                    CameraConfig(port=0, name="cam1", output_folder="images")
                ],
                resolution=CameraResolution(width=640, height=480)
            )

        # Duplicate camera names
        with pytest.raises(ValidationError):
            CameraParameters(
                cameras=[
                    CameraConfig(port=0, name="camera", output_folder="images"),
                    CameraConfig(port=1, name="camera", output_folder="images")
                ],
                resolution=CameraResolution(width=640, height=480)
            )


class TestSensorConfiguration:
    """Test sensor configuration models."""

    def test_barometer_parameters(self):
        """Test BarometerParameters model."""
        params = BarometerParameters(
            i2c_bus=1,
            address=0x77,
            sea_level_pressure=1013.25
        )
        assert params.i2c_bus == 1
        assert params.address == 0x77
        assert params.sea_level_pressure == 1013.25

    def test_barometer_parameters_defaults(self):
        """Test BarometerParameters defaults."""
        params = BarometerParameters()
        assert params.i2c_bus == 1
        assert params.address == 0x77
        assert params.sea_level_pressure == 1013.25

    def test_barometer_parameters_validation(self):
        """Test BarometerParameters validation."""
        # Valid addresses
        BarometerParameters(address=0x76)
        BarometerParameters(address=0x77)

        # Invalid addresses
        with pytest.raises(ValidationError):
            BarometerParameters(address=0x75)

        with pytest.raises(ValidationError):
            BarometerParameters(address=0x78)

        # Invalid pressure
        with pytest.raises(ValidationError):
            BarometerParameters(sea_level_pressure=0)

        with pytest.raises(ValidationError):
            BarometerParameters(sea_level_pressure=-100)

    def test_imu_parameters(self):
        """Test ImuParameters model."""
        params = ImuParameters(
            i2c_bus=1,
            address=0x6A,
            accel_range="8G",
            gyro_range="1000DPS"
        )
        assert params.i2c_bus == 1
        assert params.address == 0x6A
        assert params.accel_range == "8G"
        assert params.gyro_range == "1000DPS"

    def test_imu_parameters_defaults(self):
        """Test ImuParameters defaults."""
        params = ImuParameters()
        assert params.i2c_bus == 1
        assert params.address == 0x6A
        assert params.accel_range == "4G"
        assert params.gyro_range == "500DPS"

    def test_imu_parameters_validation(self):
        """Test ImuParameters validation."""
        # Valid addresses
        ImuParameters(address=0x6A)
        ImuParameters(address=0x6B)

        # Invalid addresses
        with pytest.raises(ValidationError):
            ImuParameters(address=0x69)

        with pytest.raises(ValidationError):
            ImuParameters(address=0x6C)

        # Valid accel ranges
        for range_val in ["2G", "4G", "8G", "16G"]:
            ImuParameters(accel_range=range_val)

        # Invalid accel range
        with pytest.raises(ValidationError):
            ImuParameters(accel_range="32G")

        # Valid gyro ranges
        for range_val in ["125DPS", "250DPS", "500DPS", "1000DPS", "2000DPS"]:
            ImuParameters(gyro_range=range_val)

        # Invalid gyro range
        with pytest.raises(ValidationError):
            ImuParameters(gyro_range="4000DPS")


class TestAiConfiguration:
    """Test AI configuration models."""

    def test_ai_parameters_basic(self):
        """Test basic AiParameters model."""
        params = AiParameters(
            model_path="model.tflite",
            model_name="test_model"
        )
        assert params.model_path == "model.tflite"
        assert params.model_name == "test_model"
        assert params.model_version == "1.0.0"
        assert params.use_coral_tpu is True

    def test_ai_parameters_complete(self):
        """Test complete AiParameters configuration."""
        params = AiParameters(
            model_path="models/segmentation.tflite",
            model_name="landing_segmentation",
            model_version="2.1.0",
            use_coral_tpu=False,
            output_folder="ai_outputs",
            confidence_threshold=0.8,
            max_queue_size=100,
            processing_timeout=30,
            max_retries=5,
            output_format="png",
            save_confidence_overlay=False,
            class_names=["background", "safe", "unsafe", "unknown"],
            class_colors={
                "background": [0, 0, 0],
                "safe": [0, 255, 0],
                "unsafe": [255, 0, 0],
                "unknown": [128, 128, 128]
            }
        )
        assert params.model_version == "2.1.0"
        assert params.confidence_threshold == 0.8
        assert params.max_queue_size == 100
        assert len(params.class_names) == 4
        assert params.class_colors["safe"] == [0, 255, 0]

    def test_ai_parameters_defaults(self):
        """Test AiParameters defaults."""
        params = AiParameters(
            model_path="test.tflite",
            model_name="test"
        )
        assert params.confidence_threshold == 0.5
        assert params.max_queue_size == 50
        assert params.processing_timeout == 10
        assert params.max_retries == 2
        assert params.output_format == "png"
        assert len(params.class_names) == 3

    def test_ai_parameters_validation(self):
        """Test AiParameters validation."""
        base_params = {
            "model_path": "test.tflite",
            "model_name": "test"
        }

        # Invalid confidence threshold
        with pytest.raises(ValidationError):
            AiParameters(**base_params, confidence_threshold=1.5)

        with pytest.raises(ValidationError):
            AiParameters(**base_params, confidence_threshold=-0.1)

        # Invalid max_queue_size
        with pytest.raises(ValidationError):
            AiParameters(**base_params, max_queue_size=0)

        # Invalid processing_timeout
        with pytest.raises(ValidationError):
            AiParameters(**base_params, processing_timeout=0)

        with pytest.raises(ValidationError):
            AiParameters(**base_params, processing_timeout=61)

        # Invalid output_format
        with pytest.raises(ValidationError):
            AiParameters(**base_params, output_format="gif")

        # Invalid class colors
        with pytest.raises(ValidationError):
            AiParameters(
                **base_params,
                class_names=["bg", "safe"],
                class_colors={"bg": [300, 0, 0]}  # > 255
            )

        with pytest.raises(ValidationError):
            AiParameters(
                **base_params,
                class_names=["bg", "safe"],
                class_colors={"bg": [0, 0]}  # Wrong length
            )

    def test_ai_parameters_class_color_validation(self):
        """Test class color validation with class names."""
        # Missing color for class
        with pytest.raises(ValidationError):
            AiParameters(
                model_path="test.tflite",
                model_name="test",
                class_names=["bg", "safe", "unsafe"],
                class_colors={"bg": [0, 0, 0], "safe": [0, 255, 0]}  # Missing unsafe
            )


class TestTaskConfiguration:
    """Test TaskConfig model."""

    def test_task_config_basic(self):
        """Test basic TaskConfig."""
        config = TaskConfig(
            name="test_task",
            frequency=60,
            enabled=True,
            **{'class': 'MetricsTask'}
        )
        assert config.name == "test_task"
        assert config.class_name == "MetricsTask"
        assert config.frequency == 60
        assert config.enabled is True

    def test_task_config_with_parameters(self):
        """Test TaskConfig with parameters."""
        camera_params = CameraParameters(
            cameras=[CameraConfig(port=0, name="cam0", output_folder="images")],
            resolution=CameraResolution(width=640, height=480)
        )

        config = TaskConfig(
            name="camera_task",
            frequency=10,
            enabled=True,
            parameters=camera_params,
            **{'class': 'CameraTask'}
        )
        assert isinstance(config.parameters, CameraParameters)

    def test_task_config_defaults(self):
        """Test TaskConfig defaults."""
        config = TaskConfig(
            name="test_task",
            frequency=60,
            **{'class': 'MetricsTask'}
        )
        assert config.enabled is True
        assert config.parameters is None

    def test_task_config_validation(self):
        """Test TaskConfig validation."""
        # Invalid frequency
        with pytest.raises(ValidationError):
            TaskConfig(name="test", frequency=0, **{'class': 'MetricsTask'})

        # Invalid task name
        with pytest.raises(ValidationError):
            TaskConfig(name="", frequency=60, **{'class': 'MetricsTask'})

        with pytest.raises(ValidationError):
            TaskConfig(name="task with spaces", frequency=60, **{'class': 'MetricsTask'})

        with pytest.raises(ValidationError):
            TaskConfig(name="task@#$%", frequency=60, **{'class': 'MetricsTask'})

    def test_task_config_parameter_validation(self):
        """Test TaskConfig parameter validation by class name."""
        # Camera task with camera parameters
        camera_params_dict = {
            "cameras": [{"port": 0, "name": "cam0", "output_folder": "images"}],
            "resolution": {"width": 640, "height": 480}
        }

        config = TaskConfig(
            name="camera_task",
            frequency=10,
            parameters=camera_params_dict,
            **{'class': 'CameraTask'}
        )
        assert isinstance(config.parameters, CameraParameters)

        # Baro task with baro parameters
        baro_params_dict = {
            "i2c_bus": 1,
            "address": 0x77,
            "sea_level_pressure": 1013.25
        }

        config = TaskConfig(
            name="baro_task",
            frequency=30,
            parameters=baro_params_dict,
            **{'class': 'BaroTask'}
        )
        assert isinstance(config.parameters, BarometerParameters)


class TestMachaConfig:
    """Test MachaConfig main configuration model."""

    def test_macha_config_complete(self, valid_config_dict):
        """Test complete MachaConfig."""
        config = MachaConfig(**valid_config_dict)

        assert config.app.name == "test_app"
        assert config.logging.level == LogLevel.INFO
        assert config.db.filename == "test.db"
        assert len(config.tasks) == 2

    def test_macha_config_validation(self):
        """Test MachaConfig validation."""
        base_config = {
            "app": {"name": "test", "debug": True},
            "logging": {"level": "INFO", "file": {"path": "test.log"}},
            "db": {"filename": "test.db", "connection_string": "sqlite:///test.db"},
            "tasks": []
        }

        # Empty tasks list
        with pytest.raises(ValidationError):
            MachaConfig(**base_config)

        # Duplicate task names
        base_config["tasks"] = [
            {"name": "task1", "class": "MetricsTask", "frequency": 60},
            {"name": "task1", "class": "CameraTask", "frequency": 30, "parameters": {
                "cameras": [{"port": 0, "name": "cam0", "output_folder": "images"}],
                "resolution": {"width": 640, "height": 480}
            }}
        ]

        with pytest.raises(ValidationError):
            MachaConfig(**base_config)

    def test_macha_config_ai_frequency_validation(self):
        """Test AI task frequency validation."""
        config_dict = {
            "app": {"name": "test", "debug": True},
            "logging": {"level": "INFO", "file": {"path": "test.log"}},
            "db": {"filename": "test.db", "connection_string": "sqlite:///test.db"},
            "tasks": [
                {
                    "name": "camera_task",
                    "class": "CameraTask",
                    "frequency": 10,
                    "enabled": True,
                    "parameters": {
                        "cameras": [{"port": 0, "name": "cam0", "output_folder": "images"}],
                        "resolution": {"width": 640, "height": 480}
                    }
                },
                {
                    "name": "ai_task",
                    "class": "AiTask",
                    "frequency": 5,  # Faster than camera - should fail
                    "enabled": True,
                    "parameters": {
                        "model_path": "test.tflite",
                        "model_name": "test"
                    }
                }
            ]
        }

        with pytest.raises(ValidationError, match="AI task.*frequency.*cannot be faster"):
            MachaConfig(**config_dict)

    def test_macha_config_ai_frequency_divisible(self):
        """Test AI task frequency must be divisible by camera frequency."""
        config_dict = {
            "app": {"name": "test", "debug": True},
            "logging": {"level": "INFO", "file": {"path": "test.log"}},
            "db": {"filename": "test.db", "connection_string": "sqlite:///test.db"},
            "tasks": [
                {
                    "name": "camera_task",
                    "class": "CameraTask",
                    "frequency": 10,
                    "enabled": True,
                    "parameters": {
                        "cameras": [{"port": 0, "name": "cam0", "output_folder": "images"}],
                        "resolution": {"width": 640, "height": 480}
                    }
                },
                {
                    "name": "ai_task",
                    "class": "AiTask",
                    "frequency": 15,  # Not divisible by 10
                    "enabled": True,
                    "parameters": {
                        "model_path": "test.tflite",
                        "model_name": "test"
                    }
                }
            ]
        }

        with pytest.raises(ValidationError, match="AI task.*frequency.*must be divisible"):
            MachaConfig(**config_dict)

    @patch('pathlib.Path.mkdir')
    def test_macha_config_directory_creation(self, mock_mkdir, valid_config_dict):
        """Test that MachaConfig creates necessary directories."""
        config = MachaConfig(**valid_config_dict)

        # Should create camera output directories
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_macha_config_directory_creation_error(self, valid_config_dict):
        """Test MachaConfig directory creation error handling."""
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ValueError, match="Cannot create directory"):
                MachaConfig(**valid_config_dict)


class TestConfigurationLoading:
    """Test configuration loading functions."""

    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from YAML file."""
        config = load_config(temp_config_file)

        assert isinstance(config, MachaConfig)
        assert config.app.name == "test_app"
        assert len(config.tasks) == 2

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_config_invalid_structure(self):
        """Test loading configuration with invalid structure."""
        invalid_config = {
            "app": {"name": "test"},
            "logging": {"level": "INFO", "file": {"path": "test.log"}},
            "db": {"filename": "test.db", "connection_string": "sqlite:///test.db"},
            "tasks": []  # Empty tasks - invalid
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Configuration validation failed"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)

    @patch('config.Path')
    def test_load_config_default_path(self, mock_path):
        """Test loading configuration with default path."""
        mock_config_file = Mock()
        mock_config_file.exists.return_value = True
        mock_path.return_value = mock_config_file

        valid_config = {
            "app": {"name": "test", "debug": True},
            "logging": {"level": "INFO", "file": {"path": "test.log"}},
            "db": {"filename": "test.db", "connection_string": "sqlite:///test.db"},
            "tasks": [{"name": "test", "class": "MetricsTask", "frequency": 60}]
        }

        with patch('builtins.open', mock_open(read_data=yaml.dump(valid_config))):
            with patch('yaml.safe_load', return_value=valid_config):
                config = load_config()

        assert isinstance(config, MachaConfig)
        mock_path.assert_called_once_with("config.yaml")


class TestConfigurationHelpers:
    """Test configuration helper functions."""

    def test_get_task_config(self, temp_config_file):
        """Test getting specific task configuration."""
        config = load_config(temp_config_file)

        camera_config = get_task_config(config, "test_camera")
        assert camera_config is not None
        assert camera_config.name == "test_camera"
        assert camera_config.class_name == "CameraTask"

        metrics_config = get_task_config(config, "test_metrics")
        assert metrics_config is not None
        assert metrics_config.name == "test_metrics"

        # Non-existent task
        non_existent = get_task_config(config, "non_existent_task")
        assert non_existent is None
