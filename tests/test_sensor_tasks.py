import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig, BarometerParameters, ImuParameters
from baro_task import BaroTask
from imu_task import ImuTask


@pytest.fixture
def mock_engine():
    """Create a mock database engine."""
    engine = Mock(spec=AsyncEngine)

    # Mock connection and execute methods
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    engine.connect.return_value = mock_conn
    return engine


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def baro_config():
    """Create a test configuration with barometer task."""
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
                "name": "test_baro",
                "class": "BaroTask",
                "frequency": 30,
                "enabled": True,
                "parameters": {
                    "i2c_bus": 1,
                    "address": 0x77,
                    "sea_level_pressure": 1013.25
                }
            }
        ]
    )


@pytest.fixture
def imu_config():
    """Create a test configuration with IMU task."""
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
                "name": "test_imu",
                "class": "ImuTask",
                "frequency": 10,
                "enabled": True,
                "parameters": {
                    "i2c_bus": 1,
                    "address": 0x6A,
                    "accel_range": "4G",
                    "gyro_range": "500DPS"
                }
            }
        ]
    )


class TestBaroTask:
    """Test cases for BaroTask."""

    def test_baro_task_initialization(self, baro_config):
        """Test BaroTask initialization with valid config."""
        task = BaroTask(baro_config)
        assert task.name == "BaroTask"
        assert isinstance(task.parameters, BarometerParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x77
        assert task.parameters.sea_level_pressure == 1013.25

    def test_baro_task_default_parameters(self):
        """Test BaroTask with default parameters when no config provided."""
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
                    "name": "dummy_task",
                    "class": "BaroTask",
                    "frequency": 60,
                    "enabled": True,
                    "parameters": {
                        "i2c_bus": 1,
                        "address": 0x77,
                        "sea_level_pressure": 1013.25
                    }
                }
            ]
        )

        task = BaroTask(config)
        assert isinstance(task.parameters, BarometerParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x77

    @patch('baro_task.SENSOR_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_baro_task_sensor_unavailable(self, baro_config, mock_engine, mock_logger):
        """Test BaroTask when sensor libraries are not available."""
        task = BaroTask(baro_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Failed to initialize barometer sensor" in result["error"]
        mock_logger.error.assert_called()

    @patch('baro_task.SENSOR_AVAILABLE', True)
    @patch('baro_task.busio')
    @patch('baro_task.board')
    @patch('baro_task.adafruit_bmp3xx')
    @pytest.mark.asyncio
    async def test_baro_task_successful_execution(self, mock_bmp3xx, mock_board, mock_busio, baro_config, mock_engine, mock_logger):
        """Test successful barometer task execution."""
        # Mock sensor initialization
        mock_i2c = Mock()
        mock_busio.I2C.return_value = mock_i2c

        mock_sensor = Mock()
        mock_sensor.pressure = 1013.25
        mock_sensor.temperature = 25.5
        mock_bmp3xx.BMP3XX_I2C.return_value = mock_sensor

        task = BaroTask(baro_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["pressure_hpa"] == 1013.25
        assert result["data"]["temperature_celsius"] == 25.5
        assert result["data"]["altitude_meters"] is not None

    @patch('baro_task.SENSOR_AVAILABLE', True)
    @patch('baro_task.busio')
    @patch('baro_task.board')
    @patch('baro_task.adafruit_bmp3xx')
    @pytest.mark.asyncio
    async def test_baro_task_sensor_read_failure(self, mock_bmp3xx, mock_board, mock_busio, baro_config, mock_engine, mock_logger):
        """Test barometer task when sensor reading fails."""
        # Mock sensor initialization
        mock_i2c = Mock()
        mock_busio.I2C.return_value = mock_i2c

        mock_sensor = Mock()
        mock_sensor.pressure = None
        mock_sensor.temperature = None
        mock_bmp3xx.BMP3XX_I2C.return_value = mock_sensor

        task = BaroTask(baro_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Failed to read valid data from barometer sensor" in result["error"]


class TestImuTask:
    """Test cases for ImuTask."""

    def test_imu_task_initialization(self, imu_config):
        """Test ImuTask initialization with valid config."""
        task = ImuTask(imu_config)
        assert task.name == "ImuTask"
        assert isinstance(task.parameters, ImuParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x6A
        assert task.parameters.accel_range == "4G"
        assert task.parameters.gyro_range == "500DPS"

    def test_imu_task_default_parameters(self):
        """Test ImuTask with default parameters when no config provided."""
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
                    "name": "dummy_task",
                    "class": "ImuTask",
                    "frequency": 60,
                    "enabled": True,
                    "parameters": {
                        "i2c_bus": 1,
                        "address": 0x6A,
                        "accel_range": "4G",
                        "gyro_range": "500DPS"
                    }
                }
            ]
        )

        task = ImuTask(config)
        assert isinstance(task.parameters, ImuParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x6A

    @patch('imu_task.SENSOR_AVAILABLE', False)
    @pytest.mark.asyncio
    async def test_imu_task_sensor_unavailable(self, imu_config, mock_engine, mock_logger):
        """Test ImuTask when sensor libraries are not available."""
        task = ImuTask(imu_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Failed to initialize IMU sensor" in result["error"]
        mock_logger.error.assert_called()

    @patch('imu_task.SENSOR_AVAILABLE', True)
    @patch('imu_task.busio')
    @patch('imu_task.board')
    @patch('imu_task.adafruit_lsm6ds')
    @patch('imu_task.LSM6DSOX')
    @patch('imu_task.LSM6DS33')
    @patch('imu_task.LSM6DSO32')
    @patch('imu_task.LSM6DS3TRC')
    @pytest.mark.asyncio
    async def test_imu_task_successful_execution(self, mock_lsm6ds3trc, mock_lsm6dso32, mock_lsm6ds33, mock_lsm6dsox, mock_lsm6ds, mock_board, mock_busio, imu_config, mock_engine, mock_logger):
        """Test successful IMU task execution."""
        # Mock sensor initialization
        mock_i2c = Mock()
        mock_busio.I2C.return_value = mock_i2c

        mock_sensor = Mock()
        mock_sensor.acceleration = (0.1, 0.2, 9.8)
        mock_sensor.gyro = (0.01, 0.02, 0.03)
        mock_sensor.temperature = 26.5

        # Add configuration attributes that ImuTask tries to set during initialization
        mock_sensor.accelerometer_range = None
        mock_sensor.gyro_range = None
        mock_sensor.accelerometer_data_rate = None
        mock_sensor.gyro_data_rate = None

        # Mock the LSM6DSOX constructor to return our mock sensor
        mock_lsm6dsox.return_value = mock_sensor

        # Make other sensor classes fail so only LSM6DSOX succeeds
        mock_lsm6ds33.side_effect = Exception("Sensor not found")
        mock_lsm6ds3trc.side_effect = Exception("Sensor not found")
        mock_lsm6dso32.side_effect = Exception("Sensor not found")

        # Mock range constants - make attributes available
        mock_lsm6ds.Range = Mock()
        mock_lsm6ds.Range.RANGE_4G = "4G"
        mock_lsm6ds.GyroRange = Mock()
        mock_lsm6ds.GyroRange.RANGE_500_DPS = "500DPS"
        mock_lsm6ds.Rate = Mock()
        mock_lsm6ds.Rate.RATE_104_HZ = "104HZ"

        task = ImuTask(imu_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["accel_x"] == 0.1
        assert result["data"]["accel_y"] == 0.2
        assert result["data"]["accel_z"] == 9.8
        assert result["data"]["gyro_x"] == 0.01
        assert result["data"]["gyro_y"] == 0.02
        assert result["data"]["gyro_z"] == 0.03
        assert result["data"]["temperature_celsius"] == 26.5
    @patch('imu_task.SENSOR_AVAILABLE', True)
    @patch('imu_task.busio')
    @patch('imu_task.board')
    @patch('imu_task.adafruit_lsm6ds')
    @patch('imu_task.LSM6DSOX')
    @patch('imu_task.LSM6DS33')
    @patch('imu_task.LSM6DSO32')
    @patch('imu_task.LSM6DS3TRC')
    @pytest.mark.asyncio
    async def test_imu_task_sensor_read_failure(self, mock_lsm6ds3trc, mock_lsm6dso32, mock_lsm6ds33, mock_lsm6dsox, mock_lsm6ds, mock_board, mock_busio, imu_config, mock_engine, mock_logger):
        """Test IMU task when sensor reading fails."""
        # Mock sensor initialization
        mock_i2c = Mock()
        mock_busio.I2C.return_value = mock_i2c

        mock_sensor = Mock()
        mock_sensor.acceleration = (None, None, None)
        mock_sensor.gyro = (None, None, None)
        mock_sensor.temperature = None

        # Mock the LSM6DSOX constructor to return our mock sensor
        mock_lsm6dsox.return_value = mock_sensor

        # Mock range constants - make attributes available
        mock_lsm6ds.Range = Mock()
        mock_lsm6ds.Range.RANGE_4G = "4G"
        mock_lsm6ds.GyroRange = Mock()
        mock_lsm6ds.GyroRange.RANGE_500_DPS = "500DPS"
        mock_lsm6ds.Rate = Mock()
        mock_lsm6ds.Rate.RATE_104_HZ = "104HZ"

        task = ImuTask(imu_config)
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Failed to read valid data from IMU sensor" in result["error"]


class TestSensorTaskParameters:
    """Test sensor task parameter validation."""

    def test_barometer_parameters_validation(self):
        """Test BarometerParameters validation."""
        # Valid parameters
        params = BarometerParameters(
            i2c_bus=1,
            address=0x77,
            sea_level_pressure=1013.25
        )
        assert params.i2c_bus == 1
        assert params.address == 0x77
        assert params.sea_level_pressure == 1013.25

        # Test invalid address
        with pytest.raises(ValueError):
            BarometerParameters(address=0x50)

    def test_imu_parameters_validation(self):
        """Test ImuParameters validation."""
        # Valid parameters
        params = ImuParameters(
            i2c_bus=1,
            address=0x6A,
            accel_range="4G",
            gyro_range="500DPS"
        )
        assert params.i2c_bus == 1
        assert params.address == 0x6A
        assert params.accel_range == "4G"
        assert params.gyro_range == "500DPS"

        # Test invalid address
        with pytest.raises(ValueError):
            ImuParameters(address=0x50)

        # Test invalid ranges
        with pytest.raises(ValueError):
            ImuParameters(accel_range="32G")

        with pytest.raises(ValueError):
            ImuParameters(gyro_range="3000DPS")
