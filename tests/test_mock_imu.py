import pytest
import asyncio
import sys
import os
import time
import math
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig, ImuParameters
from mock_imu_task import MockImuTask


@pytest.fixture
def mock_engine():
    """Create a mock database engine."""
    engine = Mock(spec=AsyncEngine)

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
                "class": "MockImuTask",
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


class TestMockImuTask:
    """Unit tests for MockImuTask."""

    def test_initialization(self, imu_config):
        """Test MockImuTask initialization."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        assert task.mock_strategy == "stationary"
        assert isinstance(task.parameters, ImuParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x6A
        assert task.parameters.accel_range == "4G"
        assert task.parameters.gyro_range == "500DPS"
        assert task.gravity == 9.81

    def test_initialization_with_default_parameters(self):
        """Test MockImuTask with minimal configuration."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "dummy_task",
                    "class": "MockImuTask",
                    "frequency": 60,
                    "enabled": True
                }
            ]
        )

        task = MockImuTask(config, mock_strategy="static")
        assert isinstance(task.parameters, ImuParameters)
        assert task.mock_strategy == "static"

    def test_strategy_determination_auto(self, imu_config):
        """Test automatic strategy determination."""
        task = MockImuTask(imu_config, mock_strategy="auto")
        assert task.mock_strategy == "stationary"

    def test_strategy_determination_explicit(self, imu_config):
        """Test explicit strategy setting."""
        strategies = ["stationary", "flight", "vibration", "static", "turbulence"]

        for strategy in strategies:
            task = MockImuTask(imu_config, mock_strategy=strategy)
            assert task.mock_strategy == strategy

    def test_generate_stationary_data(self, imu_config):
        """Test stationary data generation."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        data = task._generate_stationary_data()

        assert "accel_x" in data
        assert "accel_y" in data
        assert "accel_z" in data
        assert "gyro_x" in data
        assert "gyro_y" in data
        assert "gyro_z" in data
        assert "temperature_celsius" in data

        # Check data types and reasonable ranges for stationary
        assert isinstance(data["accel_x"], float)
        assert isinstance(data["accel_y"], float)
        assert isinstance(data["accel_z"], float)
        assert isinstance(data["gyro_x"], float)
        assert isinstance(data["gyro_y"], float)
        assert isinstance(data["gyro_z"], float)
        assert isinstance(data["temperature_celsius"], float)

        # Z-axis should be close to gravity for stationary
        assert 9.0 < data["accel_z"] < 10.5
        # X and Y should be small for stationary
        assert abs(data["accel_x"]) < 1.0
        assert abs(data["accel_y"]) < 1.0
        # Gyro rates should be small for stationary
        assert abs(data["gyro_x"]) < 0.1
        assert abs(data["gyro_y"]) < 0.1
        assert abs(data["gyro_z"]) < 0.1

    def test_generate_flight_data(self, imu_config):
        """Test flight data generation."""
        task = MockImuTask(imu_config, mock_strategy="flight")

        # Test different points in flight cycle
        flight_phases_seen = set()
        for i in range(15):
            data = task._generate_flight_data()

            assert "accel_x" in data
            assert "accel_y" in data
            assert "accel_z" in data
            assert "gyro_x" in data
            assert "gyro_y" in data
            assert "gyro_z" in data
            assert "temperature_celsius" in data

            # Should have valid flight phase
            assert hasattr(task, 'flight_phase')
            assert task.flight_phase in ["ground", "takeoff", "climb", "cruise", "descent", "landing"]
            flight_phases_seen.add(task.flight_phase)

            # Advance time for next iteration
            task.start_time -= 20  # 20 seconds earlier

        # Should see multiple flight phases
        assert len(flight_phases_seen) >= 3

    def test_generate_static_data(self, imu_config):
        """Test static data generation."""
        task = MockImuTask(imu_config, mock_strategy="static")

        data1 = task._generate_static_data()
        data2 = task._generate_static_data()

        # Static data should be identical
        assert data1["accel_x"] == data2["accel_x"] == 0.0
        assert data1["accel_y"] == data2["accel_y"] == 0.0
        assert data1["accel_z"] == data2["accel_z"] == task.gravity
        assert data1["gyro_x"] == data2["gyro_x"] == 0.0
        assert data1["gyro_y"] == data2["gyro_y"] == 0.0
        assert data1["gyro_z"] == data2["gyro_z"] == 0.0

    def test_generate_vibration_data(self, imu_config):
        """Test vibration data generation."""
        task = MockImuTask(imu_config, mock_strategy="vibration")

        # Generate multiple samples to test vibration characteristics
        samples = [task._generate_vibration_data() for _ in range(10)]

        # Should have variability due to vibrations
        accel_x_values = [s["accel_x"] for s in samples]
        accel_y_values = [s["accel_y"] for s in samples]
        gyro_values = [s["gyro_x"] for s in samples]

        # Should have some variation due to vibrations
        assert len(set([round(v, 2) for v in accel_x_values])) > 3
        assert len(set([round(v, 2) for v in accel_y_values])) > 3
        assert len(set([round(v, 3) for v in gyro_values])) > 3

        # Temperature should be elevated due to vibration heating
        temperatures = [s["temperature_celsius"] for s in samples]
        assert all(t > task.temp_base for t in temperatures)

    def test_generate_turbulence_data(self, imu_config):
        """Test turbulence data generation."""
        task = MockImuTask(imu_config, mock_strategy="turbulence")

        # Generate multiple samples
        samples = [task._generate_turbulence_data() for _ in range(20)]

        # Should have high variability
        accel_x_values = [s["accel_x"] for s in samples]
        accel_y_values = [s["accel_y"] for s in samples]
        accel_z_values = [s["accel_z"] for s in samples]

        # High variability expected in turbulence
        assert max(accel_x_values) - min(accel_x_values) > 2.0
        assert max(accel_y_values) - min(accel_y_values) > 2.0
        assert max(accel_z_values) - min(accel_z_values) > 2.0

        # Z should still be around gravity on average
        avg_z = sum(accel_z_values) / len(accel_z_values)
        assert 8.0 < avg_z < 12.0

    def test_read_mock_sensor_data_all_strategies(self, imu_config, mock_logger):
        """Test reading mock sensor data for all strategies."""
        strategies = ["stationary", "flight", "vibration", "turbulence", "static"]

        for strategy in strategies:
            task = MockImuTask(imu_config, mock_strategy=strategy)
            data = task._read_mock_sensor_data(mock_logger)

            # Should return valid data structure
            expected_keys = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature_celsius"]
            assert all(key in data for key in expected_keys)
            assert all(isinstance(data[key], (int, float)) for key in expected_keys)

    def test_read_mock_sensor_data_error_handling(self, imu_config, mock_logger):
        """Test error handling in mock sensor data reading."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        # Mock an exception in data generation
        with patch.object(task, '_generate_stationary_data', side_effect=Exception("Test error")):
            data = task._read_mock_sensor_data(mock_logger)

            # Should return None values on error
            assert all(data[key] is None for key in data.keys())
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_successful(self, imu_config, mock_engine, mock_logger):
        """Test successful task execution."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is True
        assert result["error"] is None
        assert "data" in result
        assert "mock_strategy" in result
        assert result["mock_strategy"] == "stationary"

        # Check data structure
        data = result["data"]
        expected_keys = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "temperature_celsius"]
        assert all(key in data for key in expected_keys)

    @pytest.mark.asyncio
    async def test_execute_database_error(self, imu_config, mock_logger):
        """Test task execution with database error."""
        # Create mock engine that raises exception
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Database error")

        task = MockImuTask(imu_config, mock_strategy="static")
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Database error" in result["error"]
        assert "data" in result  # Should still have generated data

    @pytest.mark.asyncio
    async def test_execute_invalid_data(self, imu_config, mock_engine, mock_logger):
        """Test task execution when mock data generation fails."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        # Mock data generation to return None values
        with patch.object(task, '_read_mock_sensor_data', return_value={
            "accel_x": None, "accel_y": None, "accel_z": None,
            "gyro_x": None, "gyro_y": None, "gyro_z": None,
            "temperature_celsius": None
        }):
            result = await task.execute(mock_engine, mock_logger)

            assert result["success"] is False
            assert "Failed to generate valid mock IMU data" in result["error"]

    @pytest.mark.asyncio
    async def test_database_integration(self, imu_config, mock_engine, mock_logger):
        """Test database integration and SQL query structure."""
        task = MockImuTask(imu_config, mock_strategy="flight")

        await task.execute(mock_engine, mock_logger)

        # Verify database operations
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

        # Check SQL call structure
        call_args = mock_conn.execute.call_args
        sql_query = call_args[0][0].text  # Get the actual SQL text from TextClause
        assert "INSERT INTO imu_readings" in sql_query

    def test_reset_simulation(self, imu_config):
        """Test simulation reset functionality."""
        task = MockImuTask(imu_config, mock_strategy="flight")

        # Advance simulation
        original_start_time = task.start_time
        task.velocity = [10.0, 5.0, 2.0]
        task.angular_velocity = [0.1, 0.2, 0.05]
        task.flight_phase = "cruise"

        # Reset simulation
        task.reset_simulation()

        # Should reset to initial state
        assert task.start_time > original_start_time
        assert task.velocity == [0.0, 0.0, 0.0]
        assert task.angular_velocity == [0.0, 0.0, 0.0]
        assert task.flight_phase == "ground"

    def test_set_mock_strategy(self, imu_config):
        """Test changing mock strategy at runtime."""
        task = MockImuTask(imu_config, mock_strategy="stationary")
        original_start_time = task.start_time

        # Change strategy
        task.set_mock_strategy("flight")

        assert task.mock_strategy == "flight"
        assert task.start_time >= original_start_time  # Should reset time

    def test_get_simulation_state(self, imu_config):
        """Test getting simulation state for debugging."""
        task = MockImuTask(imu_config, mock_strategy="flight")
        task.flight_phase = "climb"
        task.velocity = [15.0, 0.0, 5.0]

        state = task.get_simulation_state()

        expected_keys = ["mock_strategy", "flight_phase", "velocity", "angular_velocity", "elapsed_time"]
        assert all(key in state for key in expected_keys)
        assert state["mock_strategy"] == "flight"
        assert state["flight_phase"] == "climb"
        assert isinstance(state["elapsed_time"], float)

    def test_flight_phase_progression(self, imu_config):
        """Test that flight phases progress correctly over time."""
        task = MockImuTask(imu_config, mock_strategy="flight")
        task.flight_duration = 100.0  # Short duration for testing

        phases_seen = set()

        # Simulate flight progression
        for i in range(20):
            task.start_time = time.time() - (i * 10)  # 10 second intervals
            task._generate_flight_data()
            phases_seen.add(task.flight_phase)

        # Should see multiple phases during progression
        assert len(phases_seen) >= 3
        assert "ground" in phases_seen

    def test_accelerometer_data_physics(self, imu_config):
        """Test that accelerometer data follows basic physics principles."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        # For stationary data, should see gravity primarily on Z-axis
        samples = [task._generate_stationary_data() for _ in range(10)]

        # Average Z acceleration should be close to gravity
        avg_z = sum(s["accel_z"] for s in samples) / len(samples)
        assert 9.5 < avg_z < 10.1

        # X and Y should be small and roughly centered on zero
        avg_x = sum(s["accel_x"] for s in samples) / len(samples)
        avg_y = sum(s["accel_y"] for s in samples) / len(samples)
        assert abs(avg_x) < 0.2
        assert abs(avg_y) < 0.2

    def test_gyroscope_data_ranges(self, imu_config):
        """Test gyroscope data stays within reasonable ranges."""
        strategies = ["stationary", "flight", "vibration", "turbulence"]

        for strategy in strategies:
            task = MockImuTask(imu_config, mock_strategy=strategy)

            # Generate multiple samples
            samples = [getattr(task, f'_generate_{strategy}_data')() for _ in range(10)]

            for sample in samples:
                # Gyro rates should be reasonable (not exceeding typical sensor ranges)
                assert -10.0 < sample["gyro_x"] < 10.0  # rad/s
                assert -10.0 < sample["gyro_y"] < 10.0
                assert -10.0 < sample["gyro_z"] < 10.0

    def test_temperature_variations_over_time(self, imu_config):
        """Test temperature variations over time."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        # Generate data over time
        temperatures = []
        for i in range(20):
            task.start_time = time.time() - (i * 1800)  # 30 minute intervals
            data = task._generate_stationary_data()
            temperatures.append(data["temperature_celsius"])

        # Should have variations but stay in reasonable range
        assert len(set([round(t, 1) for t in temperatures])) > 5  # Some variation
        assert all(15 < t < 50 for t in temperatures)  # Reasonable range for IMU sensor

    def test_vibration_frequency_characteristics(self, imu_config):
        """Test that vibration mode generates expected frequency content."""
        task = MockImuTask(imu_config, mock_strategy="vibration")

        # Generate samples at high rate to capture vibrations
        samples = []
        base_time = time.time()
        for i in range(100):
            task.start_time = base_time - (i * 0.01)  # 10ms intervals
            data = task._generate_vibration_data()
            samples.append(data)

        # Should have rapid variations characteristic of vibrations
        accel_x_values = [s["accel_x"] for s in samples]

        # Calculate approximate frequency content by counting zero crossings
        zero_crossings = 0
        for i in range(1, len(accel_x_values)):
            if accel_x_values[i] * accel_x_values[i-1] < 0:
                zero_crossings += 1

        # Should have many zero crossings indicating high frequency content
        assert zero_crossings > 10

    def test_sensor_noise_characteristics(self, imu_config):
        """Test that sensor noise is within expected bounds."""
        task = MockImuTask(imu_config, mock_strategy="stationary")

        # Generate many samples to analyze noise
        samples = [task._generate_stationary_data() for _ in range(100)]

        # Calculate standard deviations
        accel_x_values = [s["accel_x"] for s in samples]
        gyro_x_values = [s["gyro_x"] for s in samples]

        # Calculate std dev
        import statistics
        accel_std = statistics.stdev(accel_x_values)
        gyro_std = statistics.stdev(gyro_x_values)

        # Should have reasonable noise levels
        assert 0.01 < accel_std < 0.5  # Reasonable accelerometer noise
        assert 0.001 < gyro_std < 0.1  # Reasonable gyroscope noise


if __name__ == "__main__":
    pytest.main([__file__])
