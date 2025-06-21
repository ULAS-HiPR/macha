import pytest
import asyncio
import sys
import os
import time
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig, BarometerParameters
from mock_baro_task import MockBaroTask


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
                "class": "MockBaroTask",
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


class TestMockBaroTask:
    """Unit tests for MockBaroTask."""

    def test_initialization(self, baro_config):
        """Test MockBaroTask initialization."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        assert task.mock_strategy == "realistic"
        assert isinstance(task.parameters, BarometerParameters)
        assert task.parameters.i2c_bus == 1
        assert task.parameters.address == 0x77
        assert task.parameters.sea_level_pressure == 1013.25
        assert task.base_pressure == 1013.25

    def test_initialization_with_default_parameters(self):
        """Test MockBaroTask with minimal configuration."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "dummy_task",
                    "class": "MockBaroTask",
                    "frequency": 60,
                    "enabled": True
                }
            ]
        )

        task = MockBaroTask(config, mock_strategy="static")
        assert isinstance(task.parameters, BarometerParameters)
        assert task.mock_strategy == "static"

    def test_strategy_determination_auto(self, baro_config):
        """Test automatic strategy determination."""
        task = MockBaroTask(baro_config, mock_strategy="auto")
        assert task.mock_strategy == "realistic"

    def test_strategy_determination_explicit(self, baro_config):
        """Test explicit strategy setting."""
        strategies = ["realistic", "flight", "static", "noisy"]

        for strategy in strategies:
            task = MockBaroTask(baro_config, mock_strategy=strategy)
            assert task.mock_strategy == strategy

    def test_generate_realistic_data(self, baro_config, mock_logger):
        """Test realistic data generation."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        data = task._generate_realistic_data()

        assert "pressure_hpa" in data
        assert "temperature_celsius" in data
        assert "altitude_meters" in data

        # Check data types and reasonable ranges
        assert isinstance(data["pressure_hpa"], float)
        assert isinstance(data["temperature_celsius"], float)
        assert isinstance(data["altitude_meters"], float)

        # Pressure should be around sea level with variations
        assert 900 < data["pressure_hpa"] < 1100
        # Temperature should be reasonable
        assert -10 < data["temperature_celsius"] < 50
        # Altitude should be reasonable for pressure variations
        assert -200 < data["altitude_meters"] < 2000

    def test_generate_flight_data(self, baro_config, mock_logger):
        """Test flight data generation."""
        task = MockBaroTask(baro_config, mock_strategy="flight")

        # Test different points in flight cycle
        for i in range(10):
            data = task._generate_flight_data()

            assert "pressure_hpa" in data
            assert "temperature_celsius" in data
            assert "altitude_meters" in data

            # Should have valid flight phase
            assert hasattr(task, 'flight_phase')
            assert task.flight_phase in ["ground", "takeoff", "climb", "cruise", "descent", "landing"]

            # Altitude should be reasonable during flight phases
            if task.flight_phase in ["takeoff", "climb"]:
                assert data["altitude_meters"] >= -50  # Allow for some below-sea-level variation

            # Advance time slightly for next iteration
            task.start_time -= 30  # 30 seconds earlier

    def test_generate_static_data(self, baro_config, mock_logger):
        """Test static data generation."""
        task = MockBaroTask(baro_config, mock_strategy="static")

        data1 = task._generate_static_data()
        data2 = task._generate_static_data()

        # Static data should be consistent
        assert data1["pressure_hpa"] == data2["pressure_hpa"]
        assert data1["altitude_meters"] == data2["altitude_meters"]
        # Temperature might have small offset but should be close
        assert abs(data1["temperature_celsius"] - data2["temperature_celsius"]) < 0.1

    def test_generate_noisy_data(self, baro_config, mock_logger):
        """Test noisy data generation."""
        task = MockBaroTask(baro_config, mock_strategy="noisy")

        # Generate multiple samples to test noise characteristics
        samples = [task._generate_noisy_data() for _ in range(10)]

        pressures = [s["pressure_hpa"] for s in samples]
        temperatures = [s["temperature_celsius"] for s in samples]

        # Should have variability (not all the same)
        assert len(set(pressures)) > 5  # At least some variation
        assert len(set(temperatures)) > 5

        # Should still be in reasonable ranges despite noise
        assert all(800 < p < 1200 for p in pressures)
        assert all(-20 < t < 70 for t in temperatures)

    def test_read_mock_sensor_data_all_strategies(self, baro_config, mock_logger):
        """Test reading mock sensor data for all strategies."""
        strategies = ["realistic", "flight", "static", "noisy"]

        for strategy in strategies:
            task = MockBaroTask(baro_config, mock_strategy=strategy)
            data = task._read_mock_sensor_data(mock_logger)

            # Should return valid data structure
            expected_keys = ["pressure_hpa", "temperature_celsius", "altitude_meters"]
            assert all(key in data for key in expected_keys)
            assert all(isinstance(data[key], (int, float)) for key in expected_keys)

    def test_read_mock_sensor_data_error_handling(self, baro_config, mock_logger):
        """Test error handling in mock sensor data reading."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        # Mock an exception in data generation
        with patch.object(task, '_generate_realistic_data', side_effect=Exception("Test error")):
            data = task._read_mock_sensor_data(mock_logger)

            # Should return None values on error
            assert all(data[key] is None for key in data.keys())
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_successful(self, baro_config, mock_engine, mock_logger):
        """Test successful task execution."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is True
        assert result["error"] is None
        assert "data" in result
        assert "mock_strategy" in result
        assert result["mock_strategy"] == "realistic"

        # Check data structure
        data = result["data"]
        expected_keys = ["pressure_hpa", "temperature_celsius", "altitude_meters"]
        assert all(key in data for key in expected_keys)

    @pytest.mark.asyncio
    async def test_execute_database_error(self, baro_config, mock_logger):
        """Test task execution with database error."""
        # Create mock engine that raises exception
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Database error")

        task = MockBaroTask(baro_config, mock_strategy="static")
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Database error" in result["error"]
        assert "data" in result  # Should still have generated data

    @pytest.mark.asyncio
    async def test_execute_invalid_data(self, baro_config, mock_engine, mock_logger):
        """Test task execution when mock data generation fails."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        # Mock data generation to return None values
        with patch.object(task, '_read_mock_sensor_data', return_value={
            "pressure_hpa": None,
            "temperature_celsius": None,
            "altitude_meters": None
        }):
            result = await task.execute(mock_engine, mock_logger)

            assert result["success"] is False
            assert "Failed to generate valid mock barometer data" in result["error"]

    @pytest.mark.asyncio
    async def test_database_integration(self, baro_config, mock_engine, mock_logger):
        """Test database integration and SQL query structure."""
        task = MockBaroTask(baro_config, mock_strategy="flight")

        await task.execute(mock_engine, mock_logger)

        # Verify database operations
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

        # Check SQL call structure
        call_args = mock_conn.execute.call_args
        sql_query = call_args[0][0].text  # Get the actual SQL text from TextClause
        assert "INSERT INTO barometer_readings" in sql_query

    def test_reset_simulation(self, baro_config):
        """Test simulation reset functionality."""
        task = MockBaroTask(baro_config, mock_strategy="flight")

        # Advance simulation
        original_start_time = task.start_time
        task.current_altitude = 100.0
        task.flight_phase = "cruise"
        task.pressure_drift = 5.0

        # Reset simulation
        task.reset_simulation()

        # Should reset to initial state
        assert task.start_time > original_start_time
        assert task.current_altitude == 0.0
        assert task.flight_phase == "ground"
        assert task.pressure_drift == 0.0

    def test_set_mock_strategy(self, baro_config):
        """Test changing mock strategy at runtime."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")
        original_start_time = task.start_time

        # Change strategy
        task.set_mock_strategy("flight")

        assert task.mock_strategy == "flight"
        assert task.start_time >= original_start_time  # Should reset time

    def test_get_simulation_state(self, baro_config):
        """Test getting simulation state for debugging."""
        task = MockBaroTask(baro_config, mock_strategy="flight")
        task.flight_phase = "climb"
        task.current_altitude = 150.0

        state = task.get_simulation_state()

        expected_keys = ["mock_strategy", "flight_phase", "current_altitude", "pressure_drift", "elapsed_time"]
        assert all(key in state for key in expected_keys)
        assert state["mock_strategy"] == "flight"
        assert state["flight_phase"] == "climb"
        assert isinstance(state["elapsed_time"], float)

    def test_flight_phase_progression(self, baro_config):
        """Test that flight phases progress correctly over time."""
        task = MockBaroTask(baro_config, mock_strategy="flight")
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

    def test_pressure_altitude_relationship(self, baro_config):
        """Test that pressure and altitude have correct physical relationship."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        # Generate multiple samples
        samples = [task._generate_realistic_data() for _ in range(10)]

        for sample in samples:
            pressure = sample["pressure_hpa"]
            altitude = sample["altitude_meters"]

            # Verify barometric formula relationship
            # altitude = 44330 * (1 - (pressure / sea_level_pressure)^(1/5.255))
            expected_altitude = 44330.0 * (1.0 - pow(pressure / task.parameters.sea_level_pressure, 1.0 / 5.255))

            # Should match within rounding error
            assert abs(altitude - expected_altitude) < 0.1

    def test_temperature_variations(self, baro_config, mock_logger):
        """Test temperature generation patterns."""
        task = MockBaroTask(baro_config, mock_strategy="realistic")

        # Generate data over time
        temperatures = []
        for i in range(20):
            task.start_time = time.time() - (i * 300)  # 5 minute intervals
            data = task._generate_realistic_data()
            temperatures.append(data["temperature_celsius"])

        # Should have variations but stay in reasonable range
        assert len(set(temperatures)) > 5  # Some variation
        assert all(10 < t < 40 for t in temperatures)  # Reasonable range

        # Should not have extreme jumps between adjacent readings
        for i in range(1, len(temperatures)):
            temp_diff = abs(temperatures[i] - temperatures[i-1])
            assert temp_diff < 10  # No huge jumps


if __name__ == "__main__":
    pytest.main([__file__])
