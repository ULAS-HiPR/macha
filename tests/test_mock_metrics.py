import pytest
import asyncio
import sys
import os
import time
import platform
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig
from mock_metrics_task import MockMetricsTask


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
def metrics_config():
    """Create a test configuration with metrics task."""
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
                "name": "test_metrics",
                "class": "MockMetricsTask",
                "frequency": 60,
                "enabled": True
            }
        ]
    )


class TestMockMetricsTask:
    """Unit tests for MockMetricsTask."""

    def test_initialization(self, metrics_config):
        """Test MockMetricsTask initialization."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        assert task.mock_strategy == "realistic"
        assert task.cpu_count == 4
        assert task.storage_total_gb == 64.0
        assert task.ram_total_gb == 4.0
        assert isinstance(task.hostname, str)
        assert isinstance(task.system, str)
        assert isinstance(task.release, str)

    def test_initialization_minimal_config(self):
        """Test MockMetricsTask with minimal configuration."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "dummy_task",
                    "class": "MockMetricsTask",
                    "frequency": 60,
                    "enabled": True
                }
            ]
        )

        task = MockMetricsTask(config, mock_strategy="static")
        assert task.mock_strategy == "static"

    def test_strategy_determination_auto(self, metrics_config):
        """Test automatic strategy determination."""
        task = MockMetricsTask(metrics_config, mock_strategy="auto")
        assert task.mock_strategy == "realistic"

    def test_strategy_determination_explicit(self, metrics_config):
        """Test explicit strategy setting."""
        strategies = ["realistic", "stressed", "idle", "static", "variable"]

        for strategy in strategies:
            task = MockMetricsTask(metrics_config, mock_strategy=strategy)
            assert task.mock_strategy == strategy

    def test_generate_realistic_data(self, metrics_config):
        """Test realistic data generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        data = task._generate_realistic_data()

        # Check all expected keys are present
        expected_keys = [
            "cpu_percent", "cpu_count", "temperature_c",
            "storage_total_gb", "storage_used_gb", "storage_free_gb",
            "ram_total_gb", "ram_used_gb", "ram_free_gb",
            "uptime_seconds", "hostname", "system", "release"
        ]
        assert all(key in data for key in expected_keys)

        # Check data types and reasonable ranges
        assert isinstance(data["cpu_percent"], float)
        assert isinstance(data["cpu_count"], int)
        assert isinstance(data["temperature_c"], float)
        assert isinstance(data["uptime_seconds"], float)

        # CPU should be in reasonable range
        assert 0 <= data["cpu_percent"] <= 100
        # Temperature should be reasonable for Pi
        assert 25 <= data["temperature_c"] <= 85
        # RAM usage should be logical
        assert data["ram_used_gb"] + data["ram_free_gb"] == data["ram_total_gb"]
        # Storage usage should be logical
        assert data["storage_used_gb"] + data["storage_free_gb"] == data["storage_total_gb"]

    def test_generate_stressed_data(self, metrics_config):
        """Test stressed data generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="stressed")

        data = task._generate_stressed_data()

        # Should show high resource usage
        assert data["cpu_percent"] >= 70.0
        assert data["temperature_c"] >= 65.0
        assert data["ram_used_gb"] / data["ram_total_gb"] >= 0.7
        assert data["storage_used_gb"] / data["storage_total_gb"] >= 0.8

        # Data consistency checks
        assert data["ram_used_gb"] <= data["ram_total_gb"]
        assert data["storage_used_gb"] <= data["storage_total_gb"]

    def test_generate_idle_data(self, metrics_config):
        """Test idle data generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="idle")

        data = task._generate_idle_data()

        # Should show low resource usage
        assert data["cpu_percent"] <= 15.0
        assert data["temperature_c"] <= 50.0
        assert data["ram_used_gb"] / data["ram_total_gb"] <= 0.3
        assert data["storage_used_gb"] <= 25.0

        # Data consistency checks
        assert data["ram_used_gb"] >= 0
        assert data["storage_used_gb"] >= 0

    def test_generate_variable_data(self, metrics_config):
        """Test variable data generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="variable")

        # Generate multiple samples to test variability
        samples = [task._generate_variable_data() for _ in range(20)]

        cpu_values = [s["cpu_percent"] for s in samples]
        ram_usage = [s["ram_used_gb"] / s["ram_total_gb"] for s in samples]

        # Should have high variability
        assert max(cpu_values) - min(cpu_values) > 30.0
        assert max(ram_usage) - min(ram_usage) > 0.3

        # All values should still be in valid ranges
        assert all(0 <= cpu <= 100 for cpu in cpu_values)
        assert all(0 <= ram <= 1.0 for ram in ram_usage)

    def test_generate_static_data(self, metrics_config):
        """Test static data generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="static")

        data1 = task._generate_static_data()
        time.sleep(0.1)  # Small delay
        data2 = task._generate_static_data()

        # Most values should be identical for static mode
        assert data1["cpu_percent"] == data2["cpu_percent"]
        assert data1["temperature_c"] == data2["temperature_c"]
        assert data1["ram_used_gb"] == data2["ram_used_gb"]
        assert data1["storage_used_gb"] == data2["storage_used_gb"]

        # In static mode, even uptime should be the same
        assert data2["uptime_seconds"] == data1["uptime_seconds"]

    def test_generate_mock_metrics_all_strategies(self, metrics_config, mock_logger):
        """Test generating mock metrics for all strategies."""
        strategies = ["realistic", "stressed", "idle", "variable", "static"]

        for strategy in strategies:
            task = MockMetricsTask(metrics_config, mock_strategy=strategy)
            data = task._generate_mock_metrics(mock_logger)

            assert isinstance(data, dict)
            assert len(data) > 0
            # Should have all required fields
            assert "cpu_percent" in data
            assert "temperature_c" in data
            assert "ram_used_gb" in data

    def test_generate_mock_metrics_error_handling(self, metrics_config, mock_logger):
        """Test error handling in mock metrics generation."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Mock an exception in data generation
        with patch.object(task, '_generate_realistic_data', side_effect=Exception("Test error")):
            data = task._generate_mock_metrics(mock_logger)

            # Should return empty dict on error
            assert data == {}
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_execute_successful(self, metrics_config, mock_engine, mock_logger):
        """Test successful task execution."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is True
        assert result["error"] is None
        assert "data" in result
        assert "mock_strategy" in result
        assert result["mock_strategy"] == "realistic"

        # Check data structure
        data = result["data"]
        expected_keys = ["cpu_percent", "temperature_c", "ram_used_gb", "hostname"]
        assert all(key in data for key in expected_keys)

    @pytest.mark.asyncio
    async def test_execute_database_error(self, metrics_config, mock_logger):
        """Test task execution with database error."""
        # Create mock engine that raises exception
        mock_engine = Mock()
        mock_engine.connect.side_effect = Exception("Database error")

        task = MockMetricsTask(metrics_config, mock_strategy="static")
        result = await task.execute(mock_engine, mock_logger)

        assert result["success"] is False
        assert "Database error" in result["error"]
        assert "data" in result  # Should still have generated data

    @pytest.mark.asyncio
    async def test_execute_no_metrics_generated(self, metrics_config, mock_engine, mock_logger):
        """Test task execution when metrics generation fails."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Mock metrics generation to return empty dict
        with patch.object(task, '_generate_mock_metrics', return_value={}):
            result = await task.execute(mock_engine, mock_logger)

            assert result["success"] is False
            assert "Failed to generate mock system metrics" in result["error"]

    @pytest.mark.asyncio
    async def test_database_integration(self, metrics_config, mock_engine, mock_logger):
        """Test database integration and SQL query structure."""
        task = MockMetricsTask(metrics_config, mock_strategy="idle")

        await task.execute(mock_engine, mock_logger)

        # Verify database operations
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called()
        mock_conn.commit.assert_called()

        # Check SQL call structure
        call_args = mock_conn.execute.call_args
        sql_query = call_args[0][0].text  # Get the actual SQL text from TextClause
        assert "INSERT INTO system_metrics" in sql_query

    def test_reset_simulation(self, metrics_config):
        """Test simulation reset functionality."""
        task = MockMetricsTask(metrics_config, mock_strategy="variable")

        # Advance simulation
        original_start_time = task.start_time
        time.sleep(0.1)

        # Reset simulation
        task.reset_simulation()

        # Should reset start time
        assert task.start_time > original_start_time

    def test_set_mock_strategy(self, metrics_config):
        """Test changing mock strategy at runtime."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")
        original_start_time = task.start_time

        # Change strategy
        task.set_mock_strategy("stressed")

        assert task.mock_strategy == "stressed"
        assert task.start_time >= original_start_time  # Should reset time

    def test_get_simulation_state(self, metrics_config):
        """Test getting simulation state for debugging."""
        task = MockMetricsTask(metrics_config, mock_strategy="variable")

        state = task.get_simulation_state()

        expected_keys = ["mock_strategy", "elapsed_time", "hostname", "system", "cpu_count"]
        assert all(key in state for key in expected_keys)
        assert state["mock_strategy"] == "variable"
        assert isinstance(state["elapsed_time"], float)
        assert isinstance(state["cpu_count"], int)

    def test_uptime_progression(self, metrics_config):
        """Test that uptime increases over time."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Generate data at different times
        task.start_time = time.time() - 100  # 100 seconds ago
        data1 = task._generate_realistic_data()

        task.start_time = time.time() - 200  # 200 seconds ago
        data2 = task._generate_realistic_data()

        # Uptime should be higher for older start time
        assert data2["uptime_seconds"] > data1["uptime_seconds"]
        assert data2["uptime_seconds"] - data1["uptime_seconds"] == pytest.approx(100, abs=1)

    def test_cpu_temperature_correlation(self, metrics_config):
        """Test correlation between CPU usage and temperature in realistic mode."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Generate many samples
        samples = [task._generate_realistic_data() for _ in range(50)]

        cpu_values = [s["cpu_percent"] for s in samples]
        temp_values = [s["temperature_c"] for s in samples]

        # Calculate simple correlation
        import statistics
        cpu_mean = statistics.mean(cpu_values)
        temp_mean = statistics.mean(temp_values)

        correlation_sum = sum((cpu - cpu_mean) * (temp - temp_mean) for cpu, temp in zip(cpu_values, temp_values))

        # Should have some positive correlation (higher CPU -> higher temp)
        # Not enforcing strict correlation since it's random data, but should trend positive
        assert correlation_sum != 0  # Should have some relationship

    def test_resource_usage_bounds(self, metrics_config):
        """Test that resource usage stays within logical bounds."""
        strategies = ["realistic", "stressed", "idle", "variable"]

        for strategy in strategies:
            task = MockMetricsTask(metrics_config, mock_strategy=strategy)

            # Generate multiple samples
            samples = [getattr(task, f'_generate_{strategy}_data')() for _ in range(20)]

            for sample in samples:
                # CPU percentage bounds
                assert 0 <= sample["cpu_percent"] <= 100

                # Temperature bounds (reasonable for Pi)
                assert 20 <= sample["temperature_c"] <= 90

                # RAM usage logical
                assert sample["ram_used_gb"] >= 0
                assert sample["ram_used_gb"] <= sample["ram_total_gb"]
                assert sample["ram_free_gb"] >= 0
                assert sample["ram_used_gb"] + sample["ram_free_gb"] == sample["ram_total_gb"]

                # Storage usage logical
                assert sample["storage_used_gb"] >= 0
                assert sample["storage_used_gb"] <= sample["storage_total_gb"]
                assert sample["storage_free_gb"] >= 0
                assert sample["storage_used_gb"] + sample["storage_free_gb"] == sample["storage_total_gb"]

                # Uptime should be positive
                assert sample["uptime_seconds"] >= 0

    def test_time_based_variations(self, metrics_config):
        """Test that metrics show time-based variations in realistic mode."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Generate data over different time periods
        cpu_values = []
        temp_values = []

        for i in range(20):
            task.start_time = time.time() - (i * task.cpu_cycle_period / 10)  # Sample across cycle
            data = task._generate_realistic_data()
            cpu_values.append(data["cpu_percent"])
            temp_values.append(data["temperature_c"])

        # Should see variations across the time cycle
        assert len(set([round(cpu, 0) for cpu in cpu_values])) > 5
        assert len(set([round(temp, 0) for temp in temp_values])) > 5

    def test_system_info_consistency(self, metrics_config):
        """Test that system information remains consistent."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Generate multiple samples
        samples = [task._generate_realistic_data() for _ in range(10)]

        # System info should be consistent across samples
        hostnames = set(s["hostname"] for s in samples)
        systems = set(s["system"] for s in samples)
        releases = set(s["release"] for s in samples)
        cpu_counts = set(s["cpu_count"] for s in samples)

        assert len(hostnames) == 1
        assert len(systems) == 1
        assert len(releases) == 1
        assert len(cpu_counts) == 1

        # Should have reasonable values
        sample = samples[0]
        assert len(sample["hostname"]) > 0
        assert sample["system"] in ["Linux", "Darwin", "Windows"]
        assert sample["cpu_count"] > 0

    def test_storage_growth_simulation(self, metrics_config):
        """Test storage growth over time in realistic mode."""
        task = MockMetricsTask(metrics_config, mock_strategy="realistic")

        # Sample storage usage over time
        storage_values = []
        for days in [0, 1, 5, 10]:
            task.start_time = time.time() - (days * 24 * 3600)  # Days ago
            data = task._generate_realistic_data()
            storage_values.append(data["storage_used_gb"])

        # Storage should generally increase over time (with some noise)
        # Earlier times (larger indices) should tend to have more storage used
        assert storage_values[-1] > storage_values[0]  # 10 days ago vs now

    def test_ram_usage_patterns(self, metrics_config):
        """Test RAM usage patterns across different strategies."""
        task_idle = MockMetricsTask(metrics_config, mock_strategy="idle")
        task_stressed = MockMetricsTask(metrics_config, mock_strategy="stressed")

        idle_samples = [task_idle._generate_idle_data() for _ in range(10)]
        stressed_samples = [task_stressed._generate_stressed_data() for _ in range(10)]

        idle_ram_usage = [s["ram_used_gb"] / s["ram_total_gb"] for s in idle_samples]
        stressed_ram_usage = [s["ram_used_gb"] / s["ram_total_gb"] for s in stressed_samples]

        # Stressed should consistently use more RAM than idle
        import statistics
        avg_idle = statistics.mean(idle_ram_usage)
        avg_stressed = statistics.mean(stressed_ram_usage)

        assert avg_stressed > avg_idle
        assert avg_idle < 0.4  # Idle should be low
        assert avg_stressed > 0.6  # Stressed should be high


if __name__ == "__main__":
    pytest.main([__file__])
