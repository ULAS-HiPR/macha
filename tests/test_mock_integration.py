import pytest
import asyncio
import sys
import os
import tempfile
import time
from unittest.mock import Mock, AsyncMock
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import MachaConfig
from mock_camera_task import MockCameraTask
from mock_baro_task import MockBaroTask
from mock_imu_task import MockImuTask
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
def integrated_config():
    """Create a test configuration with all mock tasks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = MachaConfig(
            app={"name": "test_integration", "debug": True},
            logging={
                "level": "INFO",
                "file": {"path": os.path.join(temp_dir, "test.log")},
                "console": {"format": "test"}
            },
            db={
                "filename": os.path.join(temp_dir, "test.db"),
                "connection_string": f"sqlite:///{os.path.join(temp_dir, 'test.db')}",
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
                            {"port": 0, "name": "test_cam0", "output_folder": os.path.join(temp_dir, "cam0")},
                            {"port": 1, "name": "test_cam1", "output_folder": os.path.join(temp_dir, "cam1")}
                        ],
                        "image_format": "jpg",
                        "resolution": {"width": 640, "height": 480},
                        "quality": 85
                    }
                },
                {
                    "name": "test_mock_baro",
                    "class": "MockBaroTask",
                    "frequency": 30,
                    "enabled": True,
                    "parameters": {
                        "i2c_bus": 1,
                        "address": 0x77,
                        "sea_level_pressure": 1013.25
                    }
                },
                {
                    "name": "test_mock_imu",
                    "class": "MockImuTask",
                    "frequency": 10,
                    "enabled": True,
                    "parameters": {
                        "i2c_bus": 1,
                        "address": 0x6A,
                        "accel_range": "4G",
                        "gyro_range": "500DPS"
                    }
                },
                {
                    "name": "test_mock_metrics",
                    "class": "MockMetricsTask",
                    "frequency": 60,
                    "enabled": True
                }
            ]
        )
        yield config


class TestMockTaskIntegration:
    """Integration tests for all mock tasks working together."""

    @pytest.mark.asyncio
    async def test_all_mock_tasks_execution(self, integrated_config, mock_engine, mock_logger):
        """Test that all mock tasks can execute successfully."""
        # Initialize all mock tasks
        camera_task = MockCameraTask(integrated_config, mock_strategy="synthetic")
        baro_task = MockBaroTask(integrated_config, mock_strategy="realistic")
        imu_task = MockImuTask(integrated_config, mock_strategy="stationary")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="realistic")

        # Execute all tasks
        camera_result = await camera_task.execute(mock_engine, mock_logger)
        baro_result = await baro_task.execute(mock_engine, mock_logger)
        imu_result = await imu_task.execute(mock_engine, mock_logger)
        metrics_result = await metrics_task.execute(mock_engine, mock_logger)

        # All tasks should succeed
        assert camera_result["captured"] >= 0
        assert baro_result["success"] is True
        assert imu_result["success"] is True
        assert metrics_result["success"] is True

        # Check that all tasks have their expected data structures
        assert "data" in baro_result
        assert "data" in imu_result
        assert "data" in metrics_result
        assert "images" in camera_result

    @pytest.mark.asyncio
    async def test_flight_simulation_coordination(self, integrated_config, mock_engine, mock_logger):
        """Test coordinated flight simulation across multiple sensors."""
        # Initialize tasks with flight-related strategies
        baro_task = MockBaroTask(integrated_config, mock_strategy="flight")
        imu_task = MockImuTask(integrated_config, mock_strategy="flight")
        camera_task = MockCameraTask(integrated_config, mock_strategy="synthetic")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="realistic")

        # Execute multiple iterations to see flight progression
        results = []
        for i in range(10):
            # Simulate time progression
            baro_task.start_time -= 30  # 30 seconds earlier each iteration
            imu_task.start_time -= 30

            baro_result = await baro_task.execute(mock_engine, mock_logger)
            imu_result = await imu_task.execute(mock_engine, mock_logger)
            camera_result = await camera_task.execute(mock_engine, mock_logger)
            metrics_result = await metrics_task.execute(mock_engine, mock_logger)

            results.append({
                "baro": baro_result,
                "imu": imu_result,
                "camera": camera_result,
                "metrics": metrics_result,
                "baro_phase": baro_task.flight_phase,
                "imu_phase": imu_task.flight_phase
            })

        # Check that flight phases are progressing
        baro_phases = [r["baro_phase"] for r in results]
        imu_phases = [r["imu_phase"] for r in results]

        # Should see multiple phases
        assert len(set(baro_phases)) >= 2
        assert len(set(imu_phases)) >= 2

        # Both should be coordinated (same or adjacent phases at same times due to small timing differences)
        flight_phase_order = ["ground", "takeoff", "climb", "cruise", "descent", "landing"]
        for i in range(len(results)):
            baro_phase = results[i]["baro_phase"]
            imu_phase = results[i]["imu_phase"]
            # Allow for adjacent phases due to slight timing differences
            baro_idx = flight_phase_order.index(baro_phase)
            imu_idx = flight_phase_order.index(imu_phase)
            assert abs(baro_idx - imu_idx) <= 1, f"Phases too different: {baro_phase} vs {imu_phase}"

    def test_mock_task_strategy_consistency(self, integrated_config):
        """Test that mock tasks maintain consistent strategies."""
        strategies = ["realistic", "static", "flight", "stressed"]

        for strategy in strategies:
            if strategy == "flight":
                # Only baro and IMU support flight strategy
                baro_task = MockBaroTask(integrated_config, mock_strategy=strategy)
                imu_task = MockImuTask(integrated_config, mock_strategy=strategy)
                assert baro_task.mock_strategy == strategy
                assert imu_task.mock_strategy == strategy
            elif strategy == "stressed":
                # Only metrics supports stressed strategy
                metrics_task = MockMetricsTask(integrated_config, mock_strategy=strategy)
                assert metrics_task.mock_strategy == strategy
            else:
                # All tasks support realistic and static
                camera_task = MockCameraTask(integrated_config, mock_strategy=strategy)
                baro_task = MockBaroTask(integrated_config, mock_strategy=strategy)
                imu_task = MockImuTask(integrated_config, mock_strategy=strategy)
                metrics_task = MockMetricsTask(integrated_config, mock_strategy=strategy)

                assert camera_task.mock_strategy == strategy
                assert baro_task.mock_strategy == strategy
                assert imu_task.mock_strategy == strategy
                assert metrics_task.mock_strategy == strategy

    @pytest.mark.asyncio
    async def test_database_integration_all_tasks(self, integrated_config, mock_engine, mock_logger):
        """Test that all mock tasks integrate with database correctly."""
        # Initialize all tasks
        camera_task = MockCameraTask(integrated_config, mock_strategy="synthetic")
        baro_task = MockBaroTask(integrated_config, mock_strategy="static")
        imu_task = MockImuTask(integrated_config, mock_strategy="static")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="static")

        # Execute all tasks
        await camera_task.execute(mock_engine, mock_logger)
        await baro_task.execute(mock_engine, mock_logger)
        await imu_task.execute(mock_engine, mock_logger)
        await metrics_task.execute(mock_engine, mock_logger)

        # Verify database operations for each task
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value

        # Should have called execute for each task
        assert mock_conn.execute.call_count >= 4

        # Should have committed for each task
        assert mock_conn.commit.call_count >= 4

        # Check that different table inserts were called
        # Extract SQL text from TextClause objects
        call_args_list = mock_conn.execute.call_args_list

        # Should see inserts to different tables
        table_inserts = []
        for call_args in call_args_list:
            sql_query = call_args[0][0].text  # Get the actual SQL text from TextClause
            if "INSERT INTO images" in sql_query:
                table_inserts.append("images")
            elif "INSERT INTO barometer_readings" in sql_query:
                table_inserts.append("barometer_readings")
            elif "INSERT INTO imu_readings" in sql_query:
                table_inserts.append("imu_readings")
            elif "INSERT INTO system_metrics" in sql_query:
                table_inserts.append("system_metrics")

        # Should have inserted into multiple tables
        assert len(set(table_inserts)) >= 3

    def test_mock_task_data_correlation(self, integrated_config, mock_logger):
        """Test data correlation between related mock tasks."""
        # Initialize flight simulation tasks
        baro_task = MockBaroTask(integrated_config, mock_strategy="flight")
        imu_task = MockImuTask(integrated_config, mock_strategy="flight")

        # Synchronize their start times
        sync_time = time.time()
        baro_task.start_time = sync_time
        imu_task.start_time = sync_time

        # Generate data at the same time points
        for i in range(5):
            baro_data = baro_task._read_mock_sensor_data(mock_logger)
            imu_data = imu_task._read_mock_sensor_data(mock_logger)

            # In ground phase, altitude should be low and accelerations minimal
            if baro_task.flight_phase == "ground" or imu_task.flight_phase == "ground":
                assert baro_data["altitude_meters"] < 50
                assert abs(imu_data["accel_x"]) < 1.0
                assert abs(imu_data["accel_y"]) < 2.0

            # During takeoff/climb, should see reasonable altitude and acceleration
            if (baro_task.flight_phase in ["takeoff", "climb"] or
                imu_task.flight_phase in ["takeoff", "climb"]):
                assert baro_data["altitude_meters"] >= -50  # Allow for some below-sea-level variation
                # Could have forward acceleration during takeoff
                assert -5.0 <= imu_data["accel_y"] <= 10.0

            # Advance time
            baro_task.start_time -= 30
            imu_task.start_time -= 30

    def test_performance_characteristics(self, integrated_config, mock_engine, mock_logger):
        """Test performance characteristics of mock tasks."""
        # Initialize tasks
        camera_task = MockCameraTask(integrated_config, mock_strategy="synthetic")
        baro_task = MockBaroTask(integrated_config, mock_strategy="realistic")
        imu_task = MockImuTask(integrated_config, mock_strategy="stationary")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="realistic")

        # Time the execution of each task
        async def time_task_execution(task):
            start_time = time.time()
            await task.execute(mock_engine, mock_logger)
            return time.time() - start_time

        # All tasks should execute quickly (mock data generation should be fast)
        camera_time = asyncio.run(time_task_execution(camera_task))
        baro_time = asyncio.run(time_task_execution(baro_task))
        imu_time = asyncio.run(time_task_execution(imu_task))
        metrics_time = asyncio.run(time_task_execution(metrics_task))

        # All should complete in reasonable time (allowing for test overhead)
        assert camera_time < 5.0  # seconds
        assert baro_time < 1.0
        assert imu_time < 1.0
        assert metrics_time < 1.0

    def test_error_recovery_coordination(self, integrated_config, mock_logger):
        """Test error recovery when some mock tasks fail."""
        # Create mock engine that fails for some operations
        failing_engine = Mock()
        failing_conn = AsyncMock()
        failing_conn.__aenter__ = AsyncMock(return_value=failing_conn)
        failing_conn.__aexit__ = AsyncMock(return_value=None)

        # Make some database operations fail
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise Exception("Database error")
            return AsyncMock()

        failing_conn.execute.side_effect = side_effect
        failing_engine.connect.return_value = failing_conn

        # Initialize tasks
        baro_task = MockBaroTask(integrated_config, mock_strategy="static")
        imu_task = MockImuTask(integrated_config, mock_strategy="static")

        # Execute tasks - some should fail, some should succeed
        async def test_execution():
            baro_result = await baro_task.execute(failing_engine, mock_logger)
            imu_result = await imu_task.execute(failing_engine, mock_logger)
            return baro_result, imu_result

        baro_result, imu_result = asyncio.run(test_execution())

        # At least one should have failed due to database error
        results = [baro_result, imu_result]
        success_count = sum(1 for r in results if r["success"])
        failure_count = sum(1 for r in results if not r["success"])

        # Should have both successes and failures
        assert failure_count > 0
        # All should still have data even if database storage failed
        assert all("data" in r for r in results)

    def test_mock_task_state_management(self, integrated_config):
        """Test state management across multiple mock tasks."""
        # Initialize tasks
        baro_task = MockBaroTask(integrated_config, mock_strategy="flight")
        imu_task = MockImuTask(integrated_config, mock_strategy="flight")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="variable")

        # Get initial states
        baro_state = baro_task.get_simulation_state()
        imu_state = imu_task.get_simulation_state()
        metrics_state = metrics_task.get_simulation_state()

        # All should have state information
        assert "mock_strategy" in baro_state
        assert "mock_strategy" in imu_state
        assert "mock_strategy" in metrics_state

        assert "elapsed_time" in baro_state
        assert "elapsed_time" in imu_state
        assert "elapsed_time" in metrics_state

        # Reset all simulations
        baro_task.reset_simulation()
        imu_task.reset_simulation()
        metrics_task.reset_simulation()

        # Get new states
        new_baro_state = baro_task.get_simulation_state()
        new_imu_state = imu_task.get_simulation_state()
        new_metrics_state = metrics_task.get_simulation_state()

        # Elapsed times should be reset (close to 0)
        assert new_baro_state["elapsed_time"] < baro_state["elapsed_time"]
        assert new_imu_state["elapsed_time"] < imu_state["elapsed_time"]
        assert new_metrics_state["elapsed_time"] < metrics_state["elapsed_time"]

    @pytest.mark.asyncio
    async def test_concurrent_execution(self, integrated_config, mock_engine, mock_logger):
        """Test concurrent execution of multiple mock tasks."""
        # Initialize tasks
        camera_task = MockCameraTask(integrated_config, mock_strategy="synthetic")
        baro_task = MockBaroTask(integrated_config, mock_strategy="realistic")
        imu_task = MockImuTask(integrated_config, mock_strategy="stationary")
        metrics_task = MockMetricsTask(integrated_config, mock_strategy="realistic")

        # Execute all tasks concurrently
        results = await asyncio.gather(
            camera_task.execute(mock_engine, mock_logger),
            baro_task.execute(mock_engine, mock_logger),
            imu_task.execute(mock_engine, mock_logger),
            metrics_task.execute(mock_engine, mock_logger),
            return_exceptions=True
        )

        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)

        # Check results
        camera_result, baro_result, imu_result, metrics_result = results

        assert camera_result["captured"] >= 0
        assert baro_result["success"] is True
        assert imu_result["success"] is True
        assert metrics_result["success"] is True

    def test_configuration_compatibility(self, integrated_config):
        """Test that mock tasks are compatible with the same configuration as real tasks."""
        # Mock tasks should be able to initialize with the same config as real tasks
        # This ensures they can be drop-in replacements

        # Test that task parameters are correctly parsed
        camera_task = MockCameraTask(integrated_config)
        baro_task = MockBaroTask(integrated_config)
        imu_task = MockImuTask(integrated_config)
        metrics_task = MockMetricsTask(integrated_config)

        # Camera parameters
        assert len(camera_task.cameras) == 2
        assert camera_task.cameras[0].name == "test_cam0"
        assert camera_task.cameras[1].name == "test_cam1"
        assert camera_task.image_format == "jpg"

        # Barometer parameters
        assert baro_task.parameters.i2c_bus == 1
        assert baro_task.parameters.address == 0x77
        assert baro_task.parameters.sea_level_pressure == 1013.25

        # IMU parameters
        assert imu_task.parameters.i2c_bus == 1
        assert imu_task.parameters.address == 0x6A
        assert imu_task.parameters.accel_range == "4G"
        assert imu_task.parameters.gyro_range == "500DPS"

        # Metrics task (no specific parameters, but should have default strategy)
        assert metrics_task.mock_strategy in ["auto", "realistic"]  # Allow for default strategy variation


if __name__ == "__main__":
    pytest.main([__file__])
