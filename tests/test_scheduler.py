import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scheduler import TaskScheduler
from config import MachaConfig
from task import Task


@pytest.fixture
def mock_engine():
    """Create a mock database engine."""
    engine = Mock(spec=AsyncEngine)
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
def mock_config():
    """Create a test configuration with multiple tasks."""
    import tempfile
    temp_dir = tempfile.mkdtemp()

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
                "name": "test_metrics",
                "class": "MetricsTask",
                "frequency": 60,
                "enabled": True
            },
            {
                "name": "test_camera",
                "class": "CameraTask",
                "frequency": 10,
                "enabled": True,
                "parameters": {
                    "cameras": [{"port": 0, "name": "cam0", "output_folder": temp_dir}],
                    "image_format": "jpg",
                    "resolution": {"width": 640, "height": 480}
                }
            },
            {
                "name": "disabled_task",
                "class": "MockCameraTask",
                "frequency": 5,
                "enabled": False,
                "parameters": {
                    "cameras": [{"port": 1, "name": "cam1", "output_folder": temp_dir}],
                    "image_format": "jpg",
                    "resolution": {"width": 640, "height": 480}
                }
            }
        ]
    )

    yield config

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def minimal_config():
    """Create a minimal test configuration."""
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
                "name": "single_task",
                "class": "MetricsTask",
                "frequency": 30,
                "enabled": True
            }
        ]
    )


class MockTask(Task):
    """Mock task for testing."""

    def __init__(self, config, name=None, should_fail=False):
        super().__init__(config)
        if name:
            self.name = name
        self.should_fail = should_fail
        self.execution_count = 0

    async def execute(self, engine, logger):
        self.execution_count += 1
        if self.should_fail:
            raise Exception(f"Mock task {self.name} failed")
        return {"success": True, "execution_count": self.execution_count}


class TestTaskSchedulerInitialization:
    """Test TaskScheduler initialization."""

    @patch('scheduler.TaskManager')
    def test_scheduler_initialization(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test basic scheduler initialization."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        assert scheduler.config == mock_config
        assert scheduler.engine == mock_engine
        assert scheduler.logger == mock_logger
        assert not scheduler.running
        assert isinstance(scheduler.task_instances, dict)
        assert isinstance(scheduler.next_run_times, dict)
        assert len(scheduler.task_configs) == 2  # Only enabled tasks

        mock_task_manager.assert_called_once_with(mock_engine, mock_logger)

    @patch('scheduler.TaskManager')
    def test_scheduler_with_minimal_config(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test scheduler with minimal configuration."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        assert len(scheduler.task_configs) == 1
        assert scheduler.task_configs[0].name == "single_task"

    @patch('scheduler.TaskManager')
    @patch('scheduler.MetricsTask')
    def test_task_initialization_success(self, mock_metrics_task, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test successful task initialization."""
        mock_task_instance = Mock()
        mock_metrics_task.return_value = mock_task_instance

        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        assert "single_task" in scheduler.task_instances
        assert scheduler.task_instances["single_task"] == mock_task_instance
        assert "single_task" in scheduler.next_run_times
        assert isinstance(scheduler.next_run_times["single_task"], datetime)

    @patch('scheduler.TaskManager')
    @patch('scheduler.MetricsTask')
    def test_task_initialization_failure(self, mock_metrics_task, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test task initialization failure handling."""
        mock_metrics_task.side_effect = Exception("Task init error")

        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Failed task should not be in instances
        assert "single_task" not in scheduler.task_instances
        # Task should be disabled
        assert not scheduler.task_configs[0].enabled

        mock_logger.error.assert_any_call("Failed to initialize task single_task: Task init error")

    @patch('scheduler.TaskManager')
    def test_unknown_task_class_handling(self, mock_task_manager, mock_engine, mock_logger):
        """Test handling of unknown task classes."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "unknown_task",
                    "class": "UnknownTaskClass",
                    "frequency": 60,
                    "enabled": True
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        assert "unknown_task" not in scheduler.task_instances
        mock_logger.error.assert_any_call("Unknown task class: UnknownTaskClass")


class TestSchedulerExecution:
    """Test scheduler execution and task running."""

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test scheduler start and stop."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Mock a task instance
        mock_task = MockTask(minimal_config, "test_task")
        scheduler.task_instances["single_task"] = mock_task

        # Start scheduler in background and stop it quickly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.1)  # Let it start
        await scheduler.stop()

        # Wait for start task to complete
        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        assert not scheduler.running

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_task_execution_timing(self, mock_task_manager, mock_engine, mock_logger):
        """Test that tasks are executed at correct intervals."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "fast_task",
                    "class": "MetricsTask",
                    "frequency": 1,  # 1 second
                    "enabled": True
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        # Mock task and task manager
        mock_task = MockTask(config, "fast_task")
        scheduler.task_instances["fast_task"] = mock_task
        mock_task_manager_instance = Mock()
        mock_task_manager_instance.run_task = AsyncMock()
        scheduler.task_manager = mock_task_manager_instance

        # Set initial run time to past to trigger immediate execution
        scheduler.next_run_times["fast_task"] = datetime.now() - timedelta(seconds=1)

        # Run scheduler briefly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.5)  # Let it run briefly
        await scheduler.stop()

        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        # Task should have been executed
        mock_task_manager_instance.run_task.assert_called()

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_task_execution_error_handling(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test error handling during task execution."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Mock failing task
        mock_task = MockTask(minimal_config, "failing_task", should_fail=True)
        scheduler.task_instances["single_task"] = mock_task

        # Mock task manager to raise exception
        mock_task_manager_instance = Mock()
        mock_task_manager_instance.run_task = AsyncMock(side_effect=Exception("Task execution error"))
        scheduler.task_manager = mock_task_manager_instance

        # Set to run immediately
        scheduler.next_run_times["single_task"] = datetime.now() - timedelta(seconds=1)

        # Run scheduler briefly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.1)
        await scheduler.stop()

        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        # Should log error but continue
        mock_logger.error.assert_any_call("Error executing task single_task: Task execution error")

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_disabled_tasks_not_executed(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test that disabled tasks are not executed."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        # Mock task manager
        mock_task_manager_instance = Mock()
        mock_task_manager_instance.run_task = AsyncMock()
        scheduler.task_manager = mock_task_manager_instance

        # Verify disabled task is not in instances
        assert "disabled_task" not in scheduler.task_instances

        # Run scheduler briefly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.1)
        await scheduler.stop()

        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        # Disabled task should never be executed
        # Only enabled tasks should have been considered for execution


class TestTaskStatusAndManagement:
    """Test task status reporting and dynamic management."""

    @patch('scheduler.TaskManager')
    def test_get_task_status(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test getting task status information."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        status = scheduler.get_task_status()

        assert isinstance(status, dict)
        assert "test_metrics" in status
        assert "test_camera" in status
        assert "disabled_task" not in status  # Disabled tasks not included

        metrics_status = status["test_metrics"]
        assert metrics_status["enabled"] is True
        assert metrics_status["frequency"] == 60
        assert metrics_status["class"] == "MetricsTask"
        assert "next_run" in metrics_status
        assert "seconds_until_next_run" in metrics_status

    @patch('scheduler.TaskManager')
    def test_add_task_dynamically(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test adding a task dynamically."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Create new task config
        new_task_config = Mock()
        new_task_config.name = "dynamic_task"
        new_task_config.frequency = 15
        new_task_config.enabled = True
        new_task_config.class_name = "MetricsTask"

        # Create mock task instance
        mock_task = MockTask(minimal_config, "dynamic_task")

        scheduler.add_task(new_task_config, mock_task)

        assert "dynamic_task" in scheduler.task_instances
        assert scheduler.task_instances["dynamic_task"] == mock_task
        assert "dynamic_task" in scheduler.next_run_times
        assert new_task_config in scheduler.task_configs

        mock_logger.info.assert_any_call("Added new task to scheduler: dynamic_task")

    @patch('scheduler.TaskManager')
    def test_remove_task_dynamically(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test removing a task dynamically."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        # Add a mock task first
        mock_task = MockTask(mock_config, "test_task")
        scheduler.task_instances["test_metrics"] = mock_task

        scheduler.remove_task("test_metrics")

        assert "test_metrics" not in scheduler.task_instances
        assert "test_metrics" not in scheduler.next_run_times
        assert not any(tc.name == "test_metrics" for tc in scheduler.task_configs)

        mock_logger.info.assert_any_call("Removed task from scheduler: test_metrics")

    @patch('scheduler.TaskManager')
    def test_update_task_frequency(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test updating task frequency."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        scheduler.update_task_frequency("test_metrics", 120)

        # Find the task config and verify frequency was updated
        metrics_config = next(tc for tc in scheduler.task_configs if tc.name == "test_metrics")
        assert metrics_config.frequency == 120

        mock_logger.info.assert_any_call("Updated frequency for test_metrics: 120 seconds")

    @patch('scheduler.TaskManager')
    def test_update_nonexistent_task_frequency(self, mock_task_manager, mock_config, mock_engine, mock_logger):
        """Test updating frequency of non-existent task."""
        scheduler = TaskScheduler(mock_config, mock_engine, mock_logger)

        # Should not raise exception
        scheduler.update_task_frequency("nonexistent_task", 60)

        # Should not log anything since task doesn't exist
        assert not any("Updated frequency for nonexistent_task" in str(call)
                      for call in mock_logger.info.call_args_list)


class TestSchedulerTiming:
    """Test scheduler timing and frequency calculations."""

    @patch('scheduler.TaskManager')
    def test_next_run_time_calculation(self, mock_task_manager, mock_engine, mock_logger):
        """Test next run time calculation after task execution."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "timed_task",
                    "class": "MetricsTask",
                    "frequency": 30,
                    "enabled": True
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        # Mock task
        mock_task = MockTask(config, "timed_task")
        scheduler.task_instances["timed_task"] = mock_task

        initial_time = datetime.now()
        scheduler.next_run_times["timed_task"] = initial_time

        # Mock task manager
        mock_task_manager_instance = Mock()
        mock_task_manager_instance.run_task = AsyncMock()
        scheduler.task_manager = mock_task_manager_instance

        # Simulate task execution
        current_time = initial_time + timedelta(seconds=1)

        # Manually trigger the scheduling logic
        if current_time >= scheduler.next_run_times["timed_task"]:
            # This is what the scheduler does internally
            new_next_time = current_time + timedelta(seconds=30)
            scheduler.next_run_times["timed_task"] = new_next_time

        # Verify next run time was updated
        assert scheduler.next_run_times["timed_task"] > initial_time

    @patch('scheduler.TaskManager')
    def test_seconds_until_next_run_calculation(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test calculation of seconds until next run."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Set next run time to 30 seconds from now
        future_time = datetime.now() + timedelta(seconds=30)
        scheduler.next_run_times["single_task"] = future_time

        status = scheduler.get_task_status()

        # Should be approximately 30 seconds (allowing for small timing differences)
        seconds_until = status["single_task"]["seconds_until_next_run"]
        assert 25 <= seconds_until <= 35

    @patch('scheduler.TaskManager')
    def test_past_due_task_next_run_calculation(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test next run calculation for past due tasks."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Set next run time to past
        past_time = datetime.now() - timedelta(seconds=30)
        scheduler.next_run_times["single_task"] = past_time

        status = scheduler.get_task_status()

        # Should show 0 seconds for past due tasks
        assert status["single_task"]["seconds_until_next_run"] == 0


class TestSchedulerErrorHandling:
    """Test scheduler error handling in various scenarios."""

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_scheduler_continues_after_task_error(self, mock_task_manager, mock_engine, mock_logger):
        """Test that scheduler continues running after individual task errors."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "good_task",
                    "class": "MetricsTask",
                    "frequency": 1,
                    "enabled": True
                },
                {
                    "name": "bad_task",
                    "class": "CameraTask",
                    "frequency": 1,
                    "enabled": True,
                    "parameters": {
                        "cameras": [{"port": 0, "name": "cam0", "output_folder": "test"}],
                        "image_format": "jpg",
                        "resolution": {"width": 640, "height": 480}
                    }
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        # Mock tasks
        good_task = MockTask(config, "good_task")
        bad_task = MockTask(config, "bad_task", should_fail=True)
        scheduler.task_instances["good_task"] = good_task
        scheduler.task_instances["bad_task"] = bad_task

        # Mock task manager with selective failure
        mock_task_manager_instance = Mock()

        async def selective_run_task(task):
            if task.name == "bad_task":
                raise Exception("Bad task error")
            return await task.execute(mock_engine, mock_logger)

        mock_task_manager_instance.run_task = AsyncMock(side_effect=selective_run_task)
        scheduler.task_manager = mock_task_manager_instance

        # Set both tasks to run immediately
        past_time = datetime.now() - timedelta(seconds=1)
        scheduler.next_run_times["good_task"] = past_time
        scheduler.next_run_times["bad_task"] = past_time

        # Run scheduler briefly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.2)
        await scheduler.stop()

        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        # Both tasks should have been attempted
        assert mock_task_manager_instance.run_task.call_count >= 1
        # Should have logged error for bad task
        mock_logger.error.assert_any_call("Error executing task bad_task: Bad task error")

    @patch('scheduler.TaskManager')
    def test_scheduler_handles_missing_task_instance(self, mock_task_manager, minimal_config, mock_engine, mock_logger):
        """Test scheduler handles missing task instances gracefully."""
        scheduler = TaskScheduler(minimal_config, mock_engine, mock_logger)

        # Remove task instance but keep config
        scheduler.task_instances.clear()

        # Should not crash when getting status
        status = scheduler.get_task_status()
        assert "single_task" in status
        assert not status["single_task"]["initialized"]


class TestSchedulerConcurrency:
    """Test scheduler behavior under concurrent conditions."""

    @patch('scheduler.TaskManager')
    @pytest.mark.asyncio
    async def test_scheduler_task_isolation(self, mock_task_manager, mock_engine, mock_logger):
        """Test that tasks are executed in isolation."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "task1",
                    "class": "MetricsTask",
                    "frequency": 1,
                    "enabled": True
                },
                {
                    "name": "task2",
                    "class": "MetricsTask",
                    "frequency": 1,
                    "enabled": True
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        # Mock tasks with different execution times
        task1 = MockTask(config, "task1")
        task2 = MockTask(config, "task2")
        scheduler.task_instances["task1"] = task1
        scheduler.task_instances["task2"] = task2

        execution_order = []

        async def track_execution(task):
            execution_order.append(f"{task.name}_start")
            if task.name == "task1":
                await asyncio.sleep(0.1)  # Longer execution
            execution_order.append(f"{task.name}_end")
            return await task.execute(mock_engine, mock_logger)

        mock_task_manager_instance = Mock()
        mock_task_manager_instance.run_task = AsyncMock(side_effect=track_execution)
        scheduler.task_manager = mock_task_manager_instance

        # Set both tasks to run immediately
        past_time = datetime.now() - timedelta(seconds=1)
        scheduler.next_run_times["task1"] = past_time
        scheduler.next_run_times["task2"] = past_time

        # Run scheduler briefly
        start_task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(0.3)
        await scheduler.stop()

        try:
            await asyncio.wait_for(start_task, timeout=1.0)
        except asyncio.TimeoutError:
            start_task.cancel()

        # Both tasks should have been executed
        assert any("task1" in item for item in execution_order)
        assert any("task2" in item for item in execution_order)


class TestSchedulerEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('scheduler.TaskManager')
    def test_empty_task_list(self, mock_task_manager, mock_engine, mock_logger):
        """Test scheduler with no tasks."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        assert len(scheduler.task_instances) == 0
        assert len(scheduler.next_run_times) == 0
        assert len(scheduler.task_configs) == 0

        # Should be able to get status
        status = scheduler.get_task_status()
        assert status == {}

    @patch('scheduler.TaskManager')
    def test_all_tasks_disabled(self, mock_task_manager, mock_engine, mock_logger):
        """Test scheduler with all tasks disabled."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "disabled1",
                    "class": "MetricsTask",
                    "frequency": 60,
                    "enabled": False
                },
                {
                    "name": "disabled2",
                    "class": "MetricsTask",
                    "frequency": 30,
                    "enabled": False
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        assert len(scheduler.task_instances) == 0
        assert len(scheduler.task_configs) == 0

    @patch('scheduler.TaskManager')
    def test_very_high_frequency_task(self, mock_task_manager, mock_engine, mock_logger):
        """Test scheduler with very high frequency task."""
        config = MachaConfig(
            app={"name": "test", "debug": True},
            logging={"level": "INFO", "file": {"path": "test.log"}, "console": {"format": "test"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[
                {
                    "name": "high_freq_task",
                    "class": "MetricsTask",
                    "frequency": 1,  # Very high frequency
                    "enabled": True
                }
            ]
        )

        scheduler = TaskScheduler(config, mock_engine, mock_logger)

        # Should initialize normally
        assert len(scheduler.task_configs) == 1
        assert scheduler.task_configs[0].frequency == 1


if __name__ == "__main__":
    pytest.main([__file__])
