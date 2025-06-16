import asyncio
from typing import Dict
from datetime import datetime, timedelta
import logging
from sqlalchemy.ext.asyncio import AsyncEngine

from task import Task, TaskManager
from metrics_task import MetricsTask
from camera_task import CameraTask
from baro_task import BaroTask
from imu_task import ImuTask
from ai_task import AiTask
from config import MachaConfig, get_enabled_tasks


class TaskScheduler:
    """Scheduler to run tasks at their configured frequencies."""

    def __init__(
        self, config: MachaConfig, engine: AsyncEngine, logger: logging.Logger
    ):
        self.config = config
        self.engine = engine
        self.logger = logger
        self.task_manager = TaskManager(engine, logger)
        self.running = False
        self.task_instances: Dict[str, Task] = {}
        self.next_run_times: Dict[str, datetime] = {}
        self.task_configs = get_enabled_tasks(config)

        # Initialize task instances
        self._initialize_tasks()

    def _initialize_tasks(self):
        """Initialize task instances based on configuration."""
        task_classes = {
            "MetricsTask": MetricsTask, 
            "CameraTask": CameraTask,
            "BaroTask": BaroTask,
            "ImuTask": ImuTask,
            "AiTask": AiTask
        }

        for task_config in self.task_configs:
            task_class_name = task_config.class_name
            if task_class_name not in task_classes:
                self.logger.error(f"Unknown task class: {task_class_name}")
                continue

            try:
                task_class = task_classes[task_class_name]
                task_instance = task_class(self.config)
                self.task_instances[task_config.name] = task_instance

                # Set initial run time to now
                self.next_run_times[task_config.name] = datetime.now()

                self.logger.info(
                    f"Initialized task: {task_config.name} ({task_class_name})"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize task {task_config.name}: {e}")
                self.logger.error(f"Task {task_config.name} will be skipped")
                # Mark task as disabled by setting enabled to False
                task_config.enabled = False

    async def start(self):
        """Start the scheduler loop."""
        self.logger.info("Starting task scheduler")
        self.running = True

        while self.running:
            current_time = datetime.now()

            # Check each task to see if it's time to run
            for task_config in self.task_configs:
                task_name = task_config.name
                frequency = task_config.frequency

                if not task_config.enabled:
                    continue

                if task_name not in self.task_instances:
                    continue

                # Check if it's time to run this task
                if current_time >= self.next_run_times[task_name]:
                    self.logger.debug(f"Executing scheduled task: {task_name}")

                    # Run the task
                    try:
                        task_instance = self.task_instances[task_name]
                        await self.task_manager.run_task(task_instance)

                        # Schedule next run
                        self.next_run_times[task_name] = current_time + timedelta(
                            seconds=frequency
                        )
                        self.logger.debug(
                            f"Next run for {task_name}: {self.next_run_times[task_name]}"
                        )

                    except Exception as e:
                        self.logger.error(f"Error executing task {task_name}: {e}")
                        # Still schedule next run even if task failed
                        self.next_run_times[task_name] = current_time + timedelta(
                            seconds=frequency
                        )

                    await asyncio.sleep(0)

    async def stop(self):
        """Stop the scheduler."""
        self.logger.info("Stopping task scheduler")
        self.running = False

    def get_task_status(self) -> Dict[str, dict]:
        """Get status of all tasks."""
        status = {}
        current_time = datetime.now()

        for task_config in self.task_configs:
            task_name = task_config.name

            status[task_name] = {
                "enabled": task_config.enabled,
                "frequency": task_config.frequency,
                "class": task_config.class_name,
                "initialized": task_name in self.task_instances,
                "next_run": self.next_run_times.get(task_name),
                "seconds_until_next_run": None,
            }

            if task_name in self.next_run_times:
                time_diff = self.next_run_times[task_name] - current_time
                status[task_name]["seconds_until_next_run"] = max(
                    0, int(time_diff.total_seconds())
                )

        return status

    def add_task(self, task_config, task_instance: Task):
        """Dynamically add a new task to the scheduler."""
        task_name = task_config.name
        self.task_instances[task_name] = task_instance
        self.next_run_times[task_name] = datetime.now()
        self.task_configs.append(task_config)
        self.logger.info(f"Added new task to scheduler: {task_name}")

    def remove_task(self, task_name: str):
        """Remove a task from the scheduler."""
        if task_name in self.task_instances:
            del self.task_instances[task_name]
        if task_name in self.next_run_times:
            del self.next_run_times[task_name]

        # Remove from config list
        self.task_configs = [tc for tc in self.task_configs if tc.name != task_name]
        self.logger.info(f"Removed task from scheduler: {task_name}")

    def update_task_frequency(self, task_name: str, new_frequency: int):
        """Update the frequency of an existing task."""
        for task_config in self.task_configs:
            if task_config.name == task_name:
                task_config.frequency = new_frequency
                self.logger.info(
                    f"Updated frequency for {task_name}: {new_frequency} seconds"
                )
                break
