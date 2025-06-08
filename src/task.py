from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import AsyncEngine
import logging
from sqlalchemy import text
import asyncio
from config import MachaConfig


class Task(ABC):
    """Base class for tasks."""

    def __init__(self, config: MachaConfig):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the task and return results."""
        pass


class TaskManager:
    """Manages task execution and database storage."""

    def __init__(self, engine: AsyncEngine, logger: logging.Logger):
        self.engine = engine
        self.logger = logger

    async def run_task(self, task: Task):
        """Run a task and store results in the database."""
        self.logger.info(f"Starting task: {task.name}")
        try:
            # Create tasks table if it doesn't exist
            async with self.engine.connect() as conn:
                await conn.execute(
                    text("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_name TEXT NOT NULL,
                        result TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                )
                await conn.commit()

            # Execute task
            result = await task.execute(self.engine, self.logger)

            # Store result
            async with self.engine.connect() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO tasks (task_name, result) VALUES (:name, :result)"
                    ),
                    {"name": task.name, "result": str(result)},
                )
                await conn.commit()

            self.logger.info(f"Task {task.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Task {task.name} failed: {e}")
            raise
