from task import Task
from sqlalchemy.ext.asyncio import AsyncEngine
import logging
import psutil
import platform
import json
from datetime import datetime
import os
from config import MachaConfig


class MetricsTask(Task):
    """Task to collect system metrics (CPU, temp, storage, RAM, uptime)."""

    def __init__(self, config: MachaConfig):
        super().__init__(config)

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        logger.info("Collecting system metrics")

        metrics = {}

        # CPU stats
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        metrics["cpu_count"] = psutil.cpu_count()

        # Temperature (Raspberry Pi specific)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read().strip()) / 1000.0
                metrics["temperature_c"] = temp
        except FileNotFoundError:
            logger.warning("Temperature sensor not available")
            metrics["temperature_c"] = None

        # Storage
        disk = psutil.disk_usage("/")
        metrics["storage_total_gb"] = disk.total / (1024**3)
        metrics["storage_used_gb"] = disk.used / (1024**3)
        metrics["storage_free_gb"] = disk.free / (1024**3)

        # RAM
        memory = psutil.virtual_memory()
        metrics["ram_total_gb"] = memory.total / (1024**3)
        metrics["ram_used_gb"] = memory.used / (1024**3)
        metrics["ram_free_gb"] = memory.free / (1024**3)

        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        metrics["uptime_seconds"] = (datetime.now() - boot_time).total_seconds()

        # System info
        metrics["hostname"] = platform.node()
        metrics["system"] = platform.system()
        metrics["release"] = platform.release()

        logger.debug(f"Collected metrics: {json.dumps(metrics, indent=2)}")
        return metrics
