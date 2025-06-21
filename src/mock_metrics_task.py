import json
import math
import random
import time
import platform
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

from task import Task
from config import MachaConfig


class MockMetricsTask(Task):
    """
    Mock metrics task for development and testing on non-Pi systems.

    Supports multiple mock strategies:
    1. "realistic" - simulate realistic system load patterns
    2. "stressed" - high CPU/memory usage scenarios
    3. "idle" - low resource usage patterns
    4. "static" - constant values for testing
    5. "variable" - highly variable resource usage
    """

    def __init__(self, config: MachaConfig, mock_strategy: str = "auto"):
        super().__init__(config)

        self.mock_strategy = self._determine_strategy(mock_strategy)

        # Mock-specific state variables
        self.start_time = time.time()
        self.base_cpu_percent = 15.0
        self.base_temp = 45.0
        self.base_ram_usage = 0.3  # 30% base usage

        # System info (based on actual system or mock values)
        self.hostname = platform.node() or "mock-pi"
        self.system = platform.system() or "Linux"
        self.release = platform.release() or "5.15.0-rpi"
        self.cpu_count = 4  # Simulate Pi 4

        # Storage simulation (GB)
        self.storage_total_gb = 64.0
        self.storage_base_used = 25.0

        # RAM simulation (GB)
        self.ram_total_gb = 4.0

        # Load pattern parameters
        self.cpu_cycle_period = 1800.0  # 30 minutes
        self.temp_cycle_period = 3600.0  # 1 hour
        self.memory_cycle_period = 2400.0  # 40 minutes

    def _determine_strategy(self, strategy: str) -> str:
        """Determine the best mock strategy."""
        if strategy == "auto":
            # Default to realistic for most development
            return "realistic"
        return strategy

    def _generate_realistic_data(self) -> Dict[str, Optional[float]]:
        """Generate realistic system metrics with natural patterns."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # CPU usage with daily patterns and some randomness
        cpu_base_cycle = 10.0 * math.sin(elapsed / self.cpu_cycle_period) + self.base_cpu_percent
        cpu_noise = random.uniform(-5.0, 10.0)  # Asymmetric noise (spikes more common)
        cpu_percent = max(0.5, min(95.0, cpu_base_cycle + cpu_noise))

        # Temperature correlated with CPU usage + ambient variations
        temp_ambient = 5.0 * math.sin(elapsed / self.temp_cycle_period)  # Daily temp cycle
        temp_cpu_factor = (cpu_percent - 10) * 0.3  # Temperature rises with CPU usage
        temperature_c = self.base_temp + temp_ambient + temp_cpu_factor + random.uniform(-2.0, 2.0)
        temperature_c = max(25.0, min(85.0, temperature_c))  # Realistic Pi temp range

        # RAM usage with gradual changes and occasional spikes
        ram_base_cycle = 0.1 * math.sin(elapsed / self.memory_cycle_period)
        ram_usage_percent = self.base_ram_usage + ram_base_cycle + random.uniform(-0.05, 0.15)
        ram_usage_percent = max(0.15, min(0.85, ram_usage_percent))

        ram_used_gb = self.ram_total_gb * ram_usage_percent
        ram_free_gb = self.ram_total_gb - ram_used_gb

        # Storage usage grows very slowly over time
        storage_growth = elapsed / (24 * 3600) * 0.1  # 0.1 GB per day
        storage_used_gb = self.storage_base_used + storage_growth + random.uniform(-0.01, 0.05)
        storage_used_gb = max(10.0, min(self.storage_total_gb * 0.95, storage_used_gb))
        storage_free_gb = self.storage_total_gb - storage_used_gb

        # Uptime increases linearly
        uptime_seconds = elapsed

        return {
            "cpu_percent": round(cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "temperature_c": round(temperature_c, 1),
            "storage_total_gb": round(self.storage_total_gb, 1),
            "storage_used_gb": round(storage_used_gb, 1),
            "storage_free_gb": round(storage_free_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_free_gb": round(ram_free_gb, 1),
            "uptime_seconds": round(uptime_seconds, 0),
            "hostname": self.hostname,
            "system": self.system,
            "release": self.release
        }

    def _generate_stressed_data(self) -> Dict[str, Optional[float]]:
        """Generate metrics showing high system load."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # High CPU with variations
        cpu_percent = random.uniform(70.0, 95.0)

        # High temperature due to load
        temperature_c = random.uniform(65.0, 80.0)

        # High RAM usage
        ram_usage_percent = random.uniform(0.7, 0.9)
        ram_used_gb = self.ram_total_gb * ram_usage_percent
        ram_free_gb = self.ram_total_gb - ram_used_gb

        # Storage filling up
        storage_used_gb = random.uniform(self.storage_total_gb * 0.8, self.storage_total_gb * 0.95)
        storage_free_gb = self.storage_total_gb - storage_used_gb

        uptime_seconds = elapsed

        return {
            "cpu_percent": round(cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "temperature_c": round(temperature_c, 1),
            "storage_total_gb": round(self.storage_total_gb, 1),
            "storage_used_gb": round(storage_used_gb, 1),
            "storage_free_gb": round(storage_free_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_free_gb": round(ram_free_gb, 1),
            "uptime_seconds": round(uptime_seconds, 0),
            "hostname": self.hostname,
            "system": self.system,
            "release": self.release
        }

    def _generate_idle_data(self) -> Dict[str, Optional[float]]:
        """Generate metrics showing low system load."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Low CPU with small variations
        cpu_percent = random.uniform(2.0, 15.0)

        # Low temperature
        temperature_c = random.uniform(35.0, 50.0)

        # Low RAM usage
        ram_usage_percent = random.uniform(0.15, 0.3)
        ram_used_gb = self.ram_total_gb * ram_usage_percent
        ram_free_gb = self.ram_total_gb - ram_used_gb

        # Plenty of storage
        storage_used_gb = random.uniform(15.0, 25.0)
        storage_free_gb = self.storage_total_gb - storage_used_gb

        uptime_seconds = elapsed

        return {
            "cpu_percent": round(cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "temperature_c": round(temperature_c, 1),
            "storage_total_gb": round(self.storage_total_gb, 1),
            "storage_used_gb": round(storage_used_gb, 1),
            "storage_free_gb": round(storage_free_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_free_gb": round(ram_free_gb, 1),
            "uptime_seconds": round(uptime_seconds, 0),
            "hostname": self.hostname,
            "system": self.system,
            "release": self.release
        }

    def _generate_variable_data(self) -> Dict[str, Optional[float]]:
        """Generate highly variable metrics to test system responsiveness."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Highly variable CPU (simulate bursts)
        if random.random() < 0.3:  # 30% chance of CPU spike
            cpu_percent = random.uniform(80.0, 95.0)
        else:
            cpu_percent = random.uniform(5.0, 30.0)

        # Temperature follows CPU with delay
        temp_target = 40.0 + (cpu_percent - 10) * 0.4
        temperature_c = temp_target + random.uniform(-5.0, 5.0)
        temperature_c = max(25.0, min(85.0, temperature_c))

        # Variable RAM usage with occasional spikes
        if random.random() < 0.2:  # 20% chance of memory spike
            ram_usage_percent = random.uniform(0.7, 0.85)
        else:
            ram_usage_percent = random.uniform(0.2, 0.5)

        ram_used_gb = self.ram_total_gb * ram_usage_percent
        ram_free_gb = self.ram_total_gb - ram_used_gb

        # Variable storage usage
        storage_used_gb = self.storage_base_used + random.uniform(-2.0, 8.0)
        storage_used_gb = max(10.0, min(self.storage_total_gb * 0.9, storage_used_gb))
        storage_free_gb = self.storage_total_gb - storage_used_gb

        uptime_seconds = elapsed

        return {
            "cpu_percent": round(cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "temperature_c": round(temperature_c, 1),
            "storage_total_gb": round(self.storage_total_gb, 1),
            "storage_used_gb": round(storage_used_gb, 1),
            "storage_free_gb": round(storage_free_gb, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_used_gb": round(ram_used_gb, 1),
            "ram_free_gb": round(ram_free_gb, 1),
            "uptime_seconds": round(uptime_seconds, 0),
            "hostname": self.hostname,
            "system": self.system,
            "release": self.release
        }

    def _generate_static_data(self) -> Dict[str, Optional[float]]:
        """Generate static metrics for testing."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        return {
            "cpu_percent": 25.0,
            "cpu_count": self.cpu_count,
            "temperature_c": 45.0,
            "storage_total_gb": round(self.storage_total_gb, 1),
            "storage_used_gb": 20.0,
            "storage_free_gb": round(self.storage_total_gb - 20.0, 1),
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_used_gb": 1.2,
            "ram_free_gb": round(self.ram_total_gb - 1.2, 1),
            "uptime_seconds": round(elapsed, 0),
            "hostname": self.hostname,
            "system": self.system,
            "release": self.release
        }

    def _generate_mock_metrics(self, logger: logging.Logger) -> Dict[str, Optional[float]]:
        """Generate mock system metrics based on selected strategy."""
        try:
            if self.mock_strategy == "realistic":
                data = self._generate_realistic_data()
            elif self.mock_strategy == "stressed":
                data = self._generate_stressed_data()
            elif self.mock_strategy == "idle":
                data = self._generate_idle_data()
            elif self.mock_strategy == "variable":
                data = self._generate_variable_data()
            elif self.mock_strategy == "static":
                data = self._generate_static_data()
            else:
                # Default to realistic
                data = self._generate_realistic_data()

            logger.debug(f"Mock metrics ({self.mock_strategy}): "
                        f"CPU={data['cpu_percent']:.1f}%, "
                        f"Temp={data['temperature_c']:.1f}°C, "
                        f"RAM={data['ram_used_gb']:.1f}GB/{data['ram_total_gb']:.1f}GB")

            return data

        except Exception as e:
            logger.error(f"Failed to generate mock metrics: {e}")
            return {}

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute mock metrics collection."""
        logger.info(f"Collecting mock system metrics (strategy: {self.mock_strategy})")

        # Generate mock metrics
        metrics = self._generate_mock_metrics(logger)

        if not metrics:
            error_msg = "Failed to generate mock system metrics"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "mock_strategy": self.mock_strategy
            }

        # Store metrics in database (same table as real metrics task)
        try:
            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO system_metrics
                    (cpu_percent, cpu_count, temperature_c, storage_total_gb, storage_used_gb,
                     storage_free_gb, ram_total_gb, ram_used_gb, ram_free_gb, uptime_seconds,
                     hostname, system, release, raw_data)
                    VALUES (:cpu_percent, :cpu_count, :temperature_c, :storage_total_gb, :storage_used_gb,
                            :storage_free_gb, :ram_total_gb, :ram_used_gb, :ram_free_gb, :uptime_seconds,
                            :hostname, :system, :release, :raw_data)
                    """),
                    {
                        "cpu_percent": metrics.get("cpu_percent"),
                        "cpu_count": metrics.get("cpu_count"),
                        "temperature_c": metrics.get("temperature_c"),
                        "storage_total_gb": metrics.get("storage_total_gb"),
                        "storage_used_gb": metrics.get("storage_used_gb"),
                        "storage_free_gb": metrics.get("storage_free_gb"),
                        "ram_total_gb": metrics.get("ram_total_gb"),
                        "ram_used_gb": metrics.get("ram_used_gb"),
                        "ram_free_gb": metrics.get("ram_free_gb"),
                        "uptime_seconds": metrics.get("uptime_seconds"),
                        "hostname": metrics.get("hostname"),
                        "system": metrics.get("system"),
                        "release": metrics.get("release"),
                        "raw_data": json.dumps({
                            **metrics,
                            "mock_strategy": self.mock_strategy,
                            "is_mock": True
                        })
                    }
                )
                await conn.commit()

            logger.info(f"Stored mock system metrics: "
                       f"CPU={metrics['cpu_percent']:.1f}%, "
                       f"Temp={metrics['temperature_c']:.1f}°C, "
                       f"RAM={metrics['ram_used_gb']:.1f}GB")

            return {
                "success": True,
                "error": None,
                "data": metrics,
                "mock_strategy": self.mock_strategy
            }

        except Exception as e:
            error_msg = f"Failed to store mock metrics in database: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": metrics,
                "mock_strategy": self.mock_strategy
            }

    def reset_simulation(self):
        """Reset simulation state for testing."""
        self.start_time = time.time()

    def set_mock_strategy(self, strategy: str):
        """Change mock strategy at runtime."""
        self.mock_strategy = strategy
        self.reset_simulation()

    def get_simulation_state(self) -> Dict:
        """Get current simulation state for debugging."""
        return {
            "mock_strategy": self.mock_strategy,
            "elapsed_time": time.time() - self.start_time,
            "hostname": self.hostname,
            "system": self.system,
            "cpu_count": self.cpu_count
        }
