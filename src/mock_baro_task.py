import json
import math
import random
import time
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

from task import Task
from config import MachaConfig, BarometerParameters


class MockBaroTask(Task):
    """
    Mock barometer task for development and testing on non-Pi systems.

    Supports multiple mock strategies:
    1. "realistic" - simulate gradual weather-like pressure changes
    2. "flight" - simulate altitude changes during flight profile
    3. "static" - constant readings for testing
    4. "noisy" - add realistic sensor noise and drift
    """

    def __init__(self, config: MachaConfig, mock_strategy: str = "auto"):
        super().__init__(config)

        # Find barometer task config
        baro_params = None
        for task_config in config.tasks:
            if task_config.class_name in ["BaroTask", "MockBaroTask"] and task_config.name == getattr(self, 'task_name', 'barometer'):
                baro_params = task_config.parameters
                break

        if baro_params is None:
            # Use default parameters if not found in config
            baro_params = BarometerParameters()

        self.parameters = baro_params
        self.mock_strategy = self._determine_strategy(mock_strategy)

        # Mock-specific state variables
        self.start_time = time.time()
        self.base_pressure = self.parameters.sea_level_pressure
        self.current_altitude = 0.0
        self.flight_phase = "ground"  # ground, takeoff, climb, cruise, descent, landing

        # Noise and drift simulation
        self.pressure_drift = 0.0
        self.temperature_offset = random.uniform(-2.0, 2.0)

        # Flight profile parameters
        self.target_altitude = 100.0  # meters
        self.max_altitude = 500.0
        self.flight_duration = 300.0  # 5 minutes total flight

    def _determine_strategy(self, strategy: str) -> str:
        """Determine the best mock strategy."""
        if strategy == "auto":
            # Default to realistic for most development
            return "realistic"
        return strategy

    def _generate_realistic_data(self) -> Dict[str, Optional[float]]:
        """Generate realistic barometer data with weather-like variations."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Simulate slow pressure changes like weather patterns
        # Use sine waves with different periods to simulate weather fronts
        pressure_variation = (
            2.0 * math.sin(elapsed / 3600.0) +  # 1-hour cycle
            1.0 * math.sin(elapsed / 1800.0) +  # 30-minute cycle
            0.5 * math.sin(elapsed / 900.0)     # 15-minute cycle
        )

        # Add some random noise
        noise = random.uniform(-0.2, 0.2)

        pressure_hpa = self.base_pressure + pressure_variation + noise + self.pressure_drift

        # Temperature varies with time of day and pressure
        base_temp = 20.0 + self.temperature_offset
        temp_variation = 3.0 * math.sin(elapsed / 7200.0)  # 2-hour cycle
        temperature_celsius = base_temp + temp_variation + random.uniform(-0.5, 0.5)

        # Calculate altitude using barometric formula
        altitude_meters = 44330.0 * (1.0 - math.pow(pressure_hpa / self.parameters.sea_level_pressure, 1.0 / 5.255))

        # Add small drift over time
        self.pressure_drift += random.uniform(-0.001, 0.001)
        self.pressure_drift = max(-2.0, min(2.0, self.pressure_drift))  # Limit drift

        return {
            "pressure_hpa": round(pressure_hpa, 2),
            "temperature_celsius": round(temperature_celsius, 2),
            "altitude_meters": round(altitude_meters, 1)
        }

    def _generate_flight_data(self) -> Dict[str, Optional[float]]:
        """Generate barometer data simulating a flight profile."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Normalize time to flight duration
        flight_progress = (elapsed % self.flight_duration) / self.flight_duration

        # Define flight phases and altitudes
        if flight_progress < 0.1:  # 0-10%: Ground/taxi
            self.flight_phase = "ground"
            target_altitude = random.uniform(0, 5)
        elif flight_progress < 0.25:  # 10-25%: Takeoff and initial climb
            self.flight_phase = "takeoff"
            phase_progress = (flight_progress - 0.1) / 0.15
            target_altitude = phase_progress * (self.max_altitude * 0.3)
        elif flight_progress < 0.4:  # 25-40%: Climb to cruise
            self.flight_phase = "climb"
            phase_progress = (flight_progress - 0.25) / 0.15
            target_altitude = (self.max_altitude * 0.3) + phase_progress * (self.max_altitude * 0.7)
        elif flight_progress < 0.7:  # 40-70%: Cruise
            self.flight_phase = "cruise"
            target_altitude = self.max_altitude + random.uniform(-10, 10)
        elif flight_progress < 0.85:  # 70-85%: Descent
            self.flight_phase = "descent"
            phase_progress = (flight_progress - 0.7) / 0.15
            target_altitude = self.max_altitude * (1.0 - phase_progress * 0.7)
        else:  # 85-100%: Final approach and landing
            self.flight_phase = "landing"
            phase_progress = (flight_progress - 0.85) / 0.15
            target_altitude = (self.max_altitude * 0.3) * (1.0 - phase_progress)

        # Smooth altitude changes
        altitude_diff = target_altitude - self.current_altitude
        self.current_altitude += altitude_diff * 0.1  # 10% towards target each update

        # Add turbulence/noise based on flight phase
        if self.flight_phase in ["takeoff", "landing"]:
            noise_factor = 2.0
        elif self.flight_phase in ["climb", "descent"]:
            noise_factor = 1.0
        else:
            noise_factor = 0.5

        altitude_noise = random.uniform(-noise_factor, noise_factor)
        final_altitude = self.current_altitude + altitude_noise

        # Calculate pressure from altitude
        pressure_hpa = self.parameters.sea_level_pressure * math.pow(1.0 - final_altitude / 44330.0, 5.255)

        # Temperature decreases with altitude (lapse rate ~6.5°C/1000m)
        base_temp = 15.0 + self.temperature_offset
        temperature_celsius = base_temp - (final_altitude * 0.0065) + random.uniform(-1.0, 1.0)

        return {
            "pressure_hpa": round(pressure_hpa, 2),
            "temperature_celsius": round(temperature_celsius, 2),
            "altitude_meters": round(final_altitude, 1)
        }

    def _generate_static_data(self) -> Dict[str, Optional[float]]:
        """Generate static barometer readings for testing."""
        return {
            "pressure_hpa": self.parameters.sea_level_pressure,
            "temperature_celsius": 20.0 + self.temperature_offset,
            "altitude_meters": 0.0
        }

    def _generate_noisy_data(self) -> Dict[str, Optional[float]]:
        """Generate noisy sensor data to simulate sensor issues."""
        base_pressure = self.parameters.sea_level_pressure

        # High noise and drift
        pressure_noise = random.uniform(-5.0, 5.0)
        self.pressure_drift += random.uniform(-0.01, 0.01)

        pressure_hpa = base_pressure + pressure_noise + self.pressure_drift

        # Temperature with high noise
        temperature_celsius = 20.0 + self.temperature_offset + random.uniform(-3.0, 3.0)

        # Calculate altitude (may be very noisy)
        altitude_meters = 44330.0 * (1.0 - math.pow(pressure_hpa / self.parameters.sea_level_pressure, 1.0 / 5.255))

        return {
            "pressure_hpa": round(pressure_hpa, 2),
            "temperature_celsius": round(temperature_celsius, 2),
            "altitude_meters": round(altitude_meters, 1)
        }

    def _read_mock_sensor_data(self, logger: logging.Logger) -> Dict[str, Optional[float]]:
        """Read mock sensor data based on selected strategy."""
        try:
            if self.mock_strategy == "realistic":
                data = self._generate_realistic_data()
            elif self.mock_strategy == "flight":
                data = self._generate_flight_data()
            elif self.mock_strategy == "static":
                data = self._generate_static_data()
            elif self.mock_strategy == "noisy":
                data = self._generate_noisy_data()
            else:
                # Default to realistic
                data = self._generate_realistic_data()

            logger.debug(f"Mock barometer reading ({self.mock_strategy}): "
                        f"{data['pressure_hpa']:.2f}hPa, {data['temperature_celsius']:.2f}°C, "
                        f"{data['altitude_meters']:.1f}m")

            if self.mock_strategy == "flight":
                logger.debug(f"Flight phase: {self.flight_phase}")

            return data

        except Exception as e:
            logger.error(f"Failed to generate mock barometer data: {e}")
            return {
                "pressure_hpa": None,
                "temperature_celsius": None,
                "altitude_meters": None
            }

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the mock barometer reading task."""
        logger.info(f"Executing mock {self.name} task (strategy: {self.mock_strategy})")

        # Read mock sensor data
        sensor_data = self._read_mock_sensor_data(logger)

        # Check if we got valid data
        if all(value is None for value in sensor_data.values()):
            error_msg = "Failed to generate valid mock barometer data"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": sensor_data,
                "mock_strategy": self.mock_strategy
            }

        # Store data in database (same table as real barometer task)
        try:
            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO barometer_readings
                    (pressure_hpa, temperature_celsius, altitude_meters, sea_level_pressure, sensor_config)
                    VALUES (:pressure, :temperature, :altitude, :sea_level_pressure, :config)
                    """),
                    {
                        "pressure": sensor_data["pressure_hpa"],
                        "temperature": sensor_data["temperature_celsius"],
                        "altitude": sensor_data["altitude_meters"],
                        "sea_level_pressure": self.parameters.sea_level_pressure,
                        "config": json.dumps({
                            "mock_strategy": self.mock_strategy,
                            "i2c_bus": self.parameters.i2c_bus,
                            "address": self.parameters.address,
                            "sea_level_pressure": self.parameters.sea_level_pressure,
                            "flight_phase": getattr(self, 'flight_phase', 'unknown'),
                            "is_mock": True
                        })
                    }
                )
                await conn.commit()

            logger.info(f"Stored mock barometer reading: {sensor_data['pressure_hpa']:.2f}hPa, "
                       f"{sensor_data['temperature_celsius']:.2f}°C, {sensor_data['altitude_meters']:.1f}m")

            return {
                "success": True,
                "error": None,
                "data": sensor_data,
                "mock_strategy": self.mock_strategy,
                "flight_phase": getattr(self, 'flight_phase', 'unknown')
            }

        except Exception as e:
            error_msg = f"Failed to store mock barometer data in database: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": sensor_data,
                "mock_strategy": self.mock_strategy
            }

    def reset_simulation(self):
        """Reset simulation state for testing."""
        self.start_time = time.time()
        self.current_altitude = 0.0
        self.flight_phase = "ground"
        self.pressure_drift = 0.0

    def set_mock_strategy(self, strategy: str):
        """Change mock strategy at runtime."""
        self.mock_strategy = strategy
        self.reset_simulation()

    def get_simulation_state(self) -> Dict:
        """Get current simulation state for debugging."""
        return {
            "mock_strategy": self.mock_strategy,
            "flight_phase": getattr(self, 'flight_phase', 'unknown'),
            "current_altitude": getattr(self, 'current_altitude', 0.0),
            "pressure_drift": getattr(self, 'pressure_drift', 0.0),
            "elapsed_time": time.time() - self.start_time
        }
