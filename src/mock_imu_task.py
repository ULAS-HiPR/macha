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
from config import MachaConfig, ImuParameters


class MockImuTask(Task):
    """
    Mock IMU task for development and testing on non-Pi systems.

    Supports multiple mock strategies:
    1. "stationary" - minimal movement/vibration like aircraft on ground
    2. "flight" - simulate flight dynamics with realistic accelerations
    3. "vibration" - simulate motor/engine vibrations
    4. "static" - zero readings for testing
    5. "turbulence" - simulate turbulent flight conditions
    """

    def __init__(self, config: MachaConfig, mock_strategy: str = "auto"):
        super().__init__(config)

        # Find IMU task config
        imu_params = None
        for task_config in config.tasks:
            if task_config.class_name in ["ImuTask", "MockImuTask"] and task_config.name == getattr(self, 'task_name', 'imu'):
                imu_params = task_config.parameters
                break

        if imu_params is None:
            # Use default parameters if not found in config
            imu_params = ImuParameters()

        self.parameters = imu_params
        self.mock_strategy = self._determine_strategy(mock_strategy)

        # Mock-specific state variables
        self.start_time = time.time()

        # Flight dynamics state
        self.velocity = [0.0, 0.0, 0.0]  # m/s [x, y, z]
        self.angular_velocity = [0.0, 0.0, 0.0]  # rad/s [roll, pitch, yaw]
        self.attitude = [0.0, 0.0, 0.0]  # rad [roll, pitch, yaw]

        # Base accelerations (gravity when stationary)
        self.gravity = 9.81  # m/s²
        self.base_accel = [0.0, 0.0, self.gravity]

        # Sensor characteristics
        self.accel_noise_std = 0.02  # m/s² RMS noise
        self.gyro_noise_std = 0.001  # rad/s RMS noise
        self.temp_base = 25.0 + random.uniform(-5.0, 5.0)

        # Flight profile
        self.flight_phase = "ground"
        self.flight_duration = 300.0  # 5 minutes
        self.max_speed = 30.0  # m/s

        # Vibration parameters
        self.vibration_freq = [50.0, 75.0, 120.0]  # Hz
        self.vibration_amplitude = [0.5, 0.3, 0.2]  # m/s²

    def _determine_strategy(self, strategy: str) -> str:
        """Determine the best mock strategy."""
        if strategy == "auto":
            # Default to stationary for most development
            return "stationary"
        return strategy

    def _generate_stationary_data(self) -> Dict[str, Optional[float]]:
        """Generate IMU data for stationary aircraft with minimal movement."""
        # Small random movements and vibrations
        accel_x = random.gauss(0.0, 0.1)  # Small lateral movements
        accel_y = random.gauss(0.0, 0.1)  # Small longitudinal movements
        accel_z = random.gauss(self.gravity, 0.05)  # Gravity + small vertical movements

        # Very small angular rates (wind gusts, etc.)
        gyro_x = random.gauss(0.0, 0.01)  # Small roll movements
        gyro_y = random.gauss(0.0, 0.01)  # Small pitch movements
        gyro_z = random.gauss(0.0, 0.005)  # Very small yaw movements

        # Temperature with slow variations
        elapsed = time.time() - self.start_time
        temp_variation = 2.0 * math.sin(elapsed / 1800.0)  # 30-minute cycle
        temperature_celsius = self.temp_base + temp_variation + random.gauss(0.0, 0.2)

        return {
            "accel_x": round(accel_x, 3),
            "accel_y": round(accel_y, 3),
            "accel_z": round(accel_z, 3),
            "gyro_x": round(gyro_x, 4),
            "gyro_y": round(gyro_y, 4),
            "gyro_z": round(gyro_z, 4),
            "temperature_celsius": round(temperature_celsius, 2)
        }

    def _generate_flight_data(self) -> Dict[str, Optional[float]]:
        """Generate IMU data simulating realistic flight dynamics."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Normalize time to flight duration
        flight_progress = (elapsed % self.flight_duration) / self.flight_duration

        # Define flight phases with different dynamics
        if flight_progress < 0.1:  # Ground/taxi
            self.flight_phase = "ground"
            target_accel = [random.gauss(0.0, 0.2), random.gauss(0.0, 0.3), self.gravity]
            target_gyro = [random.gauss(0.0, 0.02), random.gauss(0.0, 0.02), random.gauss(0.0, 0.05)]

        elif flight_progress < 0.2:  # Takeoff
            self.flight_phase = "takeoff"
            # Strong forward acceleration and pitch up
            phase_progress = (flight_progress - 0.1) / 0.1
            accel_forward = 5.0 * phase_progress
            pitch_rate = 0.3 * phase_progress
            target_accel = [random.gauss(0.0, 0.5), accel_forward, self.gravity + random.gauss(0.0, 1.0)]
            target_gyro = [random.gauss(0.0, 0.1), pitch_rate, random.gauss(0.0, 0.05)]

        elif flight_progress < 0.35:  # Climb
            self.flight_phase = "climb"
            # Sustained climb with moderate accelerations
            target_accel = [random.gauss(0.0, 0.3), random.gauss(2.0, 0.5), self.gravity * 0.9]
            target_gyro = [random.gauss(0.0, 0.05), random.gauss(0.1, 0.02), random.gauss(0.0, 0.03)]

        elif flight_progress < 0.65:  # Cruise
            self.flight_phase = "cruise"
            # Steady flight with small corrections
            target_accel = [random.gauss(0.0, 0.2), random.gauss(0.0, 0.3), self.gravity + random.gauss(0.0, 0.2)]
            target_gyro = [random.gauss(0.0, 0.03), random.gauss(0.0, 0.02), random.gauss(0.0, 0.02)]

        elif flight_progress < 0.8:  # Descent
            self.flight_phase = "descent"
            # Controlled descent
            target_accel = [random.gauss(0.0, 0.4), random.gauss(-1.0, 0.5), self.gravity * 1.1]
            target_gyro = [random.gauss(0.0, 0.06), random.gauss(-0.05, 0.03), random.gauss(0.0, 0.04)]

        else:  # Approach and landing
            self.flight_phase = "landing"
            # Variable accelerations during landing
            phase_progress = (flight_progress - 0.8) / 0.2
            decel = -3.0 * phase_progress
            target_accel = [random.gauss(0.0, 0.8), decel, self.gravity + random.gauss(0.0, 0.5)]
            target_gyro = [random.gauss(0.0, 0.1), random.gauss(-0.1, 0.05), random.gauss(0.0, 0.08)]

        # Add sensor noise
        accel_x = target_accel[0] + random.gauss(0.0, self.accel_noise_std)
        accel_y = target_accel[1] + random.gauss(0.0, self.accel_noise_std)
        accel_z = target_accel[2] + random.gauss(0.0, self.accel_noise_std)

        gyro_x = target_gyro[0] + random.gauss(0.0, self.gyro_noise_std)
        gyro_y = target_gyro[1] + random.gauss(0.0, self.gyro_noise_std)
        gyro_z = target_gyro[2] + random.gauss(0.0, self.gyro_noise_std)

        # Temperature varies with altitude and airspeed
        temperature_celsius = self.temp_base - 0.5 * (flight_progress * 10) + random.gauss(0.0, 0.5)

        return {
            "accel_x": round(accel_x, 3),
            "accel_y": round(accel_y, 3),
            "accel_z": round(accel_z, 3),
            "gyro_x": round(gyro_x, 4),
            "gyro_y": round(gyro_y, 4),
            "gyro_z": round(gyro_z, 4),
            "temperature_celsius": round(temperature_celsius, 2)
        }

    def _generate_vibration_data(self) -> Dict[str, Optional[float]]:
        """Generate IMU data with motor/engine vibrations."""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Base acceleration (gravity)
        accel_x = 0.0
        accel_y = 0.0
        accel_z = self.gravity

        # Add multiple frequency vibrations
        for i, (freq, amp) in enumerate(zip(self.vibration_freq, self.vibration_amplitude)):
            phase = 2 * math.pi * freq * elapsed + random.uniform(0, 2*math.pi)
            vibration = amp * math.sin(phase)

            if i % 3 == 0:
                accel_x += vibration
            elif i % 3 == 1:
                accel_y += vibration
            else:
                accel_z += vibration

        # Gyro affected by vibrations too
        gyro_x = 0.1 * math.sin(2 * math.pi * 45 * elapsed) + random.gauss(0.0, 0.02)
        gyro_y = 0.1 * math.sin(2 * math.pi * 62 * elapsed + math.pi/3) + random.gauss(0.0, 0.02)
        gyro_z = 0.05 * math.sin(2 * math.pi * 38 * elapsed + math.pi/2) + random.gauss(0.0, 0.01)

        # Temperature with vibration heating effects
        temperature_celsius = self.temp_base + 2.0 + random.gauss(0.0, 0.3)

        return {
            "accel_x": round(accel_x, 3),
            "accel_y": round(accel_y, 3),
            "accel_z": round(accel_z, 3),
            "gyro_x": round(gyro_x, 4),
            "gyro_y": round(gyro_y, 4),
            "gyro_z": round(gyro_z, 4),
            "temperature_celsius": round(temperature_celsius, 2)
        }

    def _generate_turbulence_data(self) -> Dict[str, Optional[float]]:
        """Generate IMU data with turbulent flight conditions."""
        # High-frequency random accelerations simulating turbulence
        turbulence_intensity = 2.0

        accel_x = random.gauss(0.0, turbulence_intensity)
        accel_y = random.gauss(0.0, turbulence_intensity)
        accel_z = self.gravity + random.gauss(0.0, turbulence_intensity)

        # Angular rates with higher noise
        gyro_x = random.gauss(0.0, 0.2)
        gyro_y = random.gauss(0.0, 0.2)
        gyro_z = random.gauss(0.0, 0.1)

        # Temperature with rapid variations
        temperature_celsius = self.temp_base + random.gauss(0.0, 2.0)

        return {
            "accel_x": round(accel_x, 3),
            "accel_y": round(accel_y, 3),
            "accel_z": round(accel_z, 3),
            "gyro_x": round(gyro_x, 4),
            "gyro_y": round(gyro_y, 4),
            "gyro_z": round(gyro_z, 4),
            "temperature_celsius": round(temperature_celsius, 2)
        }

    def _generate_static_data(self) -> Dict[str, Optional[float]]:
        """Generate static IMU readings for testing."""
        return {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": self.gravity,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "temperature_celsius": self.temp_base
        }

    def _read_mock_sensor_data(self, logger: logging.Logger) -> Dict[str, Optional[float]]:
        """Read mock sensor data based on selected strategy."""
        try:
            if self.mock_strategy == "stationary":
                data = self._generate_stationary_data()
            elif self.mock_strategy == "flight":
                data = self._generate_flight_data()
            elif self.mock_strategy == "vibration":
                data = self._generate_vibration_data()
            elif self.mock_strategy == "turbulence":
                data = self._generate_turbulence_data()
            elif self.mock_strategy == "static":
                data = self._generate_static_data()
            else:
                # Default to stationary
                data = self._generate_stationary_data()

            logger.debug(f"Mock IMU reading ({self.mock_strategy}): "
                        f"accel=({data['accel_x']:.3f},{data['accel_y']:.3f},{data['accel_z']:.3f}) m/s², "
                        f"gyro=({data['gyro_x']:.4f},{data['gyro_y']:.4f},{data['gyro_z']:.4f}) rad/s, "
                        f"temp={data['temperature_celsius']:.2f}°C")

            if self.mock_strategy == "flight":
                logger.debug(f"Flight phase: {self.flight_phase}")

            return data

        except Exception as e:
            logger.error(f"Failed to generate mock IMU data: {e}")
            return {
                "accel_x": None,
                "accel_y": None,
                "accel_z": None,
                "gyro_x": None,
                "gyro_y": None,
                "gyro_z": None,
                "temperature_celsius": None
            }

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the mock IMU reading task."""
        logger.info(f"Executing mock {self.name} task (strategy: {self.mock_strategy})")

        # Read mock sensor data
        sensor_data = self._read_mock_sensor_data(logger)

        # Check if we got valid data
        if all(value is None for value in sensor_data.values()):
            error_msg = "Failed to generate valid mock IMU data"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": sensor_data,
                "mock_strategy": self.mock_strategy
            }

        # Store data in database (same table as real IMU task)
        try:
            async with engine.connect() as conn:
                await conn.execute(
                    text("""
                    INSERT INTO imu_readings
                    (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temperature_celsius, sensor_config)
                    VALUES (:accel_x, :accel_y, :accel_z, :gyro_x, :gyro_y, :gyro_z, :temperature, :config)
                    """),
                    {
                        "accel_x": sensor_data["accel_x"],
                        "accel_y": sensor_data["accel_y"],
                        "accel_z": sensor_data["accel_z"],
                        "gyro_x": sensor_data["gyro_x"],
                        "gyro_y": sensor_data["gyro_y"],
                        "gyro_z": sensor_data["gyro_z"],
                        "temperature": sensor_data["temperature_celsius"],
                        "config": json.dumps({
                            "mock_strategy": self.mock_strategy,
                            "i2c_bus": self.parameters.i2c_bus,
                            "address": self.parameters.address,
                            "accel_range": self.parameters.accel_range,
                            "gyro_range": self.parameters.gyro_range,
                            "flight_phase": getattr(self, 'flight_phase', 'unknown'),
                            "is_mock": True
                        })
                    }
                )
                await conn.commit()

            logger.info(f"Stored mock IMU reading: "
                       f"accel=({sensor_data['accel_x']:.3f},{sensor_data['accel_y']:.3f},{sensor_data['accel_z']:.3f}) m/s², "
                       f"gyro=({sensor_data['gyro_x']:.4f},{sensor_data['gyro_y']:.4f},{sensor_data['gyro_z']:.4f}) rad/s, "
                       f"temp={sensor_data['temperature_celsius']:.2f}°C")

            return {
                "success": True,
                "error": None,
                "data": sensor_data,
                "mock_strategy": self.mock_strategy,
                "flight_phase": getattr(self, 'flight_phase', 'unknown')
            }

        except Exception as e:
            error_msg = f"Failed to store mock IMU data in database: {e}"
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
        self.velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.attitude = [0.0, 0.0, 0.0]
        self.flight_phase = "ground"

    def set_mock_strategy(self, strategy: str):
        """Change mock strategy at runtime."""
        self.mock_strategy = strategy
        self.reset_simulation()

    def get_simulation_state(self) -> Dict:
        """Get current simulation state for debugging."""
        return {
            "mock_strategy": self.mock_strategy,
            "flight_phase": getattr(self, 'flight_phase', 'unknown'),
            "velocity": getattr(self, 'velocity', [0.0, 0.0, 0.0]),
            "angular_velocity": getattr(self, 'angular_velocity', [0.0, 0.0, 0.0]),
            "elapsed_time": time.time() - self.start_time
        }
