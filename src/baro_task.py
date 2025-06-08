import json
import math
import asyncio
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

from task import Task
from config import MachaConfig, BarometerParameters

try:
    import board
    import busio
    import adafruit_bmp3xx
    SENSOR_AVAILABLE = True
except ImportError as e:
    SENSOR_AVAILABLE = False


class BaroTask(Task):
    """Task for reading data from BMP390 barometer sensor."""
    
    def __init__(self, config: MachaConfig):
        super().__init__(config)
        self.sensor = None
        self.i2c = None
        self.initialized = False
        self.parameters: Optional[BarometerParameters] = None
        
        # Get task-specific configuration
        for task_config in config.tasks:
            if task_config.class_name == "BaroTask" and task_config.name == getattr(self, 'task_name', 'barometer'):
                self.parameters = task_config.parameters
                break
        
        if self.parameters is None:
            # Use default parameters if not found in config
            self.parameters = BarometerParameters()

    async def _initialize_sensor(self, logger: logging.Logger) -> bool:
        """Initialize the BMP390 sensor."""
        if not SENSOR_AVAILABLE:
            logger.error("Barometer sensor libraries not available. Install adafruit-blinka and adafruit-circuitpython-bmp3xx")
            return False
        
        if self.initialized:
            return True
        
        try:
            # Initialize I2C bus
            if self.parameters.i2c_bus == 1:
                self.i2c = busio.I2C(board.SCL, board.SDA)
            else:
                logger.error(f"I2C bus {self.parameters.i2c_bus} not supported. Only bus 1 is currently supported.")
                return False
            
            # Initialize the sensor
            self.sensor = adafruit_bmp3xx.BMP3XX_I2C(self.i2c, address=self.parameters.address)
            
            # Configure sensor settings for optimal performance
            self.sensor.pressure_oversampling = 8
            self.sensor.temperature_oversampling = 2
            self.sensor.filter_coefficient = 4
            
            self.initialized = True
            logger.info(f"BMP390 barometer sensor initialized on I2C bus {self.parameters.i2c_bus}, address 0x{self.parameters.address:02X}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BMP390 sensor: {e}")
            self.initialized = False
            return False

    def _read_sensor_data(self, logger: logging.Logger) -> Dict[str, Optional[float]]:
        """Read data from the barometer sensor."""
        if not self.initialized or not self.sensor:
            logger.warning("Barometer sensor not initialized")
            return {
                "pressure_hpa": None,
                "temperature_celsius": None,
                "altitude_meters": None
            }
        
        try:
            # Read raw sensor data
            pressure_hpa = self.sensor.pressure
            temperature_celsius = self.sensor.temperature
            
            # Calculate altitude using barometric formula
            # altitude = 44330 * (1 - (pressure / sea_level_pressure)^(1/5.255))
            altitude_meters = None
            if pressure_hpa and self.parameters.sea_level_pressure:
                altitude_meters = 44330.0 * (1.0 - math.pow(pressure_hpa / self.parameters.sea_level_pressure, 1.0 / 5.255))
            
            logger.debug(f"Barometer reading: {pressure_hpa:.2f}hPa, {temperature_celsius:.2f}°C, {altitude_meters:.1f}m")
            
            return {
                "pressure_hpa": pressure_hpa,
                "temperature_celsius": temperature_celsius,
                "altitude_meters": altitude_meters
            }
            
        except Exception as e:
            logger.error(f"Failed to read from BMP390 sensor: {e}")
            return {
                "pressure_hpa": None,
                "temperature_celsius": None,
                "altitude_meters": None
            }



    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the barometer reading task."""
        logger.info(f"Executing {self.name} task")
        
        # Initialize sensor if not already done
        if not await self._initialize_sensor(logger):
            error_msg = "Failed to initialize barometer sensor"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
        

        
        # Read sensor data
        sensor_data = self._read_sensor_data(logger)
        
        # Check if we got valid data
        if all(value is None for value in sensor_data.values()):
            error_msg = "Failed to read valid data from barometer sensor"
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": sensor_data
            }
        
        # Store data in database
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
                            "i2c_bus": self.parameters.i2c_bus,
                            "address": self.parameters.address,
                            "sea_level_pressure": self.parameters.sea_level_pressure
                        })
                    }
                )
                await conn.commit()
            
            logger.info(f"Stored barometer reading: {sensor_data['pressure_hpa']:.2f}hPa, "
                       f"{sensor_data['temperature_celsius']:.2f}°C, {sensor_data['altitude_meters']:.1f}m")
            
            return {
                "success": True,
                "error": None,
                "data": sensor_data
            }
            
        except Exception as e:
            error_msg = f"Failed to store barometer data in database: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": sensor_data
            }

    def __del__(self):
        """Cleanup resources when task is destroyed."""
        if self.i2c:
            try:
                self.i2c.deinit()
            except Exception:
                pass