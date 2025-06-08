import json
import asyncio
from typing import Dict, Optional
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
import logging

from task import Task
from config import MachaConfig, ImuParameters

try:
    import board
    import busio
    import adafruit_lsm6ds
    SENSOR_AVAILABLE = True
except ImportError as e:
    SENSOR_AVAILABLE = False


class ImuTask(Task):
    """Task for reading data from LSM6DSOX IMU sensor (accelerometer + gyroscope)."""
    
    def __init__(self, config: MachaConfig):
        super().__init__(config)
        self.sensor = None
        self.i2c = None
        self.initialized = False
        self.parameters: Optional[ImuParameters] = None
        
        # Get task-specific configuration
        for task_config in config.tasks:
            if task_config.class_name == "ImuTask" and task_config.name == getattr(self, 'task_name', 'imu'):
                self.parameters = task_config.parameters
                break
        
        if self.parameters is None:
            # Use default parameters if not found in config
            self.parameters = ImuParameters()

    async def _initialize_sensor(self, logger: logging.Logger) -> bool:
        """Initialize the LSM6DSOX sensor."""
        if not SENSOR_AVAILABLE:
            logger.error("IMU sensor libraries not available. Install adafruit-blinka and adafruit-circuitpython-lsm6ds")
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
            self.sensor = adafruit_lsm6ds.LSM6DSOX(self.i2c, address=self.parameters.address)
            
            # Configure accelerometer range
            if self.parameters.accel_range == "2G":
                self.sensor.accelerometer_range = adafruit_lsm6ds.Range.RANGE_2G
            elif self.parameters.accel_range == "4G":
                self.sensor.accelerometer_range = adafruit_lsm6ds.Range.RANGE_4G
            elif self.parameters.accel_range == "8G":
                self.sensor.accelerometer_range = adafruit_lsm6ds.Range.RANGE_8G
            elif self.parameters.accel_range == "16G":
                self.sensor.accelerometer_range = adafruit_lsm6ds.Range.RANGE_16G
            
            # Configure gyroscope range
            if self.parameters.gyro_range == "125DPS":
                self.sensor.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_125_DPS
            elif self.parameters.gyro_range == "250DPS":
                self.sensor.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_250_DPS
            elif self.parameters.gyro_range == "500DPS":
                self.sensor.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_500_DPS
            elif self.parameters.gyro_range == "1000DPS":
                self.sensor.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_1000_DPS
            elif self.parameters.gyro_range == "2000DPS":
                self.sensor.gyro_range = adafruit_lsm6ds.GyroRange.RANGE_2000_DPS
            
            # Set data rates to 104 Hz for both sensors
            self.sensor.accelerometer_data_rate = adafruit_lsm6ds.Rate.RATE_104_HZ
            self.sensor.gyro_data_rate = adafruit_lsm6ds.Rate.RATE_104_HZ
            
            self.initialized = True
            logger.info(f"LSM6DSOX IMU sensor initialized on I2C bus {self.parameters.i2c_bus}, address 0x{self.parameters.address:02X}")
            logger.info(f"Accelerometer range: {self.parameters.accel_range}, Gyroscope range: {self.parameters.gyro_range}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LSM6DSOX sensor: {e}")
            self.initialized = False
            return False

    def _read_sensor_data(self, logger: logging.Logger) -> Dict[str, Optional[float]]:
        """Read data from the IMU sensor."""
        if not self.initialized or not self.sensor:
            logger.warning("IMU sensor not initialized")
            return {
                "accel_x": None,
                "accel_y": None,
                "accel_z": None,
                "gyro_x": None,
                "gyro_y": None,
                "gyro_z": None,
                "temperature_celsius": None
            }
        
        try:
            # Read accelerometer data (m/s²)
            accel_x, accel_y, accel_z = self.sensor.acceleration
            
            # Read gyroscope data (rad/s)
            gyro_x, gyro_y, gyro_z = self.sensor.gyro
            
            # Read temperature (°C)
            temperature_celsius = self.sensor.temperature
            
            logger.debug(f"IMU reading: accel=({accel_x:.2f},{accel_y:.2f},{accel_z:.2f}) m/s², "
                        f"gyro=({gyro_x:.2f},{gyro_y:.2f},{gyro_z:.2f}) rad/s, temp={temperature_celsius:.2f}°C")
            
            return {
                "accel_x": accel_x,
                "accel_y": accel_y,
                "accel_z": accel_z,
                "gyro_x": gyro_x,
                "gyro_y": gyro_y,
                "gyro_z": gyro_z,
                "temperature_celsius": temperature_celsius
            }
            
        except Exception as e:
            logger.error(f"Failed to read from LSM6DSOX sensor: {e}")
            return {
                "accel_x": None,
                "accel_y": None,
                "accel_z": None,
                "gyro_x": None,
                "gyro_y": None,
                "gyro_z": None,
                "temperature_celsius": None
            }

    async def _create_table(self, engine: AsyncEngine):
        """Create the IMU readings table if it doesn't exist."""
        async with engine.connect() as conn:
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS imu_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accel_x REAL,
                    accel_y REAL,
                    accel_z REAL,
                    gyro_x REAL,
                    gyro_y REAL,
                    gyro_z REAL,
                    temperature_celsius REAL,
                    sensor_config TEXT
                )
                """)
            )
            await conn.commit()

    async def execute(self, engine: AsyncEngine, logger: logging.Logger) -> dict:
        """Execute the IMU reading task."""
        logger.info(f"Executing {self.name} task")
        
        # Initialize sensor if not already done
        if not await self._initialize_sensor(logger):
            error_msg = "Failed to initialize IMU sensor"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
        
        # Create table if needed
        await self._create_table(engine)
        
        # Read sensor data
        sensor_data = self._read_sensor_data(logger)
        
        # Check if we got valid data
        if all(value is None for value in sensor_data.values()):
            error_msg = "Failed to read valid data from IMU sensor"
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
                            "i2c_bus": self.parameters.i2c_bus,
                            "address": self.parameters.address,
                            "accel_range": self.parameters.accel_range,
                            "gyro_range": self.parameters.gyro_range
                        })
                    }
                )
                await conn.commit()
            
            logger.info(f"Stored IMU reading: accel=({sensor_data['accel_x']:.2f},{sensor_data['accel_y']:.2f},{sensor_data['accel_z']:.2f}) m/s², "
                       f"gyro=({sensor_data['gyro_x']:.2f},{sensor_data['gyro_y']:.2f},{sensor_data['gyro_z']:.2f}) rad/s, "
                       f"temp={sensor_data['temperature_celsius']:.2f}°C")
            
            return {
                "success": True,
                "error": None,
                "data": sensor_data
            }
            
        except Exception as e:
            error_msg = f"Failed to store IMU data in database: {e}"
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