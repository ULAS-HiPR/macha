import yaml
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppConfig(BaseModel):
    name: str = Field(..., description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")


class LogColors(BaseModel):
    DEBUG: str = "cyan"
    INFO: str = "green"
    WARNING: str = "yellow"
    ERROR: str = "red"
    CRITICAL: str = "red,bg_white"


class LogFileConfig(BaseModel):
    path: str = Field(..., description="Log file path")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class LogConsoleConfig(BaseModel):
    format: str = Field(
        default="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    colors: LogColors = Field(default_factory=LogColors)


class LoggingConfig(BaseModel):
    level: LogLevel = Field(default=LogLevel.INFO)
    file: LogFileConfig
    console: LogConsoleConfig = Field(default_factory=LogConsoleConfig)


class DatabaseConfig(BaseModel):
    filename: str = Field(..., description="Database filename")
    connection_string: str = Field(..., description="SQLAlchemy connection string")
    overwrite: bool = Field(default=False, description="Overwrite existing database")


class CameraResolution(BaseModel):
    width: int = Field(..., ge=320, le=4096, description="Image width in pixels")
    height: int = Field(..., ge=240, le=3072, description="Image height in pixels")


class CameraConfig(BaseModel):
    port: int = Field(..., ge=0, le=3, description="Camera port number")
    name: str = Field(..., min_length=1, description="Camera identifier")
    output_folder: str = Field(
        ..., min_length=1, description="Output directory for images"
    )

    @field_validator("output_folder")
    @classmethod
    def validate_output_folder(cls, v):
        # Validate path format but don't create directory yet
        if not v or not v.strip():
            raise ValueError("Output folder cannot be empty")
        return v.strip()


class CameraParameters(BaseModel):
    cameras: List[CameraConfig] = Field(
        ..., min_length=1, description="List of camera configurations"
    )
    image_format: str = Field(default="jpg", pattern="^(jpg|jpeg|png|bmp)$")
    quality: int = Field(default=95, ge=1, le=100, description="Image quality (1-100)")
    resolution: CameraResolution
    rotation: int = Field(default=0, description="Image rotation in degrees")
    capture_timeout: int = Field(
        default=10, ge=1, le=60, description="Capture timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3, ge=1, le=10, description="Number of retry attempts"
    )

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v):
        if v not in [0, 90, 180, 270]:
            raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")
        return v

    @field_validator("cameras")
    @classmethod
    def validate_unique_ports(cls, v):
        ports = [cam.port for cam in v]
        if len(ports) != len(set(ports)):
            raise ValueError("Camera ports must be unique")
        return v

    @field_validator("cameras")
    @classmethod
    def validate_unique_names(cls, v):
        names = [cam.name for cam in v]
        if len(names) != len(set(names)):
            raise ValueError("Camera names must be unique")
        return v


class BarometerParameters(BaseModel):
    i2c_bus: int = Field(default=1, ge=0, description="I2C bus number")
    address: int = Field(default=0x77, ge=0x00, le=0xFF, description="I2C address")
    sea_level_pressure: float = Field(default=1013.25, gt=0, description="Sea level pressure in hPa")
    
    @field_validator("address")
    @classmethod
    def validate_address_format(cls, v):
        if not (0x76 <= v <= 0x77):
            raise ValueError("BMP390 address must be 0x76 or 0x77")
        return v


class ImuParameters(BaseModel):
    i2c_bus: int = Field(default=1, ge=0, description="I2C bus number")
    address: int = Field(default=0x6A, ge=0x00, le=0xFF, description="I2C address")
    accel_range: str = Field(default="4G", pattern="^(2G|4G|8G|16G)$", description="Accelerometer range")
    gyro_range: str = Field(default="500DPS", pattern="^(125DPS|250DPS|500DPS|1000DPS|2000DPS)$", description="Gyroscope range")
    
    @field_validator("address")
    @classmethod
    def validate_address_format(cls, v):
        if not (0x6A <= v <= 0x6B):
            raise ValueError("LSM6DSOX address must be 0x6A or 0x6B")
        return v


class TaskConfig(BaseModel):
    name: str = Field(..., min_length=1, description="Task name")
    class_name: str = Field(..., alias="class", description="Task class name")
    frequency: int = Field(..., ge=1, description="Task frequency in seconds")
    enabled: bool = Field(default=True, description="Whether task is enabled")
    parameters: Optional[Union[CameraParameters, BarometerParameters, ImuParameters, Dict[str, Any]]] = Field(default=None)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v, info):
        if v is None:
            return v

        # Get class_name from the data being validated
        data = info.data if hasattr(info, "data") else {}
        class_name = data.get("class")

        if class_name == "CameraTask":
            if isinstance(v, dict):
                return CameraParameters(**v)
            return v
        elif class_name == "BaroTask":
            if isinstance(v, dict):
                return BarometerParameters(**v)
            return v
        elif class_name == "ImuTask":
            if isinstance(v, dict):
                return ImuParameters(**v)
            return v
        return v

    @field_validator("name")
    @classmethod
    def validate_task_name(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Task name must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v


class MachaConfig(BaseModel):
    app: AppConfig
    logging: LoggingConfig
    db: DatabaseConfig
    tasks: List[TaskConfig] = Field(
        ..., min_length=1, description="List of task configurations"
    )

    @field_validator("tasks")
    @classmethod
    def validate_unique_task_names(cls, v):
        names = [task.name for task in v]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")
        return v

    @model_validator(mode="after")
    def validate_tasks(self):
        camera_tasks = [t for t in self.tasks if t.class_name == "CameraTask"]
        baro_tasks = [t for t in self.tasks if t.class_name == "BaroTask"]
        imu_tasks = [t for t in self.tasks if t.class_name == "ImuTask"]

        # Validate camera tasks
        for task in camera_tasks:
            if task.parameters is None:
                raise ValueError(f"{task.class_name} '{task.name}' requires parameters")
            if not isinstance(task.parameters, CameraParameters):
                raise ValueError(
                    f"{task.class_name} '{task.name}' has invalid parameters"
                )

            # Only create directories for enabled camera tasks
            if task.enabled:
                for camera in task.parameters.cameras:
                    try:
                        Path(camera.output_folder).mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        raise ValueError(
                            f"Cannot create directory '{camera.output_folder}' for camera '{camera.name}': {e}"
                        )

        # Validate barometer tasks
        for task in baro_tasks:
            if task.parameters is None:
                raise ValueError(f"{task.class_name} '{task.name}' requires parameters")
            if not isinstance(task.parameters, BarometerParameters):
                raise ValueError(
                    f"{task.class_name} '{task.name}' has invalid parameters"
                )

        # Validate IMU tasks
        for task in imu_tasks:
            if task.parameters is None:
                raise ValueError(f"{task.class_name} '{task.name}' requires parameters")
            if not isinstance(task.parameters, ImuParameters):
                raise ValueError(
                    f"{task.class_name} '{task.name}' has invalid parameters"
                )

        return self


def load_config(config_path: str = "config.yaml") -> MachaConfig:
    """Load and validate configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    try:
        config = MachaConfig(**raw_config)
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def get_task_config(config: MachaConfig, task_name: str) -> Optional[TaskConfig]:
    """Get configuration for a specific task."""
    for task in config.tasks:
        if task.name == task_name:
            return task
    return None


def get_camera_tasks(config: MachaConfig) -> List[TaskConfig]:
    """Get all camera task configurations."""
    return [task for task in config.tasks if task.class_name == "CameraTask"]


def get_enabled_tasks(config: MachaConfig) -> List[TaskConfig]:
    """Get all enabled task configurations."""
    return [task for task in config.tasks if task.enabled]
