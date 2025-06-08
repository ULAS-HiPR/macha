# Macha

CanSat software to collect flight data and pictures to run in-flight on a Coral TPU to segment safe vs unsafe landings.

## Features

- **Configurable Scheduling**: Run tasks at configurable intervals with Pydantic validation
- **System Metrics**: CPU, temperature, memory, and storage monitoring
- **Sensor Data Collection**: Barometer (BMP390) and IMU (LSM6DSOX) data logging
- **Schema Validation**: Pydantic-based configuration with comprehensive validation
- **Systemd Service**: Production-ready deployment with automatic startup

## Requirements

- Raspberry Pi 5
- Python 3.11+
- System camera packages (python3-picamera2, libcamera-apps)
- SQLite database support

## Quick Start

### 1. Initial Setup

Run the comprehensive setup script:

```bash
./scripts/setup.sh
```

This script will:
- Install all system dependencies (camera packages, libraries)
- Configure camera permissions and boot settings
- Create virtual environment with system site-packages access
- Set up user groups (video, gpio)
- Test camera functionality
- Validate configuration

### 3. Deploy Application

Start the monitoring system (registers service and enables it to start on boot):

```bash
./scripts/deploy.sh # uv run src/main.py in dev btw

# then you should reboot to make sure everything in setup.sh is linked properly
sudo reboot
```


## Scripts Directory

The `scripts/` directory contains all setup and management tools:

### Setup Scripts
- **`setup.sh`** - Complete system setup (run this first)
- **`deploy.sh`** - Deploy systemd service for production


## Configuration

Edit `config.yaml` to customize behavior. The configuration uses Pydantic for validation:

### Camera Configuration Example

```yaml
tasks:
  - name: camera_capture
    class: CameraTask
    frequency: 30  # seconds
    enabled: true
    parameters:
      cameras:
        - port: 0
          name: cam0
          output_folder: images/cam0
        - port: 1
          name: cam1
          output_folder: images/cam1
      image_format: jpg
      quality: 95
      resolution:
        width: 1920
        height: 1080
      rotation: 0
      capture_timeout: 10
      retry_attempts: 3
```

### Metrics Configuration

```yaml
tasks:
  - name: metrics_collection
    class: MetricsTask
    frequency: 60  # seconds
    enabled: true
```

### Sensor Configuration

```yaml
tasks:
  - name: barometer_reading
    class: BaroTask
    frequency: 30  # seconds
    enabled: true
    parameters:
      i2c_bus: 1
      address: 0x77  # or 0x76
      sea_level_pressure: 1013.25

  - name: imu_reading
    class: ImuTask
    frequency: 10  # seconds
    enabled: true
    parameters:
      i2c_bus: 1
      address: 0x6A  # or 0x6B
      accel_range: "4G"  # 2G, 4G, 8G, 16G
      gyro_range: "500DPS"  # 125DPS, 250DPS, 500DPS, 1000DPS, 2000DPS
```

### Configuration Validation

All configuration is validated using Pydantic schemas:
- Required fields are enforced
- Data types are validated
- Camera ports must be unique
- Camera names must be unique
- Output directories are created automatically
- Invalid configurations are rejected with clear error messages

## Database Schema

### Images Table
- `id`: Primary key
- `camera_name`: Camera identifier (cam0, cam1)
- `camera_port`: Hardware port number (0, 1)
- `filepath`: Full path to captured image
- `filename`: Image filename with timestamp
- `timestamp`: Capture timestamp (ISO format)
- `file_size_bytes`: File size in bytes
- `resolution`: Image resolution (e.g., "1920x1080")
- `format`: Image format (jpg, png)
- `quality`: Compression quality (1-100)
- `metadata`: JSON metadata including capture parameters

### Tasks Table
- `id`: Primary key
- `task_name`: Name of executed task
- `result`: Task execution result (JSON)
- `timestamp`: Execution timestamp

### Barometer Readings Table
- `id`: Primary key
- `timestamp`: Reading timestamp
- `pressure_hpa`: Atmospheric pressure in hectopascals
- `temperature_celsius`: Temperature in Celsius
- `altitude_meters`: Calculated altitude in meters
- `sea_level_pressure`: Reference sea level pressure
- `sensor_config`: JSON configuration used

### IMU Readings Table
- `id`: Primary key
- `timestamp`: Reading timestamp
- `accel_x`, `accel_y`, `accel_z`: Acceleration in m/s²
- `gyro_x`, `gyro_y`, `gyro_z`: Angular velocity in rad/s
- `temperature_celsius`: Sensor temperature in Celsius
- `sensor_config`: JSON configuration used

## File Structure

```
macha/
├── config.yaml          # Main configuration with validation
├── macha.service        # Systemd service file
├── src/
│   ├── main.py          # Main application entry point
│   ├── config.py        # Pydantic configuration schemas
│   ├── scheduler.py     # Task scheduler
│   ├── camera_task.py   # Camera capture implementation
│   ├── metrics_task.py  # System metrics collection
│   ├── baro_task.py     # Barometer sensor task (BMP390)
│   ├── imu_task.py      # IMU sensor task (LSM6DSOX)
│   ├── task.py          # Base task classes
│   ├── logger.py        # Logging setup
│   └── database.py      # Database initialization
├── scripts/
│   ├── setup.sh         # Complete system setup
│   ├── deploy.sh        # Service deployment
│   ├── run.sh           # Application launcher
│   └── test_camera.py   # System testing
├── images/              # Captured images
│   ├── cam0/           # Camera 0 images
│   └── cam1/           # Camera 1 images
└── logs/               # Application logs
```

## Usage

### Viewing Logs

```bash
# Application logs
tail -f logs/macha.log

# Service logs (if deployed)
journalctl -u macha -f
```

### Checking Status

```bash
# Service status
systemctl status macha
```

## Troubleshooting

### Camera Not Working

1. **Check hardware**: `libcamera-hello --list-cameras`
2. **Verify permissions**: User must be in `video` and `gpio` groups
3. **Camera enabled**: Check `/boot/firmware/config.txt` has `camera_auto_detect=1`
4. **Reboot required**: Many camera changes require a system reboot

### Permission Errors

```bash
# Add user to required groups
sudo usermod -a -G video,gpio $USER

# Log out and back in, or reboot
```

### Virtual Environment Issues

If camera imports fail, recreate the virtual environment:

```bash
rm -rf .venv
./scripts/setup.sh
```

### Configuration Errors

Configuration validation will show specific errors:
- Check required fields are present
- Verify data types match schema
- Ensure camera ports/names are unique
- Check file paths are valid

### Service Issues

```bash
# Check service status
systemctl status macha

# View service logs
journalctl -u macha -f

# Restart service
sudo systemctl restart macha
```

## Development

### Adding New Tasks

1. Create task class inheriting from `Task`
2. Add Pydantic schema for parameters in `config.py`
3. Register task class in `scheduler.py`
4. Add configuration to `config.yaml`

### Configuration Schema

All task parameters must be defined as Pydantic models in `src/config.py`. This ensures:
- Type safety
- Validation at startup
- Clear error messages
- Auto-generated documentation

## Logging

Logs are written to `logs/macha.log` and console. Configure log levels in `config.yaml`:

- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical errors

## Service Management

### Manual Service Commands

```bash
# Start service
sudo systemctl start macha

# Stop service
sudo systemctl stop macha

# Restart service
sudo systemctl restart macha

# Enable auto-start
sudo systemctl enable macha

# Disable auto-start
sudo systemctl disable macha

# View status
systemctl status macha

# View logs
journalctl -u macha -f
```

### Service Configuration

The service runs as user `payload` and includes:
- Automatic restart on failure
- Security hardening
- Access to camera devices
- Proper working directory
- Environment isolation

## Hardware Support

### Tested Cameras

- **ov5647** (Original Pi Camera)
- **imx219** (Pi Camera v2) - did not work

### Supported Sensors

- **BMP390** - Barometer (pressure, temperature, altitude)
  - I2C addresses: 0x77 (default), 0x76
  - Measures atmospheric pressure, temperature
  - Calculates altitude based on sea level pressure reference
  
- **LSM6DSOX** - 6-axis IMU (accelerometer + gyroscope)
  - I2C addresses: 0x6A (default), 0x6B
  - 3-axis accelerometer: ±2g, ±4g, ±8g, ±16g ranges
  - 3-axis gyroscope: ±125, ±250, ±500, ±1000, ±2000 dps ranges
  - Built-in temperature sensor

## Development

### Testing

Run tests with pytest:

```bash
# Install test dependencies
uv sync --all-extras

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_sensor_tasks.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Linting and Formatting

The project uses ruff for linting and formatting:

```bash
# Check code style
uv run ruff check .

# Format code
uv run ruff format .

# Check formatting without applying
uv run ruff format --check .
```

### GitHub Actions

The project includes CI/CD workflows:

- **PR Workflow** (`.github/workflows/pr.yml`): Runs tests and linting on pull requests
- **Release Workflow** (`.github/workflows/release.yml`): Handles version bumping and changelog generation using Conventional Commits

#### Conventional Commits

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for semantic versioning:

```bash
feat: add new sensor support
fix: resolve I2C communication issue
docs: update sensor configuration examples
test: add sensor task unit tests
```

### Sensor Development

When adding new sensor tasks:

1. Create sensor task class inheriting from `Task`
2. Add parameter schema in `config.py`
3. Update task validation in `MachaConfig.validate_tasks()`
4. Add configuration example to `config.yaml`
5. Create unit tests in `tests/`
6. Update documentation

Example sensor task structure:

```python
from task import Task
from config import MachaConfig

class MySensorTask(Task):
    def __init__(self, config: MachaConfig):
        super().__init__(config)
        self.sensor = None
        self.parameters = None  # Extract from config
    
    async def _initialize_sensor(self, logger):
        # Initialize sensor hardware
        pass
    
    async def execute(self, engine, logger):
        # Read sensor, store in database
        pass
```
