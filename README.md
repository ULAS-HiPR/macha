# Macha

CanSat software to collect flight data and pictures to run in-flight on a Coral TPU to segment safe vs unsafe landings.

## Features

- **Configurable Scheduling**: Run tasks at configurable intervals with Pydantic validation
- **System Metrics**: CPU, temperature, memory, and storage monitoring
- **Sensor Data Collection**: Barometer (BMP390) and IMU (LSM6DSOX) data logging
- **Schema Validation**: Pydantic-based configuration with comprehensive validation
- **Systemd Service**: Production-ready deployment with automatic startup

## Requirements

- Raspberry Pi 5 (for hardware sensors and camera)
- Python 3.11+
- uv package manager
- SQLite database support

### Hardware Dependencies (Pi only)
- I2C sensors: BMP390 (barometer), LSM6DSOX (IMU)
- Camera module
- System packages: libcap-dev, i2c-tools, libcamera-apps

## Quick Start

### 1. Development Setup (Any System)

For development without Pi hardware:

```bash
# Install core dependencies
uv sync

# Run with mock hardware
uv run src/main.py
```

### 2. Raspberry Pi Setup

For full hardware support on Raspberry Pi:

```bash
# Install Pi hardware dependencies
./scripts/install_pi_deps.sh

# Reboot if prompted
sudo reboot

# Test hardware
python3 scripts/test_hardware.py

# Run with full hardware support
uv sync --extra pi-hardware
uv run src/main.py
```

### 3. Deploy Application

Start the monitoring system (registers service and enables it to start on boot):

```bash
./scripts/deploy.sh

# Reboot to ensure all hardware is properly configured
sudo reboot
```


## Scripts Directory

The `scripts/` directory contains all setup and management tools:

### Setup Scripts
- **`install_pi_deps.sh`** - Install Raspberry Pi hardware dependencies
- **`setup.sh`** - Complete system setup (legacy)
- **`deploy.sh`** - Deploy systemd service for production
- **`test_hardware.py`** - Test all sensors and camera
- **`check_db.sh`** - Database status and health check


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

**Note**: Sensor tasks will gracefully handle missing hardware and log warnings if sensors are not available.

### AI Segmentation Configuration

```yaml
tasks:
  - name: ai_segmentation
    class: AiTask
    frequency: 20  # seconds (must be >= camera frequency and divisible)
    enabled: true
    parameters:
      model_path: models/landing_segmentation_edgetpu.tflite
      model_name: landing_segmentation_v1
      model_version: "1.0.0"
      use_coral_tpu: true
      output_folder: segmentation_outputs
      confidence_threshold: 0.6
      max_queue_size: 30
      processing_timeout: 8
      max_retries: 2
      output_format: png
      save_confidence_overlay: true
      class_names: ["background", "safe_landing", "unsafe_landing"]
      class_colors:
        background: [0, 0, 0]
        safe_landing: [0, 255, 0]
        unsafe_landing: [255, 0, 0]
```

**Note**: AI task processes captured images to generate segmentation masks for landing safety analysis. Requires TensorFlow Lite models optimized for Coral TPU.

### Configuration Validation

All configuration is validated using Pydantic schemas:
- Required fields are enforced
- Data types are validated
- Camera ports must be unique
- Camera names must be unique
- Output directories are created automatically
- Invalid configurations are rejected with clear error messages

## Database Schema

The database uses a centralized schema management system with automatic migrations. All tables are created and updated automatically when the application starts.

### Images Table (Camera Task)
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

### System Metrics Table (Metrics Task)
- `id`: Primary key
- `timestamp`: Reading timestamp
- `cpu_percent`: CPU usage percentage
- `cpu_count`: Number of CPU cores
- `temperature_c`: System temperature in Celsius
- `storage_total_gb`, `storage_used_gb`, `storage_free_gb`: Storage statistics in GB
- `ram_total_gb`, `ram_used_gb`, `ram_free_gb`: RAM statistics in GB
- `uptime_seconds`: System uptime in seconds
- `hostname`, `system`, `release`: System information
- `raw_data`: Complete metrics as JSON

### Tasks Table (General Task Execution)
- `id`: Primary key
- `task_name`: Name of executed task
- `result`: Task execution result (JSON)
- `timestamp`: Execution timestamp

### Barometer Readings Table (BaroTask)
- `id`: Primary key
- `timestamp`: Reading timestamp
- `pressure_hpa`: Atmospheric pressure in hectopascals
- `temperature_celsius`: Temperature in Celsius
- `altitude_meters`: Calculated altitude in meters
- `sea_level_pressure`: Reference sea level pressure
- `sensor_config`: JSON configuration used

### IMU Readings Table (ImuTask)
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
├── macha.db             # SQLite database with all task data
├── macha.service        # Systemd service file
├── DATABASE_COMMANDS.md # SQL queries and database reference
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
├── tests/               # Test suite
│   ├── test_imports.py  # Import validation tests
│   └── test_sensor_tasks.py # Sensor task tests
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

### Database Access

All task data is stored in `macha.db` with automatic schema migrations. Access the database using:

```bash
# SQLite CLI
sqlite3 macha.db

# Quick status check
./scripts/check_db.sh

# Quick data checks
sqlite3 macha.db "SELECT COUNT(*) FROM images;"
sqlite3 macha.db "SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 1;"
```

See `DATABASE_COMMANDS.md` for comprehensive SQL examples and queries for all tasks.

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

### Dependencies

The project uses conditional dependencies for different environments:

```bash
# Core dependencies only (CI/development)
uv sync

# With development tools
uv sync --extra dev

# With Pi hardware support
uv sync --extra pi-hardware

# Everything (Pi only)
uv sync --extra all
```

### Testing

Run tests with pytest:

```bash
# Install test dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_sensor_tasks.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Test hardware (Pi only)
python3 scripts/test_hardware.py
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

The project includes CI/CD workflows that work without Pi hardware:

- **PR Workflow** (`.github/workflows/pr.yml`): Runs tests and linting on pull requests using core dependencies only
- **Release Workflow** (`.github/workflows/release.yml`): Handles version bumping and changelog generation using Conventional Commits

**Note**: CI workflows skip Pi-specific dependencies to avoid build issues on GitHub runners.

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
