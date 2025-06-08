# Macha - Camera Monitoring System

A continuous camera monitoring system for Raspberry Pi 5 with dual camera support, built with Python and Pydantic for robust configuration management.

## Features

- **Dual Camera Support**: Capture images from both camera ports simultaneously
- **Configurable Scheduling**: Run tasks at configurable intervals with Pydantic validation
- **Database Logging**: All captures and metrics logged to SQLite database
- **System Metrics**: CPU, temperature, memory, and storage monitoring
- **Graceful Shutdown**: Proper cleanup on termination signals
- **Schema Validation**: Pydantic-based configuration with comprehensive validation
- **Systemd Service**: Production-ready deployment with automatic startup

## Requirements

- Raspberry Pi 5 with dual camera setup
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

**Important**: Reboot after setup for all changes to take effect:
```bash
sudo reboot
```

### 2. Test System

After reboot, verify everything is working:

```bash
./scripts/test_camera.py
```

This comprehensive test checks:
- System information
- Camera hardware detection
- Camera device permissions
- Boot configuration
- User permissions
- Python camera interface
- Macha configuration validation

### 3. Run Application

Start the monitoring system:

```bash
./scripts/run.sh
```

Or manually:
```bash
source .venv/bin/activate
python src/main.py
```

### 4. Deploy as Service (Optional)

For production deployment with automatic startup:

```bash
sudo ./scripts/deploy.sh
```

## Scripts Directory

The `scripts/` directory contains all setup and management tools:

### Setup Scripts
- **`setup.sh`** - Complete system setup (run this first)
- **`deploy.sh`** - Deploy systemd service for production

### Utility Scripts  
- **`run.sh`** - Simple application launcher
- **`test_camera.py`** - Comprehensive system testing

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

### Starting the Application

```bash
# Using the run script (recommended)
./scripts/run.sh

# Or manually
source .venv/bin/activate
python src/main.py
```

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

# Test system
./scripts/test_camera.py
```

## Troubleshooting

### Camera Not Working

1. **Run the test script**: `./scripts/test_camera.py`
2. **Check hardware**: `libcamera-hello --list-cameras`  
3. **Verify permissions**: User must be in `video` and `gpio` groups
4. **Camera enabled**: Check `/boot/firmware/config.txt` has `camera_auto_detect=1`
5. **Reboot required**: Many camera changes require a system reboot

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
- **imx219** (Pi Camera v2)
- Supports up to 4 camera ports (Pi 5)

### Supported Resolutions

- 640x480 (VGA)
- 1296x972 (1.3MP)  
- 1920x1080 (FHD)
- 2592x1944 (5MP, ov5647)
- 3280x2464 (8MP, imx219)

### Image Formats

- JPEG (recommended)
- PNG
- BMP

## License

This project is provided as-is for educational and development purposes.