# Database Commands Reference

This document provides SQL commands for querying data from all macha tasks.

## Database Location

The SQLite database is located at: `macha.db`

## Accessing the Database

```bash
# Open SQLite CLI
sqlite3 macha.db

# Or using Python
python3 -c "import sqlite3; conn = sqlite3.connect('macha.db'); cursor = conn.cursor()"
```

## Table Schemas

### 1. Tasks Table (General Task Execution)
```sql
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Images Table (Camera Task)
```sql
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    camera_name TEXT NOT NULL,
    camera_port INTEGER NOT NULL,
    filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER,
    resolution TEXT,
    format TEXT,
    quality INTEGER,
    metadata TEXT
);
```

### 3. System Metrics Table (Metrics Task)
```sql
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cpu_percent REAL,
    cpu_count INTEGER,
    temperature_c REAL,
    storage_total_gb REAL,
    storage_used_gb REAL,
    storage_free_gb REAL,
    ram_total_gb REAL,
    ram_used_gb REAL,
    ram_free_gb REAL,
    uptime_seconds REAL,
    hostname TEXT,
    system TEXT,
    release TEXT,
    raw_data TEXT
);
```

### 4. Barometer Readings Table (BaroTask)
```sql
CREATE TABLE IF NOT EXISTS barometer_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    pressure_hpa REAL,
    temperature_celsius REAL,
    altitude_meters REAL,
    sea_level_pressure REAL,
    sensor_config TEXT
);
```

### 5. IMU Readings Table (ImuTask)
```sql
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
);
```

## Query Examples

### Camera Task Queries

```sql
-- Get all captured images
SELECT * FROM images ORDER BY timestamp DESC;

-- Get images from specific camera
SELECT * FROM images WHERE camera_name = 'cam0' ORDER BY timestamp DESC;

-- Get recent images (last hour)
SELECT * FROM images 
WHERE timestamp > datetime('now', '-1 hour') 
ORDER BY timestamp DESC;

-- Get image statistics by camera
SELECT 
    camera_name,
    COUNT(*) as total_images,
    AVG(file_size_bytes) as avg_file_size,
    SUM(file_size_bytes) as total_size_bytes
FROM images 
GROUP BY camera_name;

-- Get images by date
SELECT * FROM images 
WHERE date(timestamp) = '2024-01-15' 
ORDER BY timestamp DESC;

-- Get largest images
SELECT camera_name, filename, file_size_bytes, timestamp 
FROM images 
ORDER BY file_size_bytes DESC 
LIMIT 10;
```

### System Metrics Queries

```sql
-- Get latest system metrics
SELECT * FROM system_metrics ORDER BY timestamp DESC LIMIT 1;

-- Get CPU usage over time
SELECT timestamp, cpu_percent, temperature_c 
FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 100;

-- Get memory usage trends
SELECT 
    timestamp,
    ram_used_gb,
    ram_total_gb,
    (ram_used_gb / ram_total_gb * 100) as ram_usage_percent
FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 50;

-- Get storage usage over time
SELECT 
    timestamp,
    storage_used_gb,
    storage_total_gb,
    (storage_used_gb / storage_total_gb * 100) as storage_usage_percent
FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 50;

-- Get temperature statistics
SELECT 
    MIN(temperature_c) as min_temp,
    MAX(temperature_c) as max_temp,
    AVG(temperature_c) as avg_temp,
    COUNT(*) as readings
FROM system_metrics 
WHERE temperature_c IS NOT NULL;

-- Get metrics from last 24 hours
SELECT * FROM system_metrics 
WHERE timestamp > datetime('now', '-24 hours') 
ORDER BY timestamp DESC;
```

### Barometer Task Queries

```sql
-- Get latest barometer readings
SELECT * FROM barometer_readings ORDER BY timestamp DESC LIMIT 10;

-- Get pressure trends
SELECT timestamp, pressure_hpa, altitude_meters 
FROM barometer_readings 
ORDER BY timestamp DESC 
LIMIT 100;

-- Get pressure statistics
SELECT 
    MIN(pressure_hpa) as min_pressure,
    MAX(pressure_hpa) as max_pressure,
    AVG(pressure_hpa) as avg_pressure,
    MIN(altitude_meters) as min_altitude,
    MAX(altitude_meters) as max_altitude,
    AVG(altitude_meters) as avg_altitude
FROM barometer_readings;

-- Get readings from specific time range
SELECT * FROM barometer_readings 
WHERE timestamp BETWEEN '2024-01-15 10:00:00' AND '2024-01-15 12:00:00'
ORDER BY timestamp;

-- Get temperature from barometer
SELECT timestamp, temperature_celsius 
FROM barometer_readings 
ORDER BY timestamp DESC 
LIMIT 50;
```

### IMU Task Queries

```sql
-- Get latest IMU readings
SELECT * FROM imu_readings ORDER BY timestamp DESC LIMIT 10;

-- Get accelerometer data
SELECT 
    timestamp,
    accel_x,
    accel_y,
    accel_z,
    sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z) as magnitude
FROM imu_readings 
ORDER BY timestamp DESC 
LIMIT 100;

-- Get gyroscope data
SELECT timestamp, gyro_x, gyro_y, gyro_z 
FROM imu_readings 
ORDER BY timestamp DESC 
LIMIT 100;

-- Get IMU statistics
SELECT 
    MIN(accel_x) as min_accel_x, MAX(accel_x) as max_accel_x,
    MIN(accel_y) as min_accel_y, MAX(accel_y) as max_accel_y,
    MIN(accel_z) as min_accel_z, MAX(accel_z) as max_accel_z,
    AVG(temperature_celsius) as avg_temp
FROM imu_readings;

-- Get high acceleration events (potential impacts)
SELECT * FROM imu_readings 
WHERE (accel_x*accel_x + accel_y*accel_y + accel_z*accel_z) > 100
ORDER BY timestamp DESC;
```

### General Task Queries

```sql
-- Get all task execution history
SELECT * FROM tasks ORDER BY timestamp DESC;

-- Get task execution counts
SELECT 
    task_name,
    COUNT(*) as execution_count,
    MIN(timestamp) as first_run,
    MAX(timestamp) as last_run
FROM tasks 
GROUP BY task_name;

-- Get recent task executions
SELECT * FROM tasks 
WHERE timestamp > datetime('now', '-1 hour') 
ORDER BY timestamp DESC;
```

### Cross-Task Analysis

```sql
-- Get data from all sensors at similar times
SELECT 
    m.timestamp as metrics_time,
    m.cpu_percent,
    m.temperature_c as system_temp,
    b.pressure_hpa,
    b.temperature_celsius as baro_temp,
    b.altitude_meters,
    i.accel_x, i.accel_y, i.accel_z
FROM system_metrics m
LEFT JOIN barometer_readings b ON 
    abs(julianday(m.timestamp) - julianday(b.timestamp)) < 0.0007  -- Within ~1 minute
LEFT JOIN imu_readings i ON 
    abs(julianday(m.timestamp) - julianday(i.timestamp)) < 0.0007  -- Within ~1 minute
ORDER BY m.timestamp DESC
LIMIT 50;

-- Data collection summary
SELECT 
    'images' as table_name,
    COUNT(*) as record_count,
    MIN(timestamp) as earliest,
    MAX(timestamp) as latest
FROM images
UNION ALL
SELECT 
    'system_metrics',
    COUNT(*),
    MIN(timestamp),
    MAX(timestamp)
FROM system_metrics
UNION ALL
SELECT 
    'barometer_readings',
    COUNT(*),
    MIN(timestamp),
    MAX(timestamp)
FROM barometer_readings
UNION ALL
SELECT 
    'imu_readings',
    COUNT(*),
    MIN(timestamp),
    MAX(timestamp)
FROM imu_readings;
```

## Data Export Commands

### Export to CSV

```sql
-- Export images data
.mode csv
.headers on
.output images_export.csv
SELECT * FROM images;
.output stdout

-- Export system metrics
.output system_metrics_export.csv
SELECT * FROM system_metrics;
.output stdout

-- Export sensor data
.output barometer_export.csv
SELECT * FROM barometer_readings;
.output stdout

.output imu_export.csv
SELECT * FROM imu_readings;
.output stdout
```

### Backup Database

```bash
# Create backup
sqlite3 macha.db ".backup backup_$(date +%Y%m%d_%H%M%S).db"

# Or use SQL dump
sqlite3 macha.db ".dump" > backup_$(date +%Y%m%d_%H%M%S).sql
```

## Database Maintenance

### Cleanup Old Data

```sql
-- Remove images older than 30 days
DELETE FROM images WHERE timestamp < datetime('now', '-30 days');

-- Remove metrics older than 7 days
DELETE FROM system_metrics WHERE timestamp < datetime('now', '-7 days');

-- Remove sensor readings older than 7 days
DELETE FROM barometer_readings WHERE timestamp < datetime('now', '-7 days');
DELETE FROM imu_readings WHERE timestamp < datetime('now', '-7 days');

-- Vacuum database to reclaim space
VACUUM;
```

### Database Statistics

```sql
-- Get table sizes
SELECT 
    name,
    (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as table_exists,
    (SELECT COUNT(*) FROM pragma_table_info(m.name)) as column_count
FROM sqlite_master m WHERE type='table';

-- Get database size info
PRAGMA page_count;
PRAGMA page_size;
PRAGMA freelist_count;
```

## Programmatic Access

### Python Example

```python
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('macha.db')

# Get recent sensor data
query = """
SELECT * FROM barometer_readings 
WHERE timestamp > ? 
ORDER BY timestamp DESC
"""
df = pd.read_sql_query(query, conn, params=[datetime.now() - timedelta(hours=1)])

# Close connection
conn.close()
```

### Bash Script Example

```bash
#!/bin/bash
# Get latest readings from all sensors

echo "=== Latest System Metrics ==="
sqlite3 macha.db "SELECT timestamp, cpu_percent, temperature_c FROM system_metrics ORDER BY timestamp DESC LIMIT 1;"

echo "=== Latest Barometer Reading ==="
sqlite3 macha.db "SELECT timestamp, pressure_hpa, altitude_meters FROM barometer_readings ORDER BY timestamp DESC LIMIT 1;"

echo "=== Latest IMU Reading ==="
sqlite3 macha.db "SELECT timestamp, accel_x, accel_y, accel_z FROM imu_readings ORDER BY timestamp DESC LIMIT 1;"

echo "=== Latest Image ==="
sqlite3 macha.db "SELECT timestamp, camera_name, filename FROM images ORDER BY timestamp DESC LIMIT 1;"
```
