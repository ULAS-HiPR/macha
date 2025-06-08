#!/bin/bash

# Database Status Check Script for Macha
# This script provides a quick overview of all data in the macha database

DB_FILE="macha.db"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_DIR/$DB_FILE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Macha Database Status Check ===${NC}"
echo "Database: $DB_PATH"
echo

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}ERROR: Database file not found at $DB_PATH${NC}"
    exit 1
fi

# Check database size
DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
echo -e "${GREEN}Database size: $DB_SIZE${NC}"
echo

# Function to run SQL query and handle errors
run_query() {
    local query="$1"
    local description="$2"
    
    echo -e "${YELLOW}$description${NC}"
    if ! sqlite3 "$DB_PATH" "$query" 2>/dev/null; then
        echo -e "${RED}  Error running query${NC}"
    fi
    echo
}

# Check table existence and record counts
echo -e "${BLUE}=== Table Overview ===${NC}"
run_query "
SELECT 
    name as table_name,
    (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as exists,
    CASE 
        WHEN name = 'images' THEN (SELECT COUNT(*) FROM images)
        WHEN name = 'system_metrics' THEN (SELECT COUNT(*) FROM system_metrics)
        WHEN name = 'barometer_readings' THEN (SELECT COUNT(*) FROM barometer_readings)
        WHEN name = 'imu_readings' THEN (SELECT COUNT(*) FROM imu_readings)
        WHEN name = 'tasks' THEN (SELECT COUNT(*) FROM tasks)
        ELSE 'N/A'
    END as record_count
FROM sqlite_master 
WHERE type='table' AND name NOT LIKE 'sqlite_%'
ORDER BY name;
" "Tables and Record Counts:"

# Latest data from each task
echo -e "${BLUE}=== Latest Data Summary ===${NC}"

# Latest image
run_query "
SELECT 'Latest Image:' as info, camera_name, filename, datetime(timestamp, 'localtime') as local_time
FROM images 
ORDER BY timestamp DESC 
LIMIT 1;
" "Camera Task - Latest Image:"

# Latest system metrics
run_query "
SELECT 'Latest Metrics:' as info, 
       printf('%.1f%%', cpu_percent) as cpu,
       printf('%.1f°C', temperature_c) as temp,
       printf('%.1fGB', ram_used_gb) as ram_used,
       datetime(timestamp, 'localtime') as local_time
FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 1;
" "Metrics Task - Latest Reading:"

# Latest barometer reading
run_query "
SELECT 'Latest Baro:' as info,
       printf('%.2f hPa', pressure_hpa) as pressure,
       printf('%.1f°C', temperature_celsius) as temp,
       printf('%.1fm', altitude_meters) as altitude,
       datetime(timestamp, 'localtime') as local_time
FROM barometer_readings 
ORDER BY timestamp DESC 
LIMIT 1;
" "Barometer Task - Latest Reading:"

# Latest IMU reading
run_query "
SELECT 'Latest IMU:' as info,
       printf('A:(%.2f,%.2f,%.2f)', accel_x, accel_y, accel_z) as accel,
       printf('G:(%.2f,%.2f,%.2f)', gyro_x, gyro_y, gyro_z) as gyro,
       datetime(timestamp, 'localtime') as local_time
FROM imu_readings 
ORDER BY timestamp DESC 
LIMIT 1;
" "IMU Task - Latest Reading:"

# Data collection rates (last hour)
echo -e "${BLUE}=== Data Collection Rates (Last Hour) ===${NC}"

run_query "
SELECT 
    'Images' as data_type,
    COUNT(*) as count_last_hour,
    printf('%.1f/min', COUNT(*) / 60.0) as rate_per_min
FROM images 
WHERE timestamp > datetime('now', '-1 hour')
UNION ALL
SELECT 
    'System Metrics',
    COUNT(*),
    printf('%.1f/min', COUNT(*) / 60.0)
FROM system_metrics 
WHERE timestamp > datetime('now', '-1 hour')
UNION ALL
SELECT 
    'Barometer',
    COUNT(*),
    printf('%.1f/min', COUNT(*) / 60.0)
FROM barometer_readings 
WHERE timestamp > datetime('now', '-1 hour')
UNION ALL
SELECT 
    'IMU',
    COUNT(*),
    printf('%.1f/min', COUNT(*) / 60.0)
FROM imu_readings 
WHERE timestamp > datetime('now', '-1 hour');
" "Collection rates in the last hour:"

# Storage usage by images
echo -e "${BLUE}=== Storage Analysis ===${NC}"

run_query "
SELECT 
    camera_name,
    COUNT(*) as image_count,
    printf('%.1f MB', SUM(file_size_bytes) / 1024.0 / 1024.0) as total_size,
    printf('%.1f KB', AVG(file_size_bytes) / 1024.0) as avg_size
FROM images 
GROUP BY camera_name
ORDER BY camera_name;
" "Image storage by camera:"

# Task execution summary
echo -e "${BLUE}=== Task Execution Summary ===${NC}"

run_query "
SELECT 
    task_name,
    COUNT(*) as executions,
    datetime(MIN(timestamp), 'localtime') as first_run,
    datetime(MAX(timestamp), 'localtime') as last_run
FROM tasks 
GROUP BY task_name
ORDER BY task_name;
" "Task execution history:"

# System health indicators
echo -e "${BLUE}=== System Health Indicators ===${NC}"

run_query "
SELECT 
    'Current' as timeframe,
    printf('%.1f%%', cpu_percent) as cpu_usage,
    printf('%.1f°C', temperature_c) as temperature,
    printf('%.1f%%', (ram_used_gb / ram_total_gb * 100)) as ram_usage,
    printf('%.1f%%', (storage_used_gb / storage_total_gb * 100)) as storage_usage
FROM system_metrics 
ORDER BY timestamp DESC 
LIMIT 1;
" "Current system status:"

run_query "
SELECT 
    'Last Hour Avg' as timeframe,
    printf('%.1f%%', AVG(cpu_percent)) as cpu_usage,
    printf('%.1f°C', AVG(temperature_c)) as temperature,
    printf('%.1f%%', AVG(ram_used_gb / ram_total_gb * 100)) as ram_usage,
    printf('%.1f%%', AVG(storage_used_gb / storage_total_gb * 100)) as storage_usage
FROM system_metrics 
WHERE timestamp > datetime('now', '-1 hour');
" "Average over last hour:"

# Data freshness check
echo -e "${BLUE}=== Data Freshness Check ===${NC}"

run_query "
SELECT 
    'Images' as data_type,
    CASE 
        WHEN MAX(timestamp) > datetime('now', '-10 minutes') THEN 'FRESH'
        WHEN MAX(timestamp) > datetime('now', '-1 hour') THEN 'STALE'
        ELSE 'OLD'
    END as status,
    printf('%.1f min ago', (julianday('now') - julianday(MAX(timestamp))) * 24 * 60) as last_update
FROM images
UNION ALL
SELECT 
    'System Metrics',
    CASE 
        WHEN MAX(timestamp) > datetime('now', '-5 minutes') THEN 'FRESH'
        WHEN MAX(timestamp) > datetime('now', '-1 hour') THEN 'STALE'
        ELSE 'OLD'
    END,
    printf('%.1f min ago', (julianday('now') - julianday(MAX(timestamp))) * 24 * 60)
FROM system_metrics
UNION ALL
SELECT 
    'Barometer',
    CASE 
        WHEN MAX(timestamp) > datetime('now', '-5 minutes') THEN 'FRESH'
        WHEN MAX(timestamp) > datetime('now', '-1 hour') THEN 'STALE'
        ELSE 'OLD'
    END,
    printf('%.1f min ago', (julianday('now') - julianday(MAX(timestamp))) * 24 * 60)
FROM barometer_readings
UNION ALL
SELECT 
    'IMU',
    CASE 
        WHEN MAX(timestamp) > datetime('now', '-5 minutes') THEN 'FRESH'
        WHEN MAX(timestamp) > datetime('now', '-1 hour') THEN 'STALE'
        ELSE 'OLD'
    END,
    printf('%.1f min ago', (julianday('now') - julianday(MAX(timestamp))) * 24 * 60)
FROM imu_readings;
" "Data freshness status:"

echo -e "${GREEN}=== Database check complete ===${NC}"
echo
echo "For detailed queries, see DATABASE_COMMANDS.md"
echo "To access database directly: sqlite3 $DB_PATH"