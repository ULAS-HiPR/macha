#!/usr/bin/env python3
"""
Database Summary Tool for Macha Project

This script provides comprehensive summaries and statistics for all data
recorded in the Macha database. It analyzes system metrics, sensor readings,
images, and task execution history.

Usage:
    python scripts/db_summary.py [OPTIONS]

Options:
    --config PATH       Path to config file (default: config.yaml)
    --db PATH          Path to database file (overrides config)
    --table TABLE      Show detailed summary for specific table
    --since DURATION   Show data from last N hours/days (e.g., "24h", "7d")
    --export FORMAT    Export summary to file (json, csv, html)
    --verbose          Show detailed statistics
    --no-color         Disable colored output
    --help             Show this help message
"""

import sys
import argparse
import asyncio
import json
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import textwrap

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import aiosqlite
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    from config import load_config, MachaConfig
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please ensure the project is properly installed.")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class DatabaseSummary:
    """Database summary and analysis tool."""

    def __init__(self, db_path: str, use_colors: bool = True):
        self.db_path = db_path
        self.use_colors = use_colors
        self.colors = Colors() if use_colors else type('Colors', (), {k: '' for k in dir(Colors) if not k.startswith('_')})()

    def print_header(self, text: str, level: int = 1) -> None:
        """Print formatted header."""
        if level == 1:
            print(f"\n{self.colors.BOLD}{self.colors.BLUE}{'=' * 60}{self.colors.RESET}")
            print(f"{self.colors.BOLD}{self.colors.BLUE}{text.center(60)}{self.colors.RESET}")
            print(f"{self.colors.BOLD}{self.colors.BLUE}{'=' * 60}{self.colors.RESET}")
        elif level == 2:
            print(f"\n{self.colors.BOLD}{self.colors.CYAN}{'-' * 40}{self.colors.RESET}")
            print(f"{self.colors.BOLD}{self.colors.CYAN}{text}{self.colors.RESET}")
            print(f"{self.colors.BOLD}{self.colors.CYAN}{'-' * 40}{self.colors.RESET}")
        else:
            print(f"\n{self.colors.BOLD}{text}{self.colors.RESET}")

    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def format_duration(self, seconds: float) -> str:
        """Format duration in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"

    def parse_time_filter(self, time_str: str) -> datetime:
        """Parse time filter string (e.g., '24h', '7d') to datetime."""
        if not time_str:
            return None

        try:
            if time_str.endswith('h'):
                hours = int(time_str[:-1])
                return datetime.now() - timedelta(hours=hours)
            elif time_str.endswith('d'):
                days = int(time_str[:-1])
                return datetime.now() - timedelta(days=days)
            elif time_str.endswith('m'):
                minutes = int(time_str[:-1])
                return datetime.now() - timedelta(minutes=minutes)
            else:
                # Try to parse as ISO format
                return datetime.fromisoformat(time_str)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid time format: {time_str}. Use format like '24h', '7d', or ISO datetime.")

    async def get_database_info(self) -> Dict[str, Any]:
        """Get basic database information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get database file size
                db_size = Path(self.db_path).stat().st_size

                # Get table information
                async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] for row in await cursor.fetchall()]

                # Get total record count
                total_records = 0
                table_info = {}

                for table in tables:
                    try:
                        async with db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                            count = (await cursor.fetchone())[0]
                            total_records += count
                            table_info[table] = {'count': count}

                        # Get table schema
                        async with db.execute(f"PRAGMA table_info({table})") as cursor:
                            schema = await cursor.fetchall()
                            table_info[table]['columns'] = len(schema)
                            table_info[table]['schema'] = schema
                    except Exception as e:
                        table_info[table] = {'error': str(e)}

                return {
                    'file_path': self.db_path,
                    'file_size': db_size,
                    'tables': tables,
                    'table_info': table_info,
                    'total_records': total_records
                }
        except Exception as e:
            raise RuntimeError(f"Failed to analyze database: {e}")

    async def get_time_range_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of data within time range."""
        time_filter = ""
        params = {}

        if since:
            time_filter = "WHERE timestamp >= :since"
            params['since'] = since.isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            summary = {}

            # System metrics summary
            query = f"""
                SELECT
                    COUNT(*) as count,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record,
                    AVG(cpu_percent) as avg_cpu,
                    MAX(cpu_percent) as max_cpu,
                    AVG(temperature_c) as avg_temp,
                    MAX(temperature_c) as max_temp,
                    AVG(ram_used_gb) as avg_ram_used,
                    MAX(ram_used_gb) as max_ram_used
                FROM system_metrics {time_filter}
            """

            try:
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['system_metrics'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'avg_cpu_percent': round(row[3], 1) if row[3] else 0,
                            'max_cpu_percent': round(row[4], 1) if row[4] else 0,
                            'avg_temperature_c': round(row[5], 1) if row[5] else 0,
                            'max_temperature_c': round(row[6], 1) if row[6] else 0,
                            'avg_ram_used_gb': round(row[7], 2) if row[7] else 0,
                            'max_ram_used_gb': round(row[8], 2) if row[8] else 0,
                        }
            except Exception:
                summary['system_metrics'] = {'count': 0}

            # Sensor readings summary
            for sensor_table in ['barometer_readings', 'imu_readings']:
                try:
                    query = f"""
                        SELECT
                            COUNT(*) as count,
                            MIN(timestamp) as first_record,
                            MAX(timestamp) as last_record
                        FROM {sensor_table} {time_filter}
                    """

                    async with db.execute(query, params) as cursor:
                        row = await cursor.fetchone()
                        if row and row[0] > 0:
                            summary[sensor_table] = {
                                'count': row[0],
                                'time_range': f"{row[1]} to {row[2]}"
                            }

                            # Get additional sensor-specific stats
                            if sensor_table == 'barometer_readings':
                                query2 = f"""
                                    SELECT
                                        AVG(pressure_hpa) as avg_pressure,
                                        MIN(pressure_hpa) as min_pressure,
                                        MAX(pressure_hpa) as max_pressure,
                                        AVG(temperature_celsius) as avg_temp,
                                        AVG(altitude_meters) as avg_altitude
                                    FROM {sensor_table} {time_filter}
                                """
                                async with db.execute(query2, params) as cursor2:
                                    row2 = await cursor2.fetchone()
                                    if row2:
                                        summary[sensor_table].update({
                                            'avg_pressure_hpa': round(row2[0], 1) if row2[0] else 0,
                                            'pressure_range_hpa': f"{round(row2[1], 1) if row2[1] else 0} - {round(row2[2], 1) if row2[2] else 0}",
                                            'avg_temperature_c': round(row2[3], 1) if row2[3] else 0,
                                            'avg_altitude_m': round(row2[4], 1) if row2[4] else 0,
                                        })

                            elif sensor_table == 'imu_readings':
                                query2 = f"""
                                    SELECT
                                        AVG(ABS(accel_x)) as avg_accel_x,
                                        AVG(ABS(accel_y)) as avg_accel_y,
                                        AVG(ABS(accel_z)) as avg_accel_z,
                                        AVG(ABS(gyro_x)) as avg_gyro_x,
                                        AVG(ABS(gyro_y)) as avg_gyro_y,
                                        AVG(ABS(gyro_z)) as avg_gyro_z,
                                        AVG(temperature_celsius) as avg_temp
                                    FROM {sensor_table} {time_filter}
                                """
                                async with db.execute(query2, params) as cursor2:
                                    row2 = await cursor2.fetchone()
                                    if row2:
                                        summary[sensor_table].update({
                                            'avg_acceleration_ms2': f"X:{round(row2[0], 2) if row2[0] else 0}, Y:{round(row2[1], 2) if row2[1] else 0}, Z:{round(row2[2], 2) if row2[2] else 0}",
                                            'avg_gyroscope_rads': f"X:{round(row2[3], 3) if row2[3] else 0}, Y:{round(row2[4], 3) if row2[4] else 0}, Z:{round(row2[5], 3) if row2[5] else 0}",
                                            'avg_temperature_c': round(row2[6], 1) if row2[6] else 0,
                                        })
                        else:
                            summary[sensor_table] = {'count': 0}
                except Exception:
                    summary[sensor_table] = {'count': 0}

            # Images summary
            try:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        SUM(file_size_bytes) as total_size,
                        COUNT(DISTINCT camera_name) as unique_cameras
                    FROM images {time_filter}
                """

                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['images'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'total_size_bytes': row[3] or 0,
                            'total_size_formatted': self.format_size(row[3] or 0),
                            'unique_cameras': row[4] or 0
                        }
                    else:
                        summary['images'] = {'count': 0}
            except Exception:
                summary['images'] = {'count': 0}

            # Tasks summary
            try:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        COUNT(DISTINCT task_name) as unique_tasks
                    FROM tasks {time_filter}
                """

                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['tasks'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'unique_tasks': row[3] or 0
                        }
                    else:
                        summary['tasks'] = {'count': 0}
            except Exception:
                summary['tasks'] = {'count': 0}

            # AI models summary
            try:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(created_at) as first_record,
                        MAX(created_at) as last_record,
                        COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_models
                    FROM ai_models
                """

                async with db.execute(query) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['ai_models'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'active_models': row[3] or 0
                        }
                    else:
                        summary['ai_models'] = {'count': 0}
            except Exception:
                summary['ai_models'] = {'count': 0}

            # AI processing queue summary
            try:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(created_at) as first_record,
                        MAX(created_at) as last_record,
                        COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                        COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                    FROM ai_processing_queue {time_filter.replace('timestamp', 'created_at') if time_filter else ''}
                """

                queue_params = {}
                if time_filter and since:
                    queue_params['since'] = since.isoformat()

                async with db.execute(query, queue_params) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['ai_processing_queue'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'pending': row[3] or 0,
                            'processing': row[4] or 0,
                            'completed': row[5] or 0,
                            'failed': row[6] or 0
                        }
                    else:
                        summary['ai_processing_queue'] = {'count': 0}
            except Exception:
                summary['ai_processing_queue'] = {'count': 0}

            # Segmentation results summary
            try:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record,
                        SUM(file_size_bytes) as total_size,
                        AVG(processing_time_ms) as avg_processing_time,
                        COUNT(DISTINCT ai_model_id) as unique_models
                    FROM segmentation_results {time_filter}
                """

                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        summary['segmentation_results'] = {
                            'count': row[0],
                            'time_range': f"{row[1]} to {row[2]}",
                            'total_size_bytes': row[3] or 0,
                            'total_size_formatted': self.format_size(row[3] or 0),
                            'avg_processing_time_ms': round(row[4], 1) if row[4] else 0,
                            'unique_models': row[5] or 0
                        }
                    else:
                        summary['segmentation_results'] = {'count': 0}
            except Exception:
                summary['segmentation_results'] = {'count': 0}

            return summary

    async def get_table_details(self, table_name: str, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get detailed information about a specific table."""
        time_filter = ""
        params = {}

        if since and table_name not in ['schema_version']:
            time_filter = "WHERE timestamp >= :since"
            params['since'] = since.isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            # Get table schema
            async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                schema = await cursor.fetchall()

            # Get record count and time range
            if 'timestamp' in [col[1] for col in schema]:
                query = f"""
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record
                    FROM {table_name} {time_filter}
                """
            else:
                query = f"SELECT COUNT(*) as count FROM {table_name}"

            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if 'timestamp' in [col[1] for col in schema] and row[0] > 0:
                    basic_info = {
                        'count': row[0],
                        'first_record': row[1],
                        'last_record': row[2]
                    }
                else:
                    basic_info = {'count': row[0] if row else 0}

            # Get sample records
            sample_query = f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT 5"
            async with db.execute(sample_query) as cursor:
                sample_records = await cursor.fetchall()

            return {
                'table_name': table_name,
                'schema': schema,
                'basic_info': basic_info,
                'sample_records': sample_records
            }

    def print_database_overview(self, db_info: Dict[str, Any]) -> None:
        """Print database overview."""
        self.print_header("DATABASE OVERVIEW")

        print(f"📁 Database File: {self.colors.CYAN}{db_info['file_path']}{self.colors.RESET}")
        print(f"💾 File Size: {self.colors.GREEN}{self.format_size(db_info['file_size'])}{self.colors.RESET}")
        print(f"📊 Total Tables: {self.colors.YELLOW}{len(db_info['tables'])}{self.colors.RESET}")
        print(f"📝 Total Records: {self.colors.YELLOW}{db_info['total_records']:,}{self.colors.RESET}")

        print(f"\n{self.colors.BOLD}Tables:{self.colors.RESET}")
        for table in sorted(db_info['tables']):
            info = db_info['table_info'].get(table, {})
            if 'error' in info:
                print(f"  ❌ {table}: {self.colors.RED}Error - {info['error']}{self.colors.RESET}")
            else:
                count = info.get('count', 0)
                columns = info.get('columns', 0)
                print(f"  📋 {table}: {self.colors.GREEN}{count:,} records{self.colors.RESET}, {columns} columns")

    def print_time_range_summary(self, summary: Dict[str, Any], since: Optional[datetime] = None) -> None:
        """Print time range summary."""
        time_desc = f"Since {since.strftime('%Y-%m-%d %H:%M:%S')}" if since else "All Time"
        self.print_header(f"DATA SUMMARY - {time_desc}")

        # System Metrics
        if 'system_metrics' in summary and summary['system_metrics']['count'] > 0:
            self.print_header("System Metrics", 2)
            metrics = summary['system_metrics']
            print(f"📊 Records: {self.colors.YELLOW}{metrics['count']:,}{self.colors.RESET}")
            print(f"⏰ Time Range: {self.colors.CYAN}{metrics['time_range']}{self.colors.RESET}")
            print(f"🖥️  CPU Usage: Avg {self.colors.GREEN}{metrics['avg_cpu_percent']}%{self.colors.RESET}, Max {self.colors.RED}{metrics['max_cpu_percent']}%{self.colors.RESET}")
            print(f"🌡️  Temperature: Avg {self.colors.GREEN}{metrics['avg_temperature_c']}°C{self.colors.RESET}, Max {self.colors.RED}{metrics['max_temperature_c']}°C{self.colors.RESET}")
            print(f"💾 RAM Usage: Avg {self.colors.GREEN}{metrics['avg_ram_used_gb']} GB{self.colors.RESET}, Max {self.colors.RED}{metrics['max_ram_used_gb']} GB{self.colors.RESET}")

        # Barometer
        if 'barometer_readings' in summary and summary['barometer_readings']['count'] > 0:
            self.print_header("Barometer Sensor", 2)
            baro = summary['barometer_readings']
            print(f"📊 Records: {self.colors.YELLOW}{baro['count']:,}{self.colors.RESET}")
            print(f"⏰ Time Range: {self.colors.CYAN}{baro['time_range']}{self.colors.RESET}")
            if 'avg_pressure_hpa' in baro:
                print(f"🌬️  Pressure: Avg {self.colors.GREEN}{baro['avg_pressure_hpa']} hPa{self.colors.RESET}, Range {self.colors.CYAN}{baro['pressure_range_hpa']} hPa{self.colors.RESET}")
                print(f"🌡️  Temperature: Avg {self.colors.GREEN}{baro['avg_temperature_c']}°C{self.colors.RESET}")
                print(f"⛰️  Altitude: Avg {self.colors.GREEN}{baro['avg_altitude_m']} m{self.colors.RESET}")

        # IMU
        if 'imu_readings' in summary and summary['imu_readings']['count'] > 0:
            self.print_header("IMU Sensor", 2)
            imu = summary['imu_readings']
            print(f"📊 Records: {self.colors.YELLOW}{imu['count']:,}{self.colors.RESET}")
            print(f"⏰ Time Range: {self.colors.CYAN}{imu['time_range']}{self.colors.RESET}")
            if 'avg_acceleration_ms2' in imu:
                print(f"📐 Acceleration (m/s²): {self.colors.GREEN}{imu['avg_acceleration_ms2']}{self.colors.RESET}")
                print(f"🔄 Gyroscope (rad/s): {self.colors.GREEN}{imu['avg_gyroscope_rads']}{self.colors.RESET}")
                print(f"🌡️  Temperature: Avg {self.colors.GREEN}{imu['avg_temperature_c']}°C{self.colors.RESET}")

        # Images
        if 'images' in summary and summary['images']['count'] > 0:
            self.print_header("Camera Images", 2)
            images = summary['images']
            print(f"📊 Records: {self.colors.YELLOW}{images['count']:,}{self.colors.RESET}")
            print(f"⏰ Time Range: {self.colors.CYAN}{images['time_range']}{self.colors.RESET}")
            print(f"📷 Unique Cameras: {self.colors.YELLOW}{images['unique_cameras']}{self.colors.RESET}")
            print(f"💾 Total Size: {self.colors.GREEN}{images['total_size_formatted']}{self.colors.RESET}")

        # Tasks
        if 'tasks' in summary and summary['tasks']['count'] > 0:
            self.print_header("Task Executions", 2)
            tasks = summary['tasks']
            print(f"📊 Records: {self.colors.YELLOW}{tasks['count']:,}{self.colors.RESET}")
            print(f"⏰ Time Range: {self.colors.CYAN}{tasks['time_range']}{self.colors.RESET}")
            print(f"🎯 Unique Tasks: {self.colors.YELLOW}{tasks['unique_tasks']}{self.colors.RESET}")

        # Collated Summary Table
        self.print_collated_summary_table(summary)

    def print_table_details(self, details: Dict[str, Any]) -> None:
        """Print detailed table information."""
        self.print_header(f"TABLE DETAILS - {details['table_name'].upper()}")

        # Schema
        print(f"{self.colors.BOLD}Schema:{self.colors.RESET}")
        for col in details['schema']:
            col_id, name, col_type, not_null, default, pk = col
            indicators = []
            if pk:
                indicators.append(f"{self.colors.YELLOW}PK{self.colors.RESET}")
            if not_null:
                indicators.append(f"{self.colors.RED}NOT NULL{self.colors.RESET}")
            if default:
                indicators.append(f"{self.colors.CYAN}DEFAULT({default}){self.colors.RESET}")

            indicator_str = f" [{', '.join(indicators)}]" if indicators else ""
            print(f"  {col_id+1:2d}. {self.colors.GREEN}{name}{self.colors.RESET} ({self.colors.BLUE}{col_type}{self.colors.RESET}){indicator_str}")

        # Basic info
        print(f"\n{self.colors.BOLD}Statistics:{self.colors.RESET}")
        basic_info = details['basic_info']
        print(f"📊 Record Count: {self.colors.YELLOW}{basic_info['count']:,}{self.colors.RESET}")

        if 'first_record' in basic_info:
            print(f"⏰ First Record: {self.colors.CYAN}{basic_info['first_record']}{self.colors.RESET}")
            print(f"⏰ Last Record: {self.colors.CYAN}{basic_info['last_record']}{self.colors.RESET}")

        # Sample records
        if details['sample_records']:
            print(f"\n{self.colors.BOLD}Sample Records (Latest 5):{self.colors.RESET}")
            column_names = [col[1] for col in details['schema']]

            # Calculate column widths
            widths = [len(name) for name in column_names]
            for record in details['sample_records'][:3]:  # Only check first 3 for width
                for i, value in enumerate(record):
                    widths[i] = max(widths[i], len(str(value)) if value is not None else 4)

            # Limit column width to reasonable size
            widths = [min(w, 20) for w in widths]

            # Print header
            header = " | ".join(f"{name[:widths[i]]:<{widths[i]}}" for i, name in enumerate(column_names))
            print(f"  {self.colors.DIM}{header}{self.colors.RESET}")
            print(f"  {self.colors.DIM}{'-' * len(header)}{self.colors.RESET}")

            # Print records
            for record in details['sample_records']:
                row = " | ".join(f"{str(value)[:widths[i]] if value is not None else 'NULL':<{widths[i]}}"
                               for i, value in enumerate(record))
                print(f"  {row}")

    def print_collated_summary_table(self, summary: Dict[str, Any]) -> None:
        """Print a final collated summary table with all components."""
        self.print_header("COLLATED SYSTEM SUMMARY", 2)

        # Collect data for the table
        table_data = []

        # System Metrics
        if 'system_metrics' in summary and summary['system_metrics']['count'] > 0:
            metrics = summary['system_metrics']
            table_data.append([
                "System Metrics",
                f"{metrics['count']:,}",
                f"{metrics.get('avg_cpu_percent', 0):.1f}%",
                f"{metrics.get('avg_temperature_c', 0):.1f}°C",
                f"{metrics.get('avg_ram_used_gb', 0):.2f} GB",
                "CPU/RAM/Temp monitoring"
            ])

        # Barometer
        if 'barometer_readings' in summary and summary['barometer_readings']['count'] > 0:
            baro = summary['barometer_readings']
            table_data.append([
                "Barometer",
                f"{baro['count']:,}",
                f"{baro.get('avg_pressure_hpa', 0):.1f} hPa",
                f"{baro.get('avg_temperature_c', 0):.1f}°C",
                f"{baro.get('avg_altitude_m', 0):.1f} m",
                "Pressure/altitude sensing"
            ])

        # IMU
        if 'imu_readings' in summary and summary['imu_readings']['count'] > 0:
            imu = summary['imu_readings']
            accel_summary = "Active" if 'avg_acceleration_ms2' in imu else "Limited"
            table_data.append([
                "IMU Sensor",
                f"{imu['count']:,}",
                accel_summary,
                f"{imu.get('avg_temperature_c', 0):.1f}°C",
                "6-axis data",
                "Motion/orientation sensing"
            ])

        # Camera
        if 'images' in summary and summary['images']['count'] > 0:
            images = summary['images']
            table_data.append([
                "Camera System",
                f"{images['count']:,}",
                f"{images.get('unique_cameras', 0)} cameras",
                f"{images.get('total_size_formatted', '0 B')}",
                "Image capture",
                "Visual data collection"
            ])

        # Tasks
        if 'tasks' in summary and summary['tasks']['count'] > 0:
            tasks = summary['tasks']
            table_data.append([
                "Task System",
                f"{tasks['count']:,}",
                f"{tasks.get('unique_tasks', 0)} types",
                "Execution logs",
                "Task coordination",
                "System orchestration"
            ])

        if not table_data:
            print(f"{self.colors.YELLOW}No data available for collated summary{self.colors.RESET}")
            return

        # Define headers and column widths
        headers = ["Component", "Records", "Key Metric", "Secondary", "Data Type", "Purpose"]
        col_widths = [15, 10, 15, 12, 15, 25]

        # Print table header
        header_line = " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        print(f"  {self.colors.BOLD}{header_line}{self.colors.RESET}")
        print(f"  {self.colors.DIM}{'-' * len(header_line)}{self.colors.RESET}")

        # Print table rows
        for row in table_data:
            # Truncate long values and pad
            formatted_row = []
            for i, value in enumerate(row):
                truncated = str(value)[:col_widths[i]]
                formatted_row.append(f"{truncated:<{col_widths[i]}}")

            row_line = " | ".join(formatted_row)
            print(f"  {row_line}")

        # Print summary statistics
        total_records = sum(int(row[1].replace(',', '')) for row in table_data if row[1].replace(',', '').isdigit())
        active_components = len(table_data)

        print(f"\n{self.colors.BOLD}Summary Statistics:{self.colors.RESET}")
        print(f"  🔧 Active Components: {self.colors.GREEN}{active_components}{self.colors.RESET}")
        print(f"  📊 Total Records: {self.colors.YELLOW}{total_records:,}{self.colors.RESET}")
        print(f"  ⚡ System Status: {self.colors.GREEN}Operational{self.colors.RESET}")

        # Health indicators
        health_indicators = []

        if 'system_metrics' in summary and summary['system_metrics']['count'] > 0:
            metrics = summary['system_metrics']
            avg_cpu = metrics.get('avg_cpu_percent', 0)
            avg_temp = metrics.get('avg_temperature_c', 0)

            if avg_cpu > 80:
                health_indicators.append(f"{self.colors.RED}High CPU usage{self.colors.RESET}")
            elif avg_cpu > 50:
                health_indicators.append(f"{self.colors.YELLOW}Moderate CPU usage{self.colors.RESET}")
            else:
                health_indicators.append(f"{self.colors.GREEN}Normal CPU usage{self.colors.RESET}")

            if avg_temp > 60:
                health_indicators.append(f"{self.colors.RED}High temperature{self.colors.RESET}")
            elif avg_temp > 50:
                health_indicators.append(f"{self.colors.YELLOW}Warm temperature{self.colors.RESET}")
            else:
                health_indicators.append(f"{self.colors.GREEN}Normal temperature{self.colors.RESET}")

        # Data collection rate
        if 'system_metrics' in summary and summary['system_metrics']['count'] > 0:
            metrics = summary['system_metrics']
            time_range = metrics.get('time_range', '')
            if ' to ' in time_range:
                try:
                    start_str, end_str = time_range.split(' to ')
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    if duration_hours > 0:
                        records_per_hour = total_records / duration_hours
                        health_indicators.append(f"{self.colors.CYAN}{records_per_hour:.1f} records/hour{self.colors.RESET}")
                except:
                    pass

        if health_indicators:
            print(f"  🏥 Health: {', '.join(health_indicators)}")

    async def export_summary(self, summary_data: Dict[str, Any], export_format: str, output_file: str) -> None:
        """Export summary to file."""
        try:
            if export_format.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(summary_data, f, indent=2, default=str)

            elif export_format.lower() == 'csv':
                # Flatten the data for CSV export
                flattened = []
                for category, data in summary_data.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            flattened.append({
                                'category': category,
                                'metric': key,
                                'value': value
                            })

                with open(output_file, 'w', newline='') as f:
                    if flattened:
                        writer = csv.DictWriter(f, fieldnames=['category', 'metric', 'value'])
                        writer.writeheader()
                        writer.writerows(flattened)

            elif export_format.lower() == 'html':
                html_content = self.generate_html_report(summary_data)
                with open(output_file, 'w') as f:
                    f.write(html_content)

            print(f"\n{self.colors.GREEN}✓ Summary exported to: {output_file}{self.colors.RESET}")

        except Exception as e:
            print(f"\n{self.colors.RED}✗ Export failed: {e}{self.colors.RESET}")

    def generate_html_report(self, summary_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Macha Database Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #4CAF50; }}
        .metric {{ margin: 5px 0; }}
        .value {{ font-weight: bold; color: #2196F3; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Macha Database Summary Report</h1>
        <p>Generated on: {timestamp}</p>
        <p>Database: {self.db_path}</p>
    </div>
"""

        for category, data in summary_data.items():
            html += f'<div class="section"><h2>{category.replace("_", " ").title()}</h2>'

            if isinstance(data, dict):
                for key, value in data.items():
                    html += f'<div class="metric">{key.replace("_", " ").title()}: <span class="value">{value}</span></div>'
            else:
                html += f'<div class="metric">Value: <span class="value">{data}</span></div>'

            html += '</div>'

        html += """
</body>
</html>
"""
        return html


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Database Summary Tool for Macha Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s                          # Show complete summary
          %(prog)s --since 24h              # Show data from last 24 hours
          %(prog)s --table system_metrics   # Show details for specific table
          %(prog)s --export json report.json # Export to JSON file
          %(prog)s --verbose --no-color     # Detailed output without colors
        """)
    )

    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--db', help='Path to database file (overrides config)')
    parser.add_argument('--table', help='Show detailed summary for specific table')
    parser.add_argument('--since', help='Show data from last N hours/days (e.g., "24h", "7d")')
    parser.add_argument('--export', help='Export summary to file (json, csv, html)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed statistics')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')

    args = parser.parse_args()

    try:
        # Determine database path
        if args.db:
            db_path = args.db
        else:
            try:
                config = load_config(args.config)
                db_path = config.db.filename
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Please specify database path with --db option")
                return 1

        # Check if database exists
        if not Path(db_path).exists():
            print(f"Error: Database file not found: {db_path}")
            return 1

        # Parse time filter
        since_datetime = None
        if args.since:
            try:
                db_summary = DatabaseSummary(db_path, use_colors=not args.no_color)
                since_datetime = db_summary.parse_time_filter(args.since)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        # Create summary tool
        db_summary = DatabaseSummary(db_path, use_colors=not args.no_color)

        # Show table details if requested
        if args.table:
            details = await db_summary.get_table_details(args.table, since_datetime)
            db_summary.print_table_details(details)
            return 0

        # Get database overview
        db_info = await db_summary.get_database_info()
        db_summary.print_database_overview(db_info)

        # Get time-range summary
        summary = await db_summary.get_time_range_summary(since_datetime)
        db_summary.print_time_range_summary(summary, since_datetime)

        # Export if requested
        if args.export:
            export_format = args.export.split('.')[-1] if '.' in args.export else 'json'
            output_file = args.export if '.' in args.export else f"macha_summary.{export_format}"

            export_data = {
                'database_info': db_info,
                'summary': summary,
                'generated_at': datetime.now().isoformat(),
                'time_filter': args.since
            }

            await db_summary.export_summary(export_data, export_format, output_file)

        return 0

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.RESET}")
        return 1
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
