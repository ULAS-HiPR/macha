#!/usr/bin/env python3
"""
Database Visualization Tool for Macha Project

This script creates visualizations and charts for the data collected in the
Macha database. It generates plots for system metrics, sensor readings,
and provides visual analysis of trends and patterns.

Usage:
    python scripts/db_visualize.py [OPTIONS]

Options:
    --config PATH       Path to config file (default: config.yaml)
    --db PATH          Path to database file (overrides config)
    --output DIR       Output directory for plots (default: plots/)
    --since DURATION   Show data from last N hours/days (e.g., "24h", "7d")
    --type TYPE        Type of visualization (all, system, sensors, images)
    --format FORMAT    Output format (png, svg, pdf, html)
    --no-show          Don't display plots (just save them)
    --verbose          Show detailed information
    --help             Show this help message
"""

import sys
import argparse
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import textwrap

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import pandas as pd
    import numpy as np
    from config import load_config, MachaConfig
    HAS_PLOTTING = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Install with: pip install matplotlib pandas numpy")
    HAS_PLOTTING = False
    # Create dummy classes for type hints
    class pd:
        class DataFrame:
            pass
    plt = None
    mdates = None
    np = None


class DataVisualizer:
    """Database visualization and plotting tool."""

    def __init__(self, db_path: str, output_dir: str = "plots", show_plots: bool = True):
        if not HAS_PLOTTING:
            raise ImportError("Plotting libraries not available. Install with: pip install matplotlib pandas numpy")

        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.show_plots = show_plots
        self.output_dir.mkdir(exist_ok=True)

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

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

    def load_data(self, table: str, since: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from database table into pandas DataFrame."""
        time_filter = ""
        params = []

        if since and table not in ['schema_version', 'sqlite_sequence']:
            time_filter = "WHERE timestamp >= ?"
            params.append(since.isoformat())

        query = f"SELECT * FROM {table} {time_filter} ORDER BY timestamp"

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)

                # Convert timestamp column to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)

                return df
        except Exception as e:
            print(f"Error loading data from {table}: {e}")
            return pd.DataFrame()

    def plot_system_metrics(self, since: Optional[datetime] = None, output_format: str = 'png') -> List[str]:
        """Create system metrics visualizations."""
        print("📊 Generating system metrics plots...")

        df = self.load_data('system_metrics', since)
        if df.empty:
            print("  No system metrics data available")
            return []

        saved_files = []

        # CPU and Temperature plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # CPU Usage
        ax1.plot(df.index, df['cpu_percent'], color='#1f77b4', linewidth=1.5, label='CPU Usage')
        ax1.fill_between(df.index, df['cpu_percent'], alpha=0.3, color='#1f77b4')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Performance Metrics')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Temperature
        if 'temperature_c' in df.columns and not df['temperature_c'].isna().all():
            ax2.plot(df.index, df['temperature_c'], color='#ff7f0e', linewidth=1.5, label='Temperature')
            ax2.fill_between(df.index, df['temperature_c'], alpha=0.3, color='#ff7f0e')
            ax2.set_ylabel('Temperature (°C)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Add temperature warning zones
            ax2.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='High Temp (60°C)')
            ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Warm (50°C)')

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 10)))

        plt.tight_layout()
        filename = self.output_dir / f"system_metrics.{output_format}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(str(filename))

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        # Memory Usage plot
        if 'ram_used_gb' in df.columns and not df['ram_used_gb'].isna().all():
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(df.index, df['ram_used_gb'], color='#2ca02c', linewidth=2, label='RAM Used')
            if 'ram_total_gb' in df.columns:
                ax.axhline(y=df['ram_total_gb'].iloc[0], color='red', linestyle='-', alpha=0.7, label='Total RAM')

            ax.fill_between(df.index, df['ram_used_gb'], alpha=0.3, color='#2ca02c')
            ax.set_ylabel('Memory (GB)')
            ax.set_xlabel('Time')
            ax.set_title('Memory Usage Over Time')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 10)))

            plt.tight_layout()
            filename = self.output_dir / f"memory_usage.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        return saved_files

    def plot_sensor_data(self, since: Optional[datetime] = None, output_format: str = 'png') -> List[str]:
        """Create sensor data visualizations."""
        print("🌡️ Generating sensor data plots...")

        saved_files = []

        # Barometer data
        baro_df = self.load_data('barometer_readings', since)
        if not baro_df.empty:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

            # Pressure
            ax1.plot(baro_df.index, baro_df['pressure_hpa'], color='#9467bd', linewidth=2, marker='o', markersize=3)
            ax1.set_ylabel('Pressure (hPa)')
            ax1.set_title('Barometer Sensor Readings')
            ax1.grid(True, alpha=0.3)

            # Temperature
            if 'temperature_celsius' in baro_df.columns:
                ax2.plot(baro_df.index, baro_df['temperature_celsius'], color='#ff7f0e', linewidth=2, marker='o', markersize=3)
                ax2.set_ylabel('Temperature (°C)')
                ax2.grid(True, alpha=0.3)

            # Altitude
            if 'altitude_meters' in baro_df.columns:
                ax3.plot(baro_df.index, baro_df['altitude_meters'], color='#8c564b', linewidth=2, marker='o', markersize=3)
                ax3.set_ylabel('Altitude (m)')
                ax3.set_xlabel('Time')
                ax3.grid(True, alpha=0.3)

            # Format x-axis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(baro_df) // 5)))

            plt.tight_layout()
            filename = self.output_dir / f"barometer_readings.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        # IMU data
        imu_df = self.load_data('imu_readings', since)
        if not imu_df.empty and len(imu_df) > 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Acceleration
            if all(col in imu_df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
                ax1.plot(imu_df.index, imu_df['accel_x'], label='X-axis', linewidth=2, marker='o', markersize=3)
                ax1.plot(imu_df.index, imu_df['accel_y'], label='Y-axis', linewidth=2, marker='s', markersize=3)
                ax1.plot(imu_df.index, imu_df['accel_z'], label='Z-axis', linewidth=2, marker='^', markersize=3)
                ax1.set_ylabel('Acceleration (m/s²)')
                ax1.set_title('IMU Sensor Readings')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Gyroscope
            if all(col in imu_df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                ax2.plot(imu_df.index, imu_df['gyro_x'], label='X-axis', linewidth=2, marker='o', markersize=3)
                ax2.plot(imu_df.index, imu_df['gyro_y'], label='Y-axis', linewidth=2, marker='s', markersize=3)
                ax2.plot(imu_df.index, imu_df['gyro_z'], label='Z-axis', linewidth=2, marker='^', markersize=3)
                ax2.set_ylabel('Angular Velocity (rad/s)')
                ax2.set_xlabel('Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S\n%m-%d'))

            plt.tight_layout()
            filename = self.output_dir / f"imu_readings.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        return saved_files

    def plot_image_statistics(self, since: Optional[datetime] = None, output_format: str = 'png') -> List[str]:
        """Create image capture statistics visualizations."""
        print("📷 Generating image statistics plots...")

        df = self.load_data('images', since)
        if df.empty:
            print("  No image data available")
            return []

        saved_files = []

        # Images per hour
        df_hourly = df.resample('H').size()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Images per hour
        ax1.bar(df_hourly.index, df_hourly.values, width=0.03, alpha=0.7, color='#d62728')
        ax1.set_ylabel('Images per Hour')
        ax1.set_title('Image Capture Statistics')
        ax1.grid(True, alpha=0.3)

        # Cumulative file size
        if 'file_size_bytes' in df.columns:
            df['cumulative_size_mb'] = df['file_size_bytes'].cumsum() / (1024 * 1024)
            ax2.plot(df.index, df['cumulative_size_mb'], color='#17becf', linewidth=2)
            ax2.fill_between(df.index, df['cumulative_size_mb'], alpha=0.3, color='#17becf')
            ax2.set_ylabel('Cumulative Size (MB)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df_hourly) // 10)))

        plt.tight_layout()
        filename = self.output_dir / f"image_statistics.{output_format}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(str(filename))

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        return saved_files

    def plot_task_analysis(self, since: Optional[datetime] = None, output_format: str = 'png') -> List[str]:
        """Create task execution analysis visualizations."""
        print("🎯 Generating task analysis plots...")

        df = self.load_data('tasks', since)
        if df.empty:
            print("  No task data available")
            return []

        saved_files = []

        # Task frequency analysis
        task_counts = df['task_name'].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Task distribution pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(task_counts)))
        wedges, texts, autotexts = ax1.pie(task_counts.values, labels=task_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Task Distribution')

        # Task executions over time
        df_hourly = df.groupby([df.index.floor('H'), 'task_name']).size().unstack(fill_value=0)

        bottom = np.zeros(len(df_hourly))
        for i, task in enumerate(df_hourly.columns):
            ax2.bar(df_hourly.index, df_hourly[task], bottom=bottom,
                   label=task, alpha=0.8, color=colors[i % len(colors)])
            bottom += df_hourly[task]

        ax2.set_ylabel('Tasks per Hour')
        ax2.set_xlabel('Time')
        ax2.set_title('Task Execution Timeline')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))

        plt.tight_layout()
        filename = self.output_dir / f"task_analysis.{output_format}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        saved_files.append(str(filename))

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        return saved_files

    def plot_ai_processing(self, since: Optional[datetime] = None, output_format: str = 'png') -> List[str]:
        """Create AI processing visualizations."""
        print("🤖 Generating AI processing plots...")

        saved_files = []

        # Load AI processing queue data
        queue_df = self.load_data('ai_processing_queue', since)
        results_df = self.load_data('segmentation_results', since)

        if queue_df.empty and results_df.empty:
            print("  No AI processing data available")
            return []

        # Processing queue status over time
        if not queue_df.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Queue status distribution over time
            queue_hourly = queue_df.groupby([queue_df.index.floor('H'), 'status']).size().unstack(fill_value=0)

            if not queue_hourly.empty:
                queue_hourly.plot(kind='bar', stacked=True, ax=ax1,
                                color={'pending': '#ff7f0e', 'processing': '#1f77b4',
                                      'completed': '#2ca02c', 'failed': '#d62728'})
                ax1.set_title('AI Processing Queue Status Over Time')
                ax1.set_ylabel('Number of Items')
                ax1.legend(title='Status')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)

            # Processing attempts distribution
            if 'attempts' in queue_df.columns:
                attempt_counts = queue_df['attempts'].value_counts().sort_index()
                ax2.bar(attempt_counts.index, attempt_counts.values, alpha=0.7, color='#9467bd')
                ax2.set_title('Processing Attempts Distribution')
                ax2.set_xlabel('Number of Attempts')
                ax2.set_ylabel('Count')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = self.output_dir / f"ai_processing_queue.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        # Segmentation results analysis
        if not results_df.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Processing time over time
            if 'processing_time_ms' in results_df.columns:
                ax1.plot(results_df.index, results_df['processing_time_ms'],
                        color='#17becf', linewidth=2, marker='o', markersize=4, alpha=0.7)
                ax1.set_title('AI Processing Performance')
                ax1.set_ylabel('Processing Time (ms)')
                ax1.grid(True, alpha=0.3)

                # Add trend line
                if len(results_df) > 1:
                    z = np.polyfit(range(len(results_df)), results_df['processing_time_ms'], 1)
                    p = np.poly1d(z)
                    ax1.plot(results_df.index, p(range(len(results_df))),
                            "--", color='red', alpha=0.8, linewidth=2, label='Trend')
                    ax1.legend()

            # Results per hour
            results_hourly = results_df.resample('H').size()
            ax2.bar(results_hourly.index, results_hourly.values, width=0.03, alpha=0.7, color='#2ca02c')
            ax2.set_title('AI Results Generated per Hour')
            ax2.set_ylabel('Results per Hour')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(results_hourly) // 10)))

            plt.tight_layout()
            filename = self.output_dir / f"ai_segmentation_results.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        # Combined AI system health dashboard
        if not queue_df.empty or not results_df.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Success rate pie chart
            if not queue_df.empty:
                status_counts = queue_df['status'].value_counts()
                colors = {'completed': '#2ca02c', 'failed': '#d62728',
                         'pending': '#ff7f0e', 'processing': '#1f77b4'}
                pie_colors = [colors.get(status, '#808080') for status in status_counts.index]

                ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                       colors=pie_colors, startangle=90)
                ax1.set_title('AI Processing Success Rate')

            # Processing time distribution
            if not results_df.empty and 'processing_time_ms' in results_df.columns:
                ax2.hist(results_df['processing_time_ms'], bins=20, alpha=0.7, color='#17becf', edgecolor='black')
                ax2.set_title('Processing Time Distribution')
                ax2.set_xlabel('Processing Time (ms)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)

            # Queue length over time (if we can calculate it)
            if not queue_df.empty:
                queue_timeline = queue_df.groupby(queue_df.index.floor('H')).size()
                ax3.plot(queue_timeline.index, queue_timeline.values,
                        color='#ff7f0e', linewidth=2, marker='s', markersize=4)
                ax3.set_title('AI Queue Activity Over Time')
                ax3.set_ylabel('Queue Items per Hour')
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)

            # File size distribution for results
            if not results_df.empty and 'file_size_bytes' in results_df.columns:
                file_sizes_mb = results_df['file_size_bytes'] / (1024 * 1024)
                ax4.hist(file_sizes_mb, bins=15, alpha=0.7, color='#9467bd', edgecolor='black')
                ax4.set_title('AI Output File Size Distribution')
                ax4.set_xlabel('File Size (MB)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = self.output_dir / f"ai_system_dashboard.{output_format}"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files.append(str(filename))

            if self.show_plots:
                plt.show()
            else:
                plt.close()

        return saved_files

    def create_dashboard(self, since: Optional[datetime] = None, output_format: str = 'png') -> str:
        """Create a comprehensive dashboard view."""
        print("📊 Generating comprehensive dashboard...")

        # Load all data
        system_df = self.load_data('system_metrics', since)
        baro_df = self.load_data('barometer_readings', since)
        imu_df = self.load_data('imu_readings', since)
        images_df = self.load_data('images', since)
        tasks_df = self.load_data('tasks', since)

        # Create dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # System CPU
        if not system_df.empty:
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(system_df.index, system_df['cpu_percent'], color='#1f77b4', linewidth=1.5)
            ax1.fill_between(system_df.index, system_df['cpu_percent'], alpha=0.3, color='#1f77b4')
            ax1.set_title('CPU Usage (%)')
            ax1.grid(True, alpha=0.3)

            # System Temperature
            ax2 = fig.add_subplot(gs[0, 2:])
            if 'temperature_c' in system_df.columns:
                ax2.plot(system_df.index, system_df['temperature_c'], color='#ff7f0e', linewidth=1.5)
                ax2.fill_between(system_df.index, system_df['temperature_c'], alpha=0.3, color='#ff7f0e')
            ax2.set_title('Temperature (°C)')
            ax2.grid(True, alpha=0.3)

        # Barometer pressure
        if not baro_df.empty:
            ax3 = fig.add_subplot(gs[1, :2])
            ax3.plot(baro_df.index, baro_df['pressure_hpa'], color='#9467bd', linewidth=2, marker='o', markersize=2)
            ax3.set_title('Atmospheric Pressure (hPa)')
            ax3.grid(True, alpha=0.3)

        # IMU acceleration magnitude
        if not imu_df.empty and len(imu_df) > 1:
            ax4 = fig.add_subplot(gs[1, 2:])
            if all(col in imu_df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
                accel_mag = np.sqrt(imu_df['accel_x']**2 + imu_df['accel_y']**2 + imu_df['accel_z']**2)
                ax4.plot(imu_df.index, accel_mag, color='#2ca02c', linewidth=2, marker='o', markersize=2)
            ax4.set_title('Acceleration Magnitude (m/s²)')
            ax4.grid(True, alpha=0.3)

        # Image count per hour
        if not images_df.empty:
            ax5 = fig.add_subplot(gs[2, :2])
            images_hourly = images_df.resample('H').size()
            ax5.bar(images_hourly.index, images_hourly.values, width=0.03, alpha=0.7, color='#d62728')
            ax5.set_title('Images per Hour')
            ax5.grid(True, alpha=0.3)

        # Task distribution
        if not tasks_df.empty:
            ax6 = fig.add_subplot(gs[2, 2:])
            task_counts = tasks_df['task_name'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(task_counts)))
            ax6.pie(task_counts.values, labels=task_counts.index, autopct='%1.1f%%', colors=colors)
            ax6.set_title('Task Distribution')

        # Memory usage
        if not system_df.empty and 'ram_used_gb' in system_df.columns:
            ax7 = fig.add_subplot(gs[3, :])
            ax7.plot(system_df.index, system_df['ram_used_gb'], color='#2ca02c', linewidth=2)
            ax7.fill_between(system_df.index, system_df['ram_used_gb'], alpha=0.3, color='#2ca02c')
            ax7.set_title('Memory Usage (GB)')
            ax7.set_xlabel('Time')
            ax7.grid(True, alpha=0.3)

        # Overall title
        time_desc = f"Since {since.strftime('%Y-%m-%d %H:%M:%S')}" if since else "All Time"
        fig.suptitle(f'Macha System Dashboard - {time_desc}', fontsize=16, fontweight='bold')

        filename = self.output_dir / f"dashboard.{output_format}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if self.show_plots:
            plt.show()
        else:
            plt.close()

        return str(filename)


async def main():
    """Main function."""
    if not HAS_PLOTTING:
        print("Error: Required plotting libraries not available.")
        print("Install with: pip install matplotlib pandas numpy")
        return 1

    parser = argparse.ArgumentParser(
        description="Database Visualization Tool for Macha Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          %(prog)s                          # Generate all plots
          %(prog)s --since 24h              # Plot data from last 24 hours
          %(prog)s --type system            # Only system metrics plots
          %(prog)s --type sensors --format svg # Sensor plots in SVG format
          %(prog)s --no-show --output /tmp/plots # Save to custom directory
        """)
    )

    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--db', help='Path to database file (overrides config)')
    parser.add_argument('--output', default='plots', help='Output directory for plots')
    parser.add_argument('--since', help='Show data from last N hours/days (e.g., "24h", "7d")')
    parser.add_argument('--type', choices=['all', 'system', 'sensors', 'images', 'tasks', 'ai', 'dashboard'],
                       default='all', help='Type of visualization')
    parser.add_argument('--format', choices=['png', 'svg', 'pdf'], default='png',
                       help='Output format')
    parser.add_argument('--no-show', action='store_true', help="Don't display plots")
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')

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
                visualizer = DataVisualizer(db_path, args.output, show_plots=not args.no_show)
                since_datetime = visualizer.parse_time_filter(args.since)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        # Create visualizer
        visualizer = DataVisualizer(db_path, args.output, show_plots=not args.no_show)

        print(f"🎨 Creating visualizations...")
        print(f"📁 Database: {db_path}")
        print(f"📊 Output: {args.output}")
        print(f"🕒 Time filter: {args.since or 'All time'}")
        print(f"📈 Format: {args.format}")

        saved_files = []

        if args.type in ['all', 'system']:
            files = visualizer.plot_system_metrics(since_datetime, args.format)
            saved_files.extend(files)

        if args.type in ['all', 'sensors']:
            files = visualizer.plot_sensor_data(since_datetime, args.format)
            saved_files.extend(files)

        if args.type in ['all', 'images']:
            files = visualizer.plot_image_statistics(since_datetime, args.format)
            saved_files.extend(files)

        if args.type in ['all', 'tasks']:
            files = visualizer.plot_task_analysis(since_datetime, args.format)
            saved_files.extend(files)

        if args.type in ['all', 'ai']:
            files = visualizer.plot_ai_processing(since_datetime, args.format)
            saved_files.extend(files)

        if args.type in ['dashboard']:
            filename = visualizer.create_dashboard(since_datetime, args.format)
            saved_files.append(filename)

        if args.type == 'all':
            filename = visualizer.create_dashboard(since_datetime, args.format)
            saved_files.append(filename)

        print(f"\n✅ Generated {len(saved_files)} visualization(s):")
        for file in saved_files:
            print(f"   📊 {file}")

        return 0

    except KeyboardInterrupt:
        print(f"\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
