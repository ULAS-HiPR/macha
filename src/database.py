from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
from pathlib import Path
import asyncio
import logging
from config import MachaConfig


class DatabaseManager:
    """Centralized database schema management and migrations."""
    
    def __init__(self, engine: AsyncEngine, logger: logging.Logger):
        self.engine = engine
        self.logger = logger
    
    async def apply_migrations(self):
        """Apply all database schema migrations."""
        self.logger.info("Applying database migrations...")
        
        # Create schema version table first
        await self._create_schema_version_table()
        
        # Get current schema version
        current_version = await self._get_schema_version()
        self.logger.info(f"Current schema version: {current_version}")
        
        # Apply migrations in order
        migrations = [
            (1, self._create_initial_tables),
            (2, self._create_camera_tables),
            (3, self._create_metrics_tables),
            (4, self._create_sensor_tables),
        ]
        
        for version, migration_func in migrations:
            if current_version < version:
                self.logger.info(f"Applying migration {version}")
                await migration_func()
                await self._update_schema_version(version)
                current_version = version
        
        self.logger.info("Database migrations complete")
    
    async def _create_schema_version_table(self):
        """Create schema version tracking table."""
        async with self.engine.connect() as conn:
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
            )
            # Insert initial version if table is empty
            await conn.execute(
                text("""
                INSERT OR IGNORE INTO schema_version (version) VALUES (0)
                """)
            )
            await conn.commit()
    
    async def _get_schema_version(self) -> int:
        """Get current schema version."""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT MAX(version) FROM schema_version")
            )
            version = result.scalar()
            return version or 0
    
    async def _update_schema_version(self, version: int):
        """Update schema version."""
        async with self.engine.connect() as conn:
            await conn.execute(
                text("INSERT INTO schema_version (version) VALUES (:version)"),
                {"version": version}
            )
            await conn.commit()
    
    async def _create_initial_tables(self):
        """Migration 1: Create initial task execution table."""
        async with self.engine.connect() as conn:
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
            )
            await conn.commit()
    
    async def _create_camera_tables(self):
        """Migration 2: Create camera/image tables."""
        async with self.engine.connect() as conn:
            await conn.execute(
                text("""
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
                )
                """)
            )
            await conn.commit()
    
    async def _create_metrics_tables(self):
        """Migration 3: Create system metrics tables."""
        async with self.engine.connect() as conn:
            await conn.execute(
                text("""
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
                )
                """)
            )
            await conn.commit()
    
    async def _create_sensor_tables(self):
        """Migration 4: Create sensor data tables."""
        async with self.engine.connect() as conn:
            # Barometer readings table
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS barometer_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pressure_hpa REAL,
                    temperature_celsius REAL,
                    altitude_meters REAL,
                    sea_level_pressure REAL,
                    sensor_config TEXT
                )
                """)
            )
            
            # IMU readings table
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


async def init_database(config: MachaConfig = None) -> AsyncEngine:
    """Initialize the SQLite database based on config and apply migrations."""
    if config is None:
        from config import load_config

        config = load_config()

    from logger import setup_logger

    logger = setup_logger(config)

    filename = config.db.filename
    connection_string = config.db.connection_string
    overwrite = config.db.overwrite

    # Ensure database directory exists
    db_path = Path(filename)
    db_dir = db_path.parent
    db_dir.mkdir(parents=True, exist_ok=True)

    # If overwrite is True, delete existing database
    if overwrite and db_path.exists():
        logger.info(f"Overwriting existing database: {filename}")
        db_path.unlink()

    # Create async engine
    try:
        engine = create_async_engine(connection_string, echo=config.app.debug)
        logger.info(f"Database initialized: {connection_string}")

        # Test connection
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            logger.debug("Database connection test successful")

        # Apply database migrations
        db_manager = DatabaseManager(engine, logger)
        await db_manager.apply_migrations()

        return engine
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def main():
    """Example usage of database initialization."""
    from logger import setup_logger

    logger = setup_logger()
    engine = await init_database()
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT sqlite_version()"))
        version = result.scalar()
        logger.info(f"SQLite version: {version}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
