from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text
from pathlib import Path
import asyncio
import logging
from config import MachaConfig


async def init_database(config: MachaConfig = None) -> AsyncEngine:
    """Initialize the SQLite database based on config."""
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
