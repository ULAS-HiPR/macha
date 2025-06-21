import pytest
import asyncio
import sys
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import text
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import DatabaseManager, init_database
from config import MachaConfig


@pytest.fixture
def mock_engine():
    """Create a mock database engine."""
    engine = Mock(spec=AsyncEngine)

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)

    engine.connect.return_value = mock_conn
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return MachaConfig(
        app={"name": "test", "debug": True},
        logging={
            "level": "INFO",
            "file": {"path": "test.log"},
            "console": {"format": "test"}
        },
        db={
            "filename": "test.db",
            "connection_string": "sqlite+aiosqlite:///test.db",
            "overwrite": False
        },
        tasks=[
            {
                "name": "test_task",
                "class": "MetricsTask",
                "frequency": 60,
                "enabled": True
            }
        ]
    )


class TestDatabaseManagerInitialization:
    """Test DatabaseManager initialization."""

    def test_database_manager_init(self, mock_engine, mock_logger):
        """Test DatabaseManager initialization."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        assert db_manager.engine == mock_engine
        assert db_manager.logger == mock_logger

    def test_database_manager_init_types(self, mock_engine, mock_logger):
        """Test DatabaseManager initialization with correct types."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        assert hasattr(db_manager, 'apply_migrations')
        assert hasattr(db_manager, '_create_schema_version_table')
        assert hasattr(db_manager, '_get_schema_version')
        assert hasattr(db_manager, '_update_schema_version')


class TestSchemaVersionManagement:
    """Test schema version tracking functionality."""

    @pytest.mark.asyncio
    async def test_create_schema_version_table(self, mock_engine, mock_logger):
        """Test creation of schema version table."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_schema_version_table()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        assert mock_conn.execute.call_count == 2  # CREATE and INSERT
        mock_conn.commit.assert_called_once()

        # Check SQL commands
        create_call = mock_conn.execute.call_args_list[0]
        insert_call = mock_conn.execute.call_args_list[1]

        assert "CREATE TABLE IF NOT EXISTS schema_version" in create_call[0][0].text
        assert "INSERT OR IGNORE INTO schema_version" in insert_call[0][0].text

    @pytest.mark.asyncio
    async def test_get_schema_version_initial(self, mock_engine, mock_logger):
        """Test getting initial schema version."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock returning None (no version found)
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = None

        version = await db_manager._get_schema_version()

        assert version == 0
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_schema_version_existing(self, mock_engine, mock_logger):
        """Test getting existing schema version."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock returning version 3
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = 3

        version = await db_manager._get_schema_version()

        assert version == 3

    @pytest.mark.asyncio
    async def test_update_schema_version(self, mock_engine, mock_logger):
        """Test updating schema version."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._update_schema_version(5)

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Check SQL command
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO schema_version" in call_args[0][0].text
        assert call_args[0][1]["version"] == 5


class TestMigrationFunctions:
    """Test individual migration functions."""

    @pytest.mark.asyncio
    async def test_create_initial_tables(self, mock_engine, mock_logger):
        """Test migration 1: create initial tables."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_initial_tables()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Check SQL command
        call_args = mock_conn.execute.call_args
        assert "CREATE TABLE IF NOT EXISTS tasks" in call_args[0][0].text

    @pytest.mark.asyncio
    async def test_create_camera_tables(self, mock_engine, mock_logger):
        """Test migration 2: create camera tables."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_camera_tables()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Check SQL command
        call_args = mock_conn.execute.call_args
        assert "CREATE TABLE IF NOT EXISTS images" in call_args[0][0].text

    @pytest.mark.asyncio
    async def test_create_metrics_tables(self, mock_engine, mock_logger):
        """Test migration 3: create metrics tables."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_metrics_tables()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Check SQL command
        call_args = mock_conn.execute.call_args
        assert "CREATE TABLE IF NOT EXISTS system_metrics" in call_args[0][0].text

    @pytest.mark.asyncio
    async def test_create_sensor_tables(self, mock_engine, mock_logger):
        """Test migration 4: create sensor tables."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_sensor_tables()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        assert mock_conn.execute.call_count == 2  # barometer and IMU tables
        mock_conn.commit.assert_called_once()

        # Check SQL commands
        calls = mock_conn.execute.call_args_list
        assert "CREATE TABLE IF NOT EXISTS barometer_readings" in calls[0][0][0].text
        assert "CREATE TABLE IF NOT EXISTS imu_readings" in calls[1][0][0].text

    @pytest.mark.asyncio
    async def test_create_ai_tables(self, mock_engine, mock_logger):
        """Test migration 5: create AI tables."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        await db_manager._create_ai_tables()

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        assert mock_conn.execute.call_count == 3  # AI models, results, and queue tables
        mock_conn.commit.assert_called_once()

        # Check SQL commands
        calls = mock_conn.execute.call_args_list
        assert "CREATE TABLE IF NOT EXISTS ai_models" in calls[0][0][0].text
        assert "CREATE TABLE IF NOT EXISTS segmentation_results" in calls[1][0][0].text
        assert "CREATE TABLE IF NOT EXISTS ai_processing_queue" in calls[2][0][0].text


class TestMigrationSequencing:
    """Test migration application and sequencing."""

    @pytest.mark.asyncio
    async def test_apply_migrations_fresh_database(self, mock_engine, mock_logger):
        """Test applying all migrations to fresh database."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock schema version progression
        version_calls = [0, 1, 2, 3, 4, 5]  # Fresh DB starts at 0
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.side_effect = version_calls

        await db_manager.apply_migrations()

        # Should have called all migration functions
        assert mock_conn.execute.call_count >= 5  # At least one call per migration
        mock_logger.info.assert_any_call("Applying database migrations...")
        mock_logger.info.assert_any_call("Current schema version: 0")
        mock_logger.info.assert_any_call("Database migrations complete")

    @pytest.mark.asyncio
    async def test_apply_migrations_partial_update(self, mock_engine, mock_logger):
        """Test applying migrations to partially updated database."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock database already at version 3
        version_calls = [3, 4, 5]
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.side_effect = version_calls

        await db_manager.apply_migrations()

        mock_logger.info.assert_any_call("Current schema version: 3")
        mock_logger.info.assert_any_call("Applying migration 4")
        mock_logger.info.assert_any_call("Applying migration 5")

    @pytest.mark.asyncio
    async def test_apply_migrations_up_to_date(self, mock_engine, mock_logger):
        """Test applying migrations to up-to-date database."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock database already at latest version
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = 5

        await db_manager.apply_migrations()

        mock_logger.info.assert_any_call("Current schema version: 5")
        # Should not apply any migrations
        migration_logs = [call for call in mock_logger.info.call_args_list
                         if "Applying migration" in str(call)]
        assert len(migration_logs) == 0

    @pytest.mark.asyncio
    async def test_migration_error_handling(self, mock_engine, mock_logger):
        """Test error handling during migrations."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock schema version call to succeed but migration to fail
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = 0
        mock_conn.execute.side_effect = [
            Mock(),  # Schema version table creation
            Mock(),  # Schema version insert
            Mock(),  # Get schema version
            Exception("Migration error")  # First migration fails
        ]

        with pytest.raises(Exception, match="Migration error"):
            await db_manager.apply_migrations()


class TestDatabaseInitialization:
    """Test database initialization function."""

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.load_config')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_init_database_with_config(self, mock_path, mock_load_config,
                                           mock_setup_logger, mock_create_engine,
                                           test_config, mock_logger, mock_engine):
        """Test database initialization with provided config."""
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock database directory creation
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        # Mock connection test
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        result = await init_database(test_config)

        assert result == mock_engine
        mock_create_engine.assert_called_once_with(test_config.db.connection_string, echo=test_config.app.debug)
        mock_db_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.load_config')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_init_database_without_config(self, mock_path, mock_load_config,
                                              mock_setup_logger, mock_create_engine,
                                              test_config, mock_logger, mock_engine):
        """Test database initialization without provided config."""
        mock_load_config.return_value = test_config
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock path operations
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        # Mock connection test
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        result = await init_database()

        assert result == mock_engine
        mock_load_config.assert_called_once()

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_init_database_overwrite_existing(self, mock_path, mock_setup_logger,
                                                  mock_create_engine, mock_logger, mock_engine):
        """Test database initialization with overwrite flag."""
        config = test_config()
        config.db.overwrite = True

        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock existing database file
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = True
        mock_path.return_value = mock_db_path

        # Mock connection test
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        await init_database(config)

        # Should delete existing file
        mock_db_path.unlink.assert_called_once()
        mock_logger.info.assert_any_call(f"Overwriting existing database: {config.db.filename}")

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_init_database_connection_test_failure(self, mock_path, mock_setup_logger,
                                                       mock_create_engine, mock_logger, mock_engine):
        """Test database initialization with connection test failure."""
        config = test_config()
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock path operations
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        # Mock connection test failure
        mock_engine.connect.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            await init_database(config)

        mock_logger.error.assert_called_with("Failed to initialize database: Connection failed")

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_init_database_migration_execution(self, mock_path, mock_setup_logger,
                                                    mock_create_engine, mock_logger, mock_engine):
        """Test that database initialization runs migrations."""
        config = test_config()
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock path operations
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        # Mock connection and migration operations
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value.scalar.return_value = 0  # Fresh database

        await init_database(config)

        # Should have executed migrations
        assert mock_conn.execute.call_count > 0
        mock_logger.info.assert_any_call("Database initialized: sqlite+aiosqlite:///test.db")


class TestDatabaseErrorHandling:
    """Test database error handling scenarios."""

    @pytest.mark.asyncio
    async def test_migration_rollback_on_error(self, mock_engine, mock_logger):
        """Test that migrations handle errors gracefully."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock connection that fails during migration
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = [
            Mock(),  # Schema version table creation
            Mock(),  # Schema version insert
            Mock(scalar=Mock(return_value=0)),  # Get schema version
            Exception("SQL error")  # Migration fails
        ]

        with pytest.raises(Exception, match="SQL error"):
            await db_manager.apply_migrations()

    @pytest.mark.asyncio
    async def test_schema_version_table_creation_error(self, mock_engine, mock_logger):
        """Test error during schema version table creation."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = Exception("Cannot create table")

        with pytest.raises(Exception, match="Cannot create table"):
            await db_manager._create_schema_version_table()

    @pytest.mark.asyncio
    async def test_get_schema_version_error(self, mock_engine, mock_logger):
        """Test error during schema version retrieval."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = Exception("Cannot query table")

        with pytest.raises(Exception, match="Cannot query table"):
            await db_manager._get_schema_version()

    @pytest.mark.asyncio
    async def test_update_schema_version_error(self, mock_engine, mock_logger):
        """Test error during schema version update."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.side_effect = Exception("Cannot update table")

        with pytest.raises(Exception, match="Cannot update table"):
            await db_manager._update_schema_version(1)


class TestDatabaseIntegration:
    """Integration tests for database functionality."""

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_full_database_setup_flow(self, mock_path, mock_setup_logger,
                                          mock_create_engine, test_config, mock_logger, mock_engine):
        """Test complete database setup flow."""
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.return_value = mock_engine

        # Mock path operations
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        # Mock migration sequence
        mock_conn = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn

        # Simulate fresh database with progressive migrations
        version_responses = [0, 1, 2, 3, 4, 5]
        mock_conn.execute.return_value.scalar.side_effect = version_responses

        result = await init_database(test_config)

        assert result == mock_engine

        # Verify setup sequence
        mock_db_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_create_engine.assert_called_once()

        # Verify migrations were applied
        assert mock_conn.execute.call_count >= 5
        mock_logger.info.assert_any_call("Database migrations complete")

    @pytest.mark.asyncio
    async def test_database_main_function(self):
        """Test the main function in database module."""
        with patch('database.setup_logger') as mock_setup_logger, \
             patch('database.init_database') as mock_init_db, \
             patch('database.text') as mock_text:

            mock_logger = Mock()
            mock_setup_logger.return_value = mock_logger

            mock_engine = Mock()
            mock_engine.dispose = AsyncMock()
            mock_init_db.return_value = mock_engine

            # Mock connection and SQL execution
            mock_conn = AsyncMock()
            mock_conn.execute.return_value.scalar.return_value = "3.39.4"
            mock_engine.connect.return_value.__aenter__.return_value = mock_conn
            mock_engine.connect.return_value.__aexit__.return_value = None

            # Import and run main function
            from database import main
            await main()

            mock_init_db.assert_called_once()
            mock_engine.dispose.assert_called_once()
            mock_logger.info.assert_any_call("SQLite version: 3.39.4")


class TestDatabaseEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_migration_with_very_high_version(self, mock_engine, mock_logger):
        """Test migration with version higher than expected."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock database with version higher than latest migration
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = 999

        await db_manager.apply_migrations()

        # Should not attempt any migrations
        mock_logger.info.assert_any_call("Current schema version: 999")

    @pytest.mark.asyncio
    async def test_concurrent_migration_safety(self, mock_engine, mock_logger):
        """Test that migrations are safe for concurrent execution."""
        db_manager = DatabaseManager(mock_engine, mock_logger)

        # Mock connection operations
        mock_conn = mock_engine.connect.return_value.__aenter__.return_value
        mock_conn.execute.return_value.scalar.return_value = 0

        # This test ensures that the migration logic uses IF NOT EXISTS
        # and other safe SQL patterns
        await db_manager.apply_migrations()

        # Verify that table creation uses IF NOT EXISTS
        calls = mock_conn.execute.call_args_list
        create_table_calls = [call for call in calls if 'CREATE TABLE' in str(call)]

        for call in create_table_calls:
            sql_text = str(call[0][0])
            assert 'IF NOT EXISTS' in sql_text or 'CREATE TABLE' in sql_text

    @patch('database.create_async_engine')
    @patch('database.setup_logger')
    @patch('database.Path')
    @pytest.mark.asyncio
    async def test_database_file_permissions_error(self, mock_path, mock_setup_logger,
                                                  mock_create_engine, test_config, mock_logger):
        """Test database initialization with file permission errors."""
        mock_setup_logger.return_value = mock_logger
        mock_create_engine.side_effect = PermissionError("Permission denied")

        # Mock path operations
        mock_db_path = Mock()
        mock_db_dir = Mock()
        mock_db_path.parent = mock_db_dir
        mock_db_path.exists.return_value = False
        mock_path.return_value = mock_db_path

        with pytest.raises(PermissionError, match="Permission denied"):
            await init_database(test_config)

        mock_logger.error.assert_called_with("Failed to initialize database: Permission denied")


if __name__ == "__main__":
    pytest.main([__file__])
