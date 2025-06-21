import pytest
import asyncio
import sys
import os
import signal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import Application, main


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock()
    config.tasks = []
    config.app = Mock()
    config.app.name = "test_app"
    return config


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
def mock_engine():
    """Create a mock database engine."""
    engine = Mock()
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def mock_scheduler():
    """Create a mock task scheduler."""
    scheduler = Mock()
    scheduler.start = AsyncMock()
    scheduler.stop = AsyncMock()
    return scheduler


class TestApplication:
    """Test cases for the Application class."""

    def test_application_initialization(self):
        """Test Application initialization."""
        app = Application()

        assert app.scheduler is None
        assert app.engine is None
        assert app.logger is None
        assert app.config is None
        assert hasattr(app, 'shutdown_event')
        assert isinstance(app.shutdown_event, asyncio.Event)

    @patch('main.load_config')
    @patch('main.setup_logger')
    @patch('main.init_database')
    @patch('main.TaskScheduler')
    @pytest.mark.asyncio
    async def test_startup_successful(self, mock_task_scheduler, mock_init_db,
                                    mock_setup_logger, mock_load_config,
                                    mock_config, mock_logger, mock_engine, mock_scheduler):
        """Test successful application startup."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_setup_logger.return_value = mock_logger
        mock_init_db.return_value = mock_engine
        mock_task_scheduler.return_value = mock_scheduler

        app = Application()
        await app.startup()

        # Verify initialization sequence
        mock_load_config.assert_called_once()
        mock_setup_logger.assert_called_once_with(mock_config)
        mock_init_db.assert_called_once_with(mock_config)
        mock_task_scheduler.assert_called_once_with(mock_config, mock_engine, mock_logger)

        # Verify app state
        assert app.config == mock_config
        assert app.logger == mock_logger
        assert app.engine == mock_engine
        assert app.scheduler == mock_scheduler

        # Verify logging calls
        mock_logger.info.assert_any_call("Starting Macha application")
        mock_logger.info.assert_any_call("Configuration loaded with 0 tasks")
        mock_logger.info.assert_any_call("Database initialized")
        mock_logger.info.assert_any_call("Task scheduler initialized")

    @patch('main.load_config')
    @pytest.mark.asyncio
    async def test_startup_config_load_failure(self, mock_load_config):
        """Test startup failure during config loading."""
        mock_load_config.side_effect = ValueError("Invalid config")

        app = Application()

        with pytest.raises(ValueError, match="Invalid config"):
            await app.startup()

    @patch('main.load_config')
    @patch('main.setup_logger')
    @patch('main.init_database')
    @pytest.mark.asyncio
    async def test_startup_database_failure(self, mock_init_db, mock_setup_logger,
                                          mock_load_config, mock_config, mock_logger):
        """Test startup failure during database initialization."""
        mock_load_config.return_value = mock_config
        mock_setup_logger.return_value = mock_logger
        mock_init_db.side_effect = Exception("Database error")

        app = Application()

        with pytest.raises(Exception, match="Database error"):
            await app.startup()

        mock_logger.error.assert_called_with("Application startup failed: Database error")

    @patch('main.load_config')
    @patch('main.setup_logger')
    @pytest.mark.asyncio
    async def test_startup_logger_not_available(self, mock_setup_logger, mock_load_config):
        """Test startup failure when logger setup fails."""
        mock_load_config.side_effect = Exception("Config error")

        app = Application()

        with pytest.raises(Exception, match="Config error"):
            await app.startup()

        # Should not call setup_logger since config loading failed first
        mock_setup_logger.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scheduler_start(self, mock_scheduler):
        """Test application run method starts scheduler."""
        app = Application()
        app.scheduler = mock_scheduler

        await app.run()

        mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt(self, mock_logger):
        """Test application run handles KeyboardInterrupt."""
        mock_scheduler = Mock()
        mock_scheduler.start = AsyncMock(side_effect=KeyboardInterrupt())

        app = Application()
        app.scheduler = mock_scheduler
        app.logger = mock_logger

        await app.run()

        mock_logger.info.assert_called_with("Received keyboard interrupt")

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, mock_logger):
        """Test application run handles general exceptions."""
        mock_scheduler = Mock()
        mock_scheduler.start = AsyncMock(side_effect=Exception("Scheduler error"))

        app = Application()
        app.scheduler = mock_scheduler
        app.logger = mock_logger

        with pytest.raises(Exception, match="Scheduler error"):
            await app.run()

        mock_logger.error.assert_called_with("Application error: Scheduler error")

    @pytest.mark.asyncio
    async def test_shutdown_complete_cleanup(self, mock_logger, mock_engine, mock_scheduler):
        """Test complete shutdown cleanup."""
        app = Application()
        app.logger = mock_logger
        app.engine = mock_engine
        app.scheduler = mock_scheduler

        await app.shutdown()

        mock_logger.info.assert_any_call("Shutting down application")
        mock_scheduler.stop.assert_called_once()
        mock_engine.dispose.assert_called_once()
        mock_logger.info.assert_any_call("Application shutdown complete")

    @pytest.mark.asyncio
    async def test_shutdown_partial_initialization(self, mock_logger):
        """Test shutdown with partial initialization."""
        app = Application()
        app.logger = mock_logger
        # engine and scheduler are None

        await app.shutdown()

        mock_logger.info.assert_any_call("Shutting down application")
        mock_logger.info.assert_any_call("Application shutdown complete")

    @pytest.mark.asyncio
    async def test_shutdown_no_logger(self):
        """Test shutdown when logger is not available."""
        app = Application()
        # No logger set

        # Should not raise exception
        await app.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_scheduler_error(self, mock_logger, mock_scheduler):
        """Test shutdown when scheduler stop fails."""
        mock_scheduler.stop.side_effect = Exception("Scheduler stop error")

        app = Application()
        app.logger = mock_logger
        app.scheduler = mock_scheduler

        # Should complete shutdown despite scheduler error
        await app.shutdown()

        mock_logger.info.assert_any_call("Shutting down application")
        mock_logger.info.assert_any_call("Application shutdown complete")

    @pytest.mark.asyncio
    async def test_shutdown_engine_error(self, mock_logger, mock_engine):
        """Test shutdown when engine disposal fails."""
        mock_engine.dispose.side_effect = Exception("Engine disposal error")

        app = Application()
        app.logger = mock_logger
        app.engine = mock_engine

        # Should complete shutdown despite engine error
        await app.shutdown()

        mock_logger.info.assert_any_call("Shutting down application")
        mock_logger.info.assert_any_call("Application shutdown complete")


class TestMainFunction:
    """Test cases for the main function and entry point."""

    @patch('main.Application')
    @pytest.mark.asyncio
    async def test_main_successful_execution(self, mock_application_class):
        """Test successful main function execution."""
        mock_app = Mock()
        mock_app.startup = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.shutdown_event = Mock()
        mock_app.shutdown_event.wait = AsyncMock()

        # Mock scheduler that returns immediately
        mock_scheduler_task = Mock()
        mock_scheduler_task.cancel = Mock()
        mock_app.scheduler = Mock()
        mock_app.scheduler.start = AsyncMock(return_value=mock_scheduler_task)

        mock_application_class.return_value = mock_app

        # Mock asyncio.wait to return shutdown task as done
        with patch('asyncio.create_task') as mock_create_task, \
             patch('asyncio.wait') as mock_wait:

            mock_scheduler_task = Mock()
            mock_shutdown_task = Mock()
            mock_create_task.side_effect = [mock_scheduler_task, mock_shutdown_task]

            # Mock wait to return shutdown task as done (simulating shutdown signal)
            mock_wait.return_value = ([mock_shutdown_task], [mock_scheduler_task])

            result = await main()

            assert result == 0
            mock_app.startup.assert_called_once()
            mock_app.shutdown.assert_called_once()

    @patch('main.Application')
    @pytest.mark.asyncio
    async def test_main_configuration_error(self, mock_application_class):
        """Test main function with configuration error."""
        mock_app = Mock()
        mock_app.startup = AsyncMock(side_effect=ValueError("Config error"))
        mock_app.shutdown = AsyncMock()
        mock_application_class.return_value = mock_app

        result = await main()

        assert result == 1
        mock_app.shutdown.assert_called_once()

    @patch('main.Application')
    @patch('builtins.print')
    @pytest.mark.asyncio
    async def test_main_general_exception(self, mock_print, mock_application_class):
        """Test main function with general exception."""
        mock_app = Mock()
        mock_app.startup = AsyncMock(side_effect=Exception("General error"))
        mock_app.shutdown = AsyncMock()
        mock_app.logger = None
        mock_application_class.return_value = mock_app

        result = await main()

        assert result == 1
        mock_print.assert_any_call("Application failed to start: General error")
        mock_app.shutdown.assert_called_once()

    @patch('main.Application')
    @pytest.mark.asyncio
    async def test_main_exception_with_logger(self, mock_application_class):
        """Test main function exception handling with logger available."""
        mock_logger = Mock()
        mock_app = Mock()
        mock_app.startup = AsyncMock(side_effect=Exception("Startup error"))
        mock_app.shutdown = AsyncMock()
        mock_app.logger = mock_logger
        mock_application_class.return_value = mock_app

        result = await main()

        assert result == 1
        mock_logger.error.assert_called_with("Application failed: Startup error")

    @patch('main.Application')
    @patch('asyncio.create_task')
    @patch('asyncio.wait')
    @pytest.mark.asyncio
    async def test_main_scheduler_completes_first(self, mock_wait, mock_create_task, mock_application_class):
        """Test main when scheduler task completes before shutdown signal."""
        mock_app = Mock()
        mock_app.startup = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.shutdown_event = Mock()
        mock_app.shutdown_event.wait = AsyncMock()
        mock_app.scheduler = Mock()
        mock_app.scheduler.start = AsyncMock()

        mock_application_class.return_value = mock_app

        mock_scheduler_task = Mock()
        mock_shutdown_task = Mock()
        mock_create_task.side_effect = [mock_scheduler_task, mock_shutdown_task]

        # Mock wait to return scheduler task as done first
        mock_wait.return_value = ([mock_scheduler_task], [mock_shutdown_task])

        result = await main()

        assert result == 0
        mock_shutdown_task.cancel.assert_called_once()

    @patch('main.Application')
    @patch('asyncio.create_task')
    @patch('asyncio.wait')
    @pytest.mark.asyncio
    async def test_main_task_cancellation_handling(self, mock_wait, mock_create_task, mock_application_class):
        """Test main function handles task cancellation properly."""
        mock_app = Mock()
        mock_app.startup = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.shutdown_event = Mock()
        mock_app.shutdown_event.wait = AsyncMock()
        mock_app.scheduler = Mock()
        mock_app.scheduler.start = AsyncMock()

        mock_application_class.return_value = mock_app

        mock_scheduler_task = Mock()
        mock_shutdown_task = Mock()
        mock_create_task.side_effect = [mock_scheduler_task, mock_shutdown_task]

        # Mock task cancellation to raise CancelledError
        async def mock_cancelled_task():
            raise asyncio.CancelledError()

        mock_shutdown_task.__await__ = lambda: mock_cancelled_task().__await__()

        # Mock wait to return shutdown task as done
        mock_wait.return_value = ([mock_shutdown_task], [mock_scheduler_task])

        result = await main()

        assert result == 0
        mock_scheduler_task.cancel.assert_called_once()


class TestSignalHandling:
    """Test signal handling functionality."""

    @patch('signal.signal')
    @patch('main.Application')
    @pytest.mark.asyncio
    async def test_signal_handlers_registration(self, mock_application_class, mock_signal):
        """Test that signal handlers are registered correctly."""
        mock_app = Mock()
        mock_app.startup = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.shutdown_event = Mock()
        mock_app.shutdown_event.wait = AsyncMock()
        mock_app.shutdown_event.set = Mock()
        mock_app.scheduler = Mock()
        mock_app.scheduler.start = AsyncMock()

        mock_application_class.return_value = mock_app

        # Mock asyncio operations to exit quickly
        with patch('asyncio.create_task') as mock_create_task, \
             patch('asyncio.wait') as mock_wait:

            mock_scheduler_task = Mock()
            mock_shutdown_task = Mock()
            mock_create_task.side_effect = [mock_scheduler_task, mock_shutdown_task]
            mock_wait.return_value = ([mock_shutdown_task], [mock_scheduler_task])

            await main()

            # Verify signal handlers were registered
            mock_signal.assert_any_call(signal.SIGTERM, unittest.mock.ANY)
            mock_signal.assert_any_call(signal.SIGINT, unittest.mock.ANY)

    @patch('main.Application')
    @patch('builtins.print')
    @pytest.mark.asyncio
    async def test_signal_handler_execution(self, mock_print, mock_application_class):
        """Test signal handler execution."""
        mock_app = Mock()
        mock_app.startup = AsyncMock()
        mock_app.shutdown = AsyncMock()
        mock_app.shutdown_event = Mock()
        mock_app.shutdown_event.wait = AsyncMock()
        mock_app.shutdown_event.set = Mock()
        mock_app.scheduler = Mock()
        mock_app.scheduler.start = AsyncMock()

        mock_application_class.return_value = mock_app

        # Capture the signal handler
        signal_handler = None

        def capture_signal_handler(sig, handler):
            nonlocal signal_handler
            signal_handler = handler

        with patch('signal.signal', side_effect=capture_signal_handler), \
             patch('asyncio.create_task') as mock_create_task, \
             patch('asyncio.wait') as mock_wait:

            mock_scheduler_task = Mock()
            mock_shutdown_task = Mock()
            mock_create_task.side_effect = [mock_scheduler_task, mock_shutdown_task]
            mock_wait.return_value = ([mock_shutdown_task], [mock_scheduler_task])

            await main()

            # Test signal handler
            assert signal_handler is not None
            signal_handler(signal.SIGTERM, None)

            mock_print.assert_any_call("\nReceived shutdown signal, gracefully shutting down...")
            mock_app.shutdown_event.set.assert_called()


class TestEntryPointExecution:
    """Test the if __name__ == '__main__' entry point."""

    @patch('main.asyncio.run')
    @patch('main.main')
    @patch('builtins.exit')
    def test_entry_point_successful(self, mock_exit, mock_main_func, mock_asyncio_run):
        """Test successful entry point execution."""
        mock_main_func.return_value = 0
        mock_asyncio_run.return_value = 0

        # Mock the entry point execution
        with patch('sys.argv', ['main.py']):
            exec("""
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("Application interrupted by user")
        exit(0)
    except Exception as e:
        print(f"Application failed: {e}")
        exit(1)
""", {'__name__': '__main__', 'asyncio': mock_asyncio_run, 'main': mock_main_func, 'exit': mock_exit, 'print': print})

        mock_asyncio_run.assert_called_once_with(mock_main_func)
        mock_exit.assert_called_once_with(0)

    @patch('main.asyncio.run')
    @patch('main.main')
    @patch('builtins.exit')
    @patch('builtins.print')
    def test_entry_point_keyboard_interrupt(self, mock_print, mock_exit, mock_main_func, mock_asyncio_run):
        """Test entry point with KeyboardInterrupt."""
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        with patch('sys.argv', ['main.py']):
            exec("""
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("Application interrupted by user")
        exit(0)
    except Exception as e:
        print(f"Application failed: {e}")
        exit(1)
""", {'__name__': '__main__', 'asyncio': mock_asyncio_run, 'main': mock_main_func, 'exit': mock_exit, 'print': mock_print})

        mock_print.assert_called_with("Application interrupted by user")
        mock_exit.assert_called_with(0)

    @patch('main.asyncio.run')
    @patch('main.main')
    @patch('builtins.exit')
    @patch('builtins.print')
    def test_entry_point_exception(self, mock_print, mock_exit, mock_main_func, mock_asyncio_run):
        """Test entry point with general exception."""
        mock_asyncio_run.side_effect = Exception("Runtime error")

        with patch('sys.argv', ['main.py']):
            exec("""
if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("Application interrupted by user")
        exit(0)
    except Exception as e:
        print(f"Application failed: {e}")
        exit(1)
""", {'__name__': '__main__', 'asyncio': mock_asyncio_run, 'main': mock_main_func, 'exit': mock_exit, 'print': mock_print})

        mock_print.assert_called_with("Application failed: Runtime error")
        mock_exit.assert_called_with(1)


class TestApplicationIntegration:
    """Integration tests for Application class."""

    @patch('main.load_config')
    @patch('main.setup_logger')
    @patch('main.init_database')
    @patch('main.TaskScheduler')
    @pytest.mark.asyncio
    async def test_full_application_lifecycle(self, mock_task_scheduler, mock_init_db,
                                            mock_setup_logger, mock_load_config,
                                            mock_config, mock_logger, mock_engine, mock_scheduler):
        """Test complete application lifecycle from startup to shutdown."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_setup_logger.return_value = mock_logger
        mock_init_db.return_value = mock_engine
        mock_task_scheduler.return_value = mock_scheduler

        app = Application()

        # Test startup
        await app.startup()
        assert app.config is not None
        assert app.logger is not None
        assert app.engine is not None
        assert app.scheduler is not None

        # Test run (mock scheduler to return immediately)
        mock_scheduler.start = AsyncMock()
        await app.run()
        mock_scheduler.start.assert_called_once()

        # Test shutdown
        await app.shutdown()
        mock_scheduler.stop.assert_called_once()
        mock_engine.dispose.assert_called_once()

    @patch('main.load_config')
    @pytest.mark.asyncio
    async def test_startup_error_recovery(self, mock_load_config):
        """Test that application can handle startup errors gracefully."""
        mock_load_config.side_effect = FileNotFoundError("Config file not found")

        app = Application()

        with pytest.raises(FileNotFoundError):
            await app.startup()

        # Should be able to call shutdown even after startup failure
        await app.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
