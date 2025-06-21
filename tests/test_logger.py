import pytest
import sys
import os
import tempfile
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logger import setup_logger
from config import MachaConfig, LogColors, LogFileConfig, LogConsoleConfig, LoggingConfig


@pytest.fixture
def test_config():
    """Create a test configuration with logging settings."""
    return MachaConfig(
        app={"name": "test_app", "debug": True},
        logging={
            "level": "INFO",
            "file": {
                "path": "test_logs/test.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "console": {
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white"
                }
            }
        },
        db={
            "filename": "test.db",
            "connection_string": "sqlite:///test.db",
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


@pytest.fixture
def minimal_config():
    """Create a minimal test configuration."""
    return MachaConfig(
        app={"name": "minimal_app", "debug": False},
        logging={
            "level": "DEBUG",
            "file": {"path": "minimal.log"}
        },
        db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
        tasks=[{"name": "task", "class": "MetricsTask", "frequency": 60, "enabled": True}]
    )


class TestSetupLoggerBasic:
    """Test basic logger setup functionality."""

    def test_setup_logger_with_config(self, test_config):
        """Test logger setup with provided configuration."""
        logger = setup_logger(test_config)

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_app"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # Console and file handlers

    def test_setup_logger_name_from_config(self, test_config):
        """Test that logger uses app name from config."""
        test_config.app.name = "custom_app_name"
        logger = setup_logger(test_config)

        assert logger.name == "custom_app_name"

    def test_setup_logger_level_from_config(self, test_config):
        """Test that logger level is set from config."""
        test_config.logging.level = "WARNING"
        logger = setup_logger(test_config)

        assert logger.level == logging.WARNING

    def test_setup_logger_different_levels(self, test_config):
        """Test logger setup with different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        expected_levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

        for level_str, expected_level in zip(levels, expected_levels):
            test_config.logging.level = level_str
            logger = setup_logger(test_config)
            assert logger.level == expected_level

    @patch('logger.load_config')
    def test_setup_logger_without_config(self, mock_load_config, test_config):
        """Test logger setup without providing config (loads default)."""
        mock_load_config.return_value = test_config

        logger = setup_logger()

        mock_load_config.assert_called_once()
        assert logger.name == "test_app"
        assert logger.level == logging.INFO

    def test_setup_logger_minimal_config(self, minimal_config):
        """Test logger setup with minimal configuration."""
        logger = setup_logger(minimal_config)

        assert logger.name == "minimal_app"
        assert logger.level == logging.DEBUG


class TestLoggerHandlers:
    """Test logger handler setup and configuration."""

    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    def test_handler_types_created(self, mock_stream_handler, mock_file_handler, test_config):
        """Test that both console and file handlers are created."""
        mock_console_handler = Mock()
        mock_file_handler_instance = Mock()
        mock_stream_handler.return_value = mock_console_handler
        mock_file_handler.return_value = mock_file_handler_instance

        with patch('pathlib.Path.mkdir'):
            logger = setup_logger(test_config)

            mock_stream_handler.assert_called_once()
            mock_file_handler.assert_called_once_with(test_config.logging.file.path)

    def test_handlers_cleared_before_setup(self, test_config):
        """Test that existing handlers are cleared to avoid duplicates."""
        # Create logger with initial handler
        logger = logging.getLogger(test_config.app.name)
        initial_handler = logging.StreamHandler()
        logger.addHandler(initial_handler)
        initial_count = len(logger.handlers)

        # Setup logger again
        setup_logger(test_config)

        # Should have exactly 2 handlers (console + file), not more
        assert len(logger.handlers) == 2

    def test_multiple_setup_calls_no_duplicate_handlers(self, test_config):
        """Test that multiple setup calls don't create duplicate handlers."""
        logger1 = setup_logger(test_config)
        handler_count_1 = len(logger1.handlers)

        logger2 = setup_logger(test_config)
        handler_count_2 = len(logger2.handlers)

        # Should be the same logger instance with same number of handlers
        assert logger1 is logger2
        assert handler_count_1 == handler_count_2 == 2

    @patch('pathlib.Path.mkdir')
    def test_log_directory_creation(self, mock_mkdir, test_config):
        """Test that log directory is created if it doesn't exist."""
        setup_logger(test_config)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    def test_log_directory_creation_error_handling(self, mock_mkdir, test_config):
        """Test handling of log directory creation errors."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        # Should not raise exception, but might log error
        logger = setup_logger(test_config)
        assert logger is not None


class TestConsoleHandler:
    """Test console handler setup and formatting."""

    @patch('logging.StreamHandler')
    @patch('colorlog.ColoredFormatter')
    def test_console_handler_formatter_setup(self, mock_colored_formatter, mock_stream_handler, test_config):
        """Test console handler formatter setup."""
        mock_console_handler = Mock()
        mock_formatter = Mock()
        mock_stream_handler.return_value = mock_console_handler
        mock_colored_formatter.return_value = mock_formatter

        with patch('pathlib.Path.mkdir'):
            setup_logger(test_config)

            mock_colored_formatter.assert_called_once()
            mock_console_handler.setFormatter.assert_called_once_with(mock_formatter)

    @patch('logging.StreamHandler')
    @patch('colorlog.ColoredFormatter')
    def test_console_formatter_format_string(self, mock_colored_formatter, mock_stream_handler, test_config):
        """Test that console formatter uses correct format string."""
        mock_console_handler = Mock()
        mock_stream_handler.return_value = mock_console_handler

        with patch('pathlib.Path.mkdir'):
            setup_logger(test_config)

            call_args = mock_colored_formatter.call_args
            format_string = call_args[0][0]
            assert format_string == test_config.logging.console.format

    @patch('logging.StreamHandler')
    @patch('colorlog.ColoredFormatter')
    def test_console_formatter_colors(self, mock_colored_formatter, mock_stream_handler, test_config):
        """Test that console formatter uses correct colors."""
        mock_console_handler = Mock()
        mock_stream_handler.return_value = mock_console_handler

        with patch('pathlib.Path.mkdir'):
            setup_logger(test_config)

            call_args = mock_colored_formatter.call_args
            log_colors = call_args[1]['log_colors']
            expected_colors = test_config.logging.console.colors.model_dump()
            assert log_colors == expected_colors

    def test_console_handler_default_colors(self, minimal_config):
        """Test console handler with default color configuration."""
        # Remove console config to use defaults
        minimal_config.logging.console = LogConsoleConfig()

        logger = setup_logger(minimal_config)

        # Should not raise exception and should have console handler
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) == 1


class TestFileHandler:
    """Test file handler setup and formatting."""

    @patch('logging.FileHandler')
    @patch('logging.Formatter')
    def test_file_handler_formatter_setup(self, mock_formatter_class, mock_file_handler, test_config):
        """Test file handler formatter setup."""
        mock_file_handler_instance = Mock()
        mock_formatter = Mock()
        mock_file_handler.return_value = mock_file_handler_instance
        mock_formatter_class.return_value = mock_formatter

        with patch('pathlib.Path.mkdir'):
            setup_logger(test_config)

            mock_formatter_class.assert_called_once_with(test_config.logging.file.format)
            mock_file_handler_instance.setFormatter.assert_called_once_with(mock_formatter)

    @patch('logging.FileHandler')
    def test_file_handler_path(self, mock_file_handler, test_config):
        """Test that file handler uses correct file path."""
        with patch('pathlib.Path.mkdir'):
            setup_logger(test_config)

            mock_file_handler.assert_called_once_with(test_config.logging.file.path)

    def test_file_handler_different_paths(self, test_config):
        """Test file handler with different file paths."""
        test_paths = [
            "logs/app.log",
            "/tmp/test.log",
            "nested/deep/path/log.txt"
        ]

        for path in test_paths:
            test_config.logging.file.path = path
            with patch('logging.FileHandler') as mock_file_handler, \
                 patch('pathlib.Path.mkdir'):
                setup_logger(test_config)
                mock_file_handler.assert_called_with(path)

    def test_file_handler_default_format(self):
        """Test file handler with default format."""
        config = MachaConfig(
            app={"name": "test", "debug": False},
            logging={"level": "INFO", "file": {"path": "test.log"}},
            db={"filename": "test.db", "connection_string": "sqlite:///test.db", "overwrite": False},
            tasks=[{"name": "task", "class": "MetricsTask", "frequency": 60, "enabled": True}]
        )

        logger = setup_logger(config)

        # Should not raise exception
        assert logger is not None
        assert len(logger.handlers) == 2


class TestLoggerFormatting:
    """Test logger formatting configuration."""

    def test_custom_console_format(self, test_config):
        """Test custom console format string."""
        custom_format = "CUSTOM: %(log_color)s%(levelname)s - %(message)s"
        test_config.logging.console.format = custom_format

        with patch('colorlog.ColoredFormatter') as mock_formatter:
            setup_logger(test_config)
            mock_formatter.assert_called_once_with(custom_format, log_colors=unittest.mock.ANY)

    def test_custom_file_format(self, test_config):
        """Test custom file format string."""
        custom_format = "FILE: %(asctime)s | %(levelname)s | %(message)s"
        test_config.logging.file.format = custom_format

        with patch('logging.Formatter') as mock_formatter:
            setup_logger(test_config)
            mock_formatter.assert_called_once_with(custom_format)

    def test_color_configuration_variations(self, test_config):
        """Test different color configurations."""
        color_configs = [
            {
                "DEBUG": "blue",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_yellow"
            },
            {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "white,bg_red"
            }
        ]

        for colors in color_configs:
            test_config.logging.console.colors = LogColors(**colors)
            with patch('colorlog.ColoredFormatter') as mock_formatter:
                setup_logger(test_config)
                call_args = mock_formatter.call_args
                assert call_args[1]['log_colors'] == colors


class TestLoggerErrorHandling:
    """Test error handling in logger setup."""

    @patch('logging.FileHandler')
    def test_file_handler_creation_error(self, mock_file_handler, test_config):
        """Test handling of file handler creation errors."""
        mock_file_handler.side_effect = PermissionError("Cannot create file")

        # Should raise the exception since file handler is required
        with pytest.raises(PermissionError):
            setup_logger(test_config)

    @patch('logging.StreamHandler')
    def test_console_handler_creation_error(self, mock_stream_handler, test_config):
        """Test handling of console handler creation errors."""
        mock_stream_handler.side_effect = Exception("Console error")

        # Should raise the exception since console handler is required
        with pytest.raises(Exception, match="Console error"):
            setup_logger(test_config)

    @patch('colorlog.ColoredFormatter')
    def test_colored_formatter_error(self, mock_colored_formatter, test_config):
        """Test handling of colored formatter creation errors."""
        mock_colored_formatter.side_effect = ImportError("colorlog not available")

        with pytest.raises(ImportError):
            setup_logger(test_config)

    def test_invalid_log_level(self, test_config):
        """Test handling of invalid log level."""
        test_config.logging.level = "INVALID_LEVEL"

        # Should raise ValueError for invalid log level
        with pytest.raises(ValueError):
            setup_logger(test_config)

    @patch('pathlib.Path.mkdir')
    def test_log_directory_permission_error(self, mock_mkdir, test_config):
        """Test handling of log directory creation permission errors."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        # Should raise exception when cannot create log directory
        with pytest.raises(PermissionError):
            setup_logger(test_config)


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_logger_actually_logs(self, test_config):
        """Test that the logger actually logs messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            test_config.logging.file.path = log_file

            logger = setup_logger(test_config)

            # Log a test message
            test_message = "Test log message"
            logger.info(test_message)

            # Check that message was written to file
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert test_message in log_content
                assert "INFO" in log_content

    def test_logger_respects_level(self, test_config):
        """Test that logger respects the configured level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            test_config.logging.file.path = log_file
            test_config.logging.level = "WARNING"

            logger = setup_logger(test_config)

            # Log messages at different levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

            # Check file content
            with open(log_file, 'r') as f:
                log_content = f.read()
                assert "Debug message" not in log_content
                assert "Info message" not in log_content
                assert "Warning message" in log_content
                assert "Error message" in log_content

    def test_multiple_loggers_same_name(self, test_config):
        """Test that multiple setup calls return the same logger instance."""
        logger1 = setup_logger(test_config)
        logger2 = setup_logger(test_config)

        assert logger1 is logger2
        assert id(logger1) == id(logger2)

    def test_logger_with_complex_path(self, test_config):
        """Test logger with complex nested path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            complex_path = os.path.join(temp_dir, "nested", "deep", "path", "app.log")
            test_config.logging.file.path = complex_path

            logger = setup_logger(test_config)

            # Should create the nested directory structure
            assert os.path.exists(os.path.dirname(complex_path))

            # Should be able to log to the file
            logger.info("Test message")
            assert os.path.exists(complex_path)


class TestLoggerConfiguration:
    """Test various logger configuration scenarios."""

    def test_debug_mode_effect(self, test_config):
        """Test that debug mode doesn't affect basic logger functionality."""
        test_config.app.debug = True
        logger1 = setup_logger(test_config)

        test_config.app.debug = False
        logger2 = setup_logger(test_config)

        # Both should work regardless of debug mode
        assert logger1.name == logger2.name
        assert len(logger1.handlers) == len(logger2.handlers) == 2

    def test_log_colors_model_conversion(self, test_config):
        """Test that log colors are properly converted from Pydantic model."""
        logger = setup_logger(test_config)

        # Should not raise any errors during color processing
        assert logger is not None

        # Colors should be accessible from the config
        colors = test_config.logging.console.colors
        assert hasattr(colors, 'DEBUG')
        assert hasattr(colors, 'INFO')
        assert hasattr(colors, 'WARNING')
        assert hasattr(colors, 'ERROR')
        assert hasattr(colors, 'CRITICAL')

    def test_default_log_colors(self):
        """Test default log color configuration."""
        default_colors = LogColors()

        assert default_colors.DEBUG == "cyan"
        assert default_colors.INFO == "green"
        assert default_colors.WARNING == "yellow"
        assert default_colors.ERROR == "red"
        assert default_colors.CRITICAL == "red,bg_white"

    def test_config_inheritance(self, test_config):
        """Test that logger configuration inherits properly."""
        # Modify parent config
        original_level = test_config.logging.level
        test_config.logging.level = "ERROR"

        logger = setup_logger(test_config)
        assert logger.level == logging.ERROR

        # Restore original
        test_config.logging.level = original_level


if __name__ == "__main__":
    pytest.main([__file__])
