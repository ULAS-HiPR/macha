import logging
import colorlog
from pathlib import Path
from config import MachaConfig


def setup_logger(config: MachaConfig = None) -> logging.Logger:
    """Set up a logger with file and console handlers based on config."""
    if config is None:
        from config import load_config

        config = load_config()

    logger = logging.getLogger(config.app.name)
    logger.setLevel(config.logging.level.value)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_format = colorlog.ColoredFormatter(
        config.logging.console.format, log_colors=config.logging.console.colors.dict()
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path(config.logging.file.path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.logging.file.path)
    file_format = logging.Formatter(config.logging.file.format)
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
