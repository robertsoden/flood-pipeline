"""
Standardized logging configuration for the pipeline.

Provides consistent logging setup across all stages with both file and console output.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    stage: str,
    project_root: Path,
    level: int = logging.INFO,
    console_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a standardized logger for a pipeline stage.

    Args:
        name: Logger name (usually __name__ from calling module)
        stage: Stage identifier (e.g., 'stage1_train', 'stage2_optimize')
        project_root: Project root directory path
        level: Logging level for file output (default: INFO)
        console_level: Logging level for console output (default: same as level)

    Returns:
        Configured logger instance

    Example:
        >>> from shared.config import PROJECT_ROOT
        >>> logger = setup_logger(__name__, 'stage2_optimize', PROJECT_ROOT)
        >>> logger.info("Processing started")
    """
    if console_level is None:
        console_level = level

    # Create logs directory
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f'{stage}_{timestamp}.log'

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - cleaner output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log the log file location
    logger.info(f"Logging to: {log_file}")

    return logger


def log_section(logger: logging.Logger, title: str, width: int = 70) -> None:
    """
    Log a section header for better readability.

    Args:
        logger: Logger instance
        title: Section title
        width: Width of the separator line

    Example:
        >>> log_section(logger, "STAGE 2: MODEL OPTIMIZATION")
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration") -> None:
    """
    Log configuration settings in a readable format.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the config section

    Example:
        >>> log_config(logger, STAGE1_CONFIG, "Stage 1 Configuration")
    """
    logger.info(f"\n{title}:")
    for key, value in config.items():
        # Format Path objects nicely
        if isinstance(value, Path):
            value = str(value)
        logger.info(f"  {key}: {value}")


def log_stats(logger: logging.Logger, stats: dict, title: str = "Statistics") -> None:
    """
    Log statistics in a readable format.

    Args:
        logger: Logger instance
        stats: Statistics dictionary
        title: Title for the stats section

    Example:
        >>> log_stats(logger, {'total': 1000, 'valid': 950, 'invalid': 50})
    """
    logger.info(f"\n{title}:")
    for key, value in stats.items():
        # Format numbers with commas if they're large
        if isinstance(value, int) and value >= 1000:
            logger.info(f"  {key}: {value:,}")
        elif isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
