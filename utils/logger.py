"""
Logging utilities for distributed training.

Provides helper functions for setting up consistent logging across
single GPU and multi-GPU distributed training environments.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    rank: int = 0,
    name: Optional[str] = None,
    level: int = logging.INFO,
    use_console: bool = True
) -> logging.Logger:
    """
    Setup logger with rank info for distributed training.

    In distributed training, only rank 0 logs at INFO level by default,
    while other ranks log at WARNING level to reduce output clutter.

    Args:
        rank: Process rank (0 for main process, >0 for workers).
              In non-distributed mode, use default 0.
        name: Logger name. If None, uses the calling module's name.
        level: Default logging level for rank 0.
        use_console: Whether to add console handler.

    Returns:
        Configured logger instance.

    Example:
        >>> # Single GPU / main process
        >>> logger = setup_logger()
        >>> logger.info("Starting training...")

        >>> # Distributed training
        >>> from model.distributed import get_rank
        >>> logger = setup_logger(rank=get_rank())
        >>> logger.info("This only prints from rank 0")

        >>> # Custom name
        >>> logger = setup_logger(rank=0, name="my_module")
    """
    # Determine effective level based on rank
    effective_level = level if rank == 0 else logging.WARNING

    # Get logger
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(effective_level)

    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    if use_console:
        # Create console handler with rank-specific format including filename and line number
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(effective_level)

        # Format includes rank info, filename and line number for distributed training
        if rank > 0:
            formatter = logging.Formatter(
                f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_file_logger(
    log_file: str,
    rank: int = 0,
    name: Optional[str] = None,
    level: int = logging.INFO,
    mode: str = 'a',
    use_console: bool = True
) -> logging.Logger:
    """Setup logger with both console and file output.

    This function configures a logger to output to both the console (stdout)
    and a log file simultaneously. Log format includes filename and line number.

    Args:
        log_file: Path to log file.
        rank: Process rank. Only rank 0 writes to file to avoid duplicates.
        name: Logger name.
        level: Logging level.
        mode: File open mode ('a' for append, 'w' for overwrite).
        use_console: Whether to also output to console.

    Returns:
        Configured logger instance with both console and file handlers.
    """
    # Determine effective level based on rank
    effective_level = level if rank == 0 else logging.WARNING

    # Get logger
    logger_name = name or __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(effective_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Log format with filename and line number
    console_format = (
        f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        if rank > 0 else
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

    # Console handler
    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(logging.Formatter(console_format))
        logger.addHandler(console_handler)

    # File handler (only for rank 0 to avoid multiple writes)
    if rank == 0:
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create a new one.

    This is a convenience function that returns a logger
    without setting up handlers. Useful for modules that
    just want to log without configuring.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
