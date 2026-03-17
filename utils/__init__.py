"""
Utility functions for CryDet.

This module provides common utilities used across the codebase:
- logger: Logging configuration for distributed training
- config: Configuration dataclasses and utilities
"""

from .logger import setup_logger, setup_file_logger, get_logger
from .config import (
    Config,
    FeatureConfig,
    DatasetConfig,
    AugmentationConfig,
    NoiseConfig,
    MixupConfig,
    LossConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    save_config,
    get_default_config
)

__all__ = [
    # Logger utilities
    'setup_logger',
    'setup_file_logger',
    'get_logger',
    # Config classes
    'Config',
    'FeatureConfig',
    'DatasetConfig',
    'AugmentationConfig',
    'NoiseConfig',
    'MixupConfig',
    'LossConfig',
    'ModelConfig',
    'TrainingConfig',
    'load_config',
    'save_config',
    'get_default_config'
]
