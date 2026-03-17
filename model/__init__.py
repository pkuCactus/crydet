"""
Transformer models for Baby Cry Detection

Supports multiple model variants for different deployment scenarios:
- Large: High-performance for server/cloud
- Medium: Lightweight for edge devices (Raspberry Pi, Edge TPU)
- Tiny: Ultra-lightweight for MCU (Cortex-M, ESP32)
"""

from .transformer import CryTransformer
from .variants import (
    create_model, create_model_from_variant,
    list_models, get_model_info, print_model_summary,
    MODEL_CONFIGS
)
from .loss import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    CombinedLoss,
    create_loss
)
from .ema import ExponentialMovingAverage
from .scheduler import WarmupCosineScheduler, LinearWarmupPolyDecayScheduler
from .distributed import (
    setup_distributed, cleanup_distributed,
    is_distributed, get_rank, get_world_size,
    is_main_process, barrier, all_reduce
)

__all__ = [
    'CryTransformer',
    'create_model', 'create_model_from_variant',
    'list_models', 'get_model_info', 'print_model_summary',
    'MODEL_CONFIGS',
    'FocalLoss', 'LabelSmoothingCrossEntropy', 'CombinedLoss', 'create_loss',
    'ExponentialMovingAverage',
    'WarmupCosineScheduler', 'LinearWarmupPolyDecayScheduler',
    'setup_distributed', 'cleanup_distributed',
    'is_distributed', 'get_rank', 'get_world_size',
    'is_main_process', 'barrier', 'all_reduce'
]
