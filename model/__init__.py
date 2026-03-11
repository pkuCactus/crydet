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

__all__ = [
    'CryTransformer',
    'create_model', 'create_model_from_variant',
    'list_models', 'get_model_info', 'print_model_summary',
    'MODEL_CONFIGS'
]
